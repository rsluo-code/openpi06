# ---------------------------------------------------------
# train_pytorch_valuenet.py
# Training script for ValueNetPytorch (Eq.1)
# Mirrors scripts/train_pytorch.py but without PI0 loss.
# ---------------------------------------------------------

import os
import time
import shutil
import logging
import platform
import dataclasses
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import safetensors.torch
import tqdm
import wandb
import gc

import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.shared.normalize as _normalize
import openpi
import jax

import openpi

print("="*25,"rsluo_test","="*25)
print("openpi at12:", openpi.__file__)

from openpi.models_pytorch.valuenet_pytorch import ValueNetPytorch

from openpi.models_pytorch.some_func import print_parameter_stats,load_param_from_ckpt_dir



import sys
python_path = sys.executable
print("当前使用的Python解释器路径：", python_path)
import dlimp as dl

print("="*25,"rsluo_test2","="*25)
print("dlimp at:", dl.__file__)
print("Has DLataset?", hasattr(dl, "DLataset"))



# -----------------------------------------
# (Same logging/DDP helpers from train_pytorch.py)
# -----------------------------------------
def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


# -----------------------------------------
# Returns (Eq.1 uses real return distribution)
# -----------------------------------------
def compute_returns(rewards, dones, gamma=1.0):
    """
    Standard return computation for offline RL (no discount γ=1).
    rewards: [B, T]
    dones:   [B, T]
    """
    B, T = rewards.shape
    R = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        running = rewards[:, t] + running * (1 - dones[:, t])
        R[:, t] = running
    return R


# -----------------------------------------
# Build dataloader (identical to PI0 pipeline)
# -----------------------------------------
def build_datasets(config):
    dl = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return dl, dl.data_config()


# -----------------------------------------
# Checkpoint I/O
# -----------------------------------------
def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    if not is_main:
        return
    if global_step % config.save_interval != 0:
        return

    ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")

    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")

    metadata = {"global_step": global_step, "config": dataclasses.asdict(config), "timestamp": time.time()}
    torch.save(metadata, tmp_dir / "metadata.pt")

    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / data_config.asset_id, data_config.norm_stats)

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)

    logging.info(f"[ValueNet] Saved checkpoint at step {global_step}")



# -----------------------------------------
# Training Loop (core replacement of PI0 → ValueNet)
# -----------------------------------------
def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Prepare checkpoint dir
    if config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # wandb
    if is_main:
        # wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        wandb.init(mode="disabled")
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )
    # dataset
    loader, data_config = build_datasets(config)

    # ------------------------------
    # Build ValueNet
    # ------------------------------
    model_cfg = config.model
    # model = PI06Pytorch(model_cfg).to(device)
    model = ValueNetPytorch(model_cfg, num_bins=201, image_keys=data_config.image_keys).to(device)

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    # if config.aux_pytorch_weight_path is not None and rank == 0:
    if config.pytorch_weight_path is not None :
        load_param_from_ckpt_dir(model, config.pytorch_weight_path,"valuenet")
    
    if use_ddp:
        dist.barrier()  # 确保所有 rank 在同一状态
    print_parameter_stats(model,"value_net")
    if use_ddp:
        # find_unused_parameters=True 让 DDP 接受“有一部分参数没参与 loss / 没梯度”的情况（比如被冻结的 PaliGemma）
        model = DistributedDataParallel(
            model,
            device_ids=[device.index], 
            find_unused_parameters=True)

    global_step = 0
    if is_main:
        logging.info("🚀 Starting ValueNet training...")

    pbar = tqdm.tqdm(total=config.num_train_steps, disable=not is_main)

    for param in model.parameters():
        param.requires_grad = False

    NUM_LANG = 26
    METRICS = ["Rt", "RtN", "value_t", "value_tN", "pretrain_At", "finetuning_At"]
    # 算出来的以多少位小数点作为计算 decimal places：小数位数
    DECIMAL_PLACES = 3
    # stats[lang_id][metric][bucket_int] = count
    stats = {
        lang_id: {m: {} for m in METRICS}
        for lang_id in range(1, NUM_LANG + 1)
    }
    def to_bucket_int(x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor 任意 shape
        返回：int tensor，表示 round(x,2)*100
        例如 -0.37 -> -37
        """
        return torch.round(x * (10.0**DECIMAL_PLACES)).to(torch.int32)

    def update_stats_bucketed(stats, lang_ids, Rt, RtN, value_t, value_tN):
        """
        lang_ids: list[int] 长度 B，取值 1..NUM_LANG
        Rt, RtN, value_t, value_tN: torch.Tensor shape [B]（在任意 device 都行）
        """
        # 两种 At
        pretrain_At = Rt - value_t
        finetuning_At = RtN + value_tN - value_t

        # 全部转成两位小数桶（int）
        Rt_b   = to_bucket_int(Rt).detach().cpu()
        RtN_b  = to_bucket_int(RtN).detach().cpu()
        vt_b   = to_bucket_int(value_t).detach().cpu()
        vtn_b  = to_bucket_int(value_tN).detach().cpu()
        pAt_b  = to_bucket_int(pretrain_At).detach().cpu()
        fAt_b  = to_bucket_int(finetuning_At).detach().cpu()
        B = len(lang_ids)

        for i in range(B):
            lang = int(lang_ids[i])
            if lang < 1 or lang > NUM_LANG:
                continue

            def inc(metric, bucket_int):
                d = stats[lang][metric]
                k = int(bucket_int)
                d[k] = d.get(k, 0) + 1

            inc("Rt", Rt_b[i])
            inc("RtN", RtN_b[i])
            inc("value_t", vt_b[i])
            inc("value_tN", vtn_b[i])
            inc("pretrain_At", pAt_b[i])
            inc("finetuning_At", fAt_b[i])

    def topk_threshold_from_buckets(buckets, top_ratio=0.3):
        """
        buckets: dict[value_str -> count]
        返回：
        threshold_value_str（'%.2f'）
        total_count
        topk_count
        """
        if not buckets:
            return None, 0, 0

        total = sum(buckets.values())
        k = max(1, int(total * top_ratio))

        # 按 value 从大到小
        items = sorted(buckets.items(), key=lambda kv: float(kv[0]), reverse=True)

        picked = 0
        threshold = None
        for v_str, cnt in items:
            take = min(cnt, k - picked)
            if take <= 0:
                break
            picked += take
            threshold = v_str
            if picked >= k:
                break

        return threshold, total, picked

    def print_top20pct_from_buckets(stats, metric, top_ratio=0.3,local_rank=0):
        for lang in range(1, NUM_LANG):
            buckets = stats[lang].get(metric, {})
            if not buckets:
                continue

            threshold, total, topk_cnt = topk_threshold_from_buckets(
                buckets, top_ratio=top_ratio
            )
            # total 决定规模
            # topk_cnt 决定你要多少
            # threshold 决定“谁能进来”
            if threshold is None:
                continue
            th = (float)(threshold /  (10.0**DECIMAL_PLACES))
            print(
                f"[rank {local_rank}][lang {lang:02d}] {metric}: "
                f"total={total}, "
                f"top{int(top_ratio*100)}%={topk_cnt}, "
                f"threshold={th:.4f}, "
            )

    import os 
    import csv
    def export_bucket_csvs(stats, metrics_to_save, out_dir="bucket_csvs"):
        """
        stats[lang_id][metric] = { value_str: count }  # value_str 是 '%.2f'
        metrics_to_save: List[str] 例如 ["value_tN", "pretrain_At", "finetuning_At"]
        """
        os.makedirs(out_dir, exist_ok=True)

        for lang_id in sorted(stats.keys()):
            for metric in metrics_to_save:
                buckets = stats[lang_id].get(metric, {})
                if not buckets:
                    continue

                out_path = os.path.join(out_dir, f"lang_{lang_id:02d}__{metric}.csv")
                with open(out_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["value", "count"])  # 表头

                    # 按 value 数值从小到大排序（你也可以改成 reverse=True）
                    for v_str, cnt in sorted(buckets.items(), key=lambda kv: float(kv[0])):
                        w.writerow([v_str, cnt])

                print(f"[saved] {out_path}  buckets={len(buckets)}")

    def export_top20_thresholds_txt(
        stats,
        out_path="thresholds_top20.txt",
        value_metric="value_t",
        valueN_metric="value_tN",
        pretrain_metric="pretrain_At",
        finetune_metric="finetuning_At",
        top_ratio=0.3
    ):
        """
        输出格式：
        idx,value_N_30,pretrain_at_30,finetuning_At_30
        """
        with open(out_path, "w") as f:
            f.write("idx,value_30,value_N_30,pretrain_at_30,finetuning_At_30(threshold, total, topk_cnt)\n")

            for lang_id in sorted(stats.keys()):
                v_th = topk_threshold_from_buckets(stats[lang_id].get(value_metric, {}), top_ratio=top_ratio)
                vN_th = topk_threshold_from_buckets(stats[lang_id].get(valueN_metric, {}), top_ratio=top_ratio)
                p_th  = topk_threshold_from_buckets(stats[lang_id].get(pretrain_metric, {}), top_ratio=top_ratio)
                f_th  = topk_threshold_from_buckets(stats[lang_id].get(finetune_metric, {}), top_ratio=top_ratio)

                # 没有数据就写 NA
                v_th = v_th if v_th is not None else "NA"
                vN_th = vN_th if vN_th is not None else "NA"
                p_th  = p_th  if p_th  is not None else "NA"
                f_th  = f_th  if f_th  is not None else "NA"

                f.write(f"{lang_id},{v_th},{vN_th},{p_th},{f_th}\n")

        print(f"[saved] {out_path}")

    if use_ddp:
        model.module.training=False
    else:
        model.training=False

    STATS_SAVE_PATH = f"/home/rsluo/codes/openpi06/z_bucket_csvs/20260204_5item_8dim/local_rank_{local_rank}"
    metrics_to_save = ["value_t","value_tN", "pretrain_At", "finetuning_At"]  
    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))
        for observation, observation_tN, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure, _ in loader:
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation) 
            observation_tN = jax.tree.map(lambda x: x.to(device), observation_tN) 
            with torch.no_grad():
                if use_ddp:
                    forward_cal_At_result = model.module.forward_cal_At(observation,observation_tN, step_index, episode_length, language_instruction_max_len)  # [B, 201]
                else:
                    forward_cal_At_result = model.forward_cal_At(observation,observation_tN, step_index, episode_length, language_instruction_max_len)  # [B, 201]


            Rt       = forward_cal_At_result["Rt"]        # [B]
            RtN      = forward_cal_At_result["RtN"]       # [B]
            value_t  = forward_cal_At_result["value_t"]   # [B]
            value_tN = forward_cal_At_result["value_tN"]  # [B]
            lang_ids = language_instruction_index.detach().cpu().tolist()  # [B] in 1..NUM_LANG
            update_stats_bucketed(stats, lang_ids, Rt, RtN, value_t, value_tN)
            # import pdb; pdb.set_trace()

            if global_step %  config.log_interval  == 0:
                print_top20pct_from_buckets(stats, "pretrain_At", top_ratio=0.3,local_rank=local_rank)
                print_top20pct_from_buckets(stats, "finetuning_At", top_ratio=0.3,local_rank=local_rank)
                print_top20pct_from_buckets(stats, "value_t", top_ratio=0.3,local_rank=local_rank)
                print_top20pct_from_buckets(stats, "value_tN", top_ratio=0.3,local_rank=local_rank)
                export_bucket_csvs(stats, metrics_to_save, out_dir=STATS_SAVE_PATH)
                export_top20_thresholds_txt(
                    stats,
                    out_path=f"{STATS_SAVE_PATH}/thresholds_top30.txt",
                    value_metric="value_t",     
                    valueN_metric="value_tN",      
                    pretrain_metric="pretrain_At",
                    finetune_metric="finetuning_At",
                    top_ratio=0.3
                )

            if is_main and global_step!= 0 and global_step %  config.log_interval  == 0:
                pbar.update( config.log_interval )
            global_step += 1
    pbar.close()

    export_bucket_csvs(stats, metrics_to_save, out_dir=STATS_SAVE_PATH)

    export_top20_thresholds_txt(
        stats,
        out_path=f"{STATS_SAVE_PATH}/thresholds_top30.txt",
        value_metric="value_t",     
        valueN_metric="value_tN",      
        pretrain_metric="pretrain_At",
        finetune_metric="finetuning_At",
        top_ratio=0.3
    )



    if is_main:
        wandb.finish()
    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
