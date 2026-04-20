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


from openpi.models_pytorch.valuenet_pytorch import ValueNetPytorch

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


def count_parameters(model: torch.nn.Module):
    """返回 (总参数量, 可训练参数量)。"""
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


def print_parameter_stats(model: torch.nn.Module):
    """打印整体参数量 + 按 top-level module 拆分的参数量。"""
    from collections import defaultdict


    # 如果是 DDP，拿里面的真实模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    total, trainable = count_parameters(model)
    print(f"[ValueNet] Total params: {total:,} "
                f"({total/1e6:.2f}M, {total/1e9:.3f}B)")
    print(f"[ValueNet] Trainable params: {trainable:,} "
                f"({trainable/1e6:.2f}M, {trainable/1e9:.3f}B)")

    # 按第一层前缀分组统计，例如:
    #   paligemma_with_expert.xxx
    #   value_head.xxx
    #   state_mlp.xxx
    group_total = defaultdict(int)
    group_trainable = defaultdict(int)

    for name, p in model.named_parameters():
        top = name.split(".")[0]  # 最前面的那一段
        num = p.numel()
        group_total[top] += num
        if p.requires_grad:
            group_trainable[top] += num

    print("[ValueNet] Per-top-module param stats:")
    for g in sorted(group_total.keys()):
        gt = group_total[g]
        gtr = group_trainable[g]
        print(
            f"  - {g:<24} "
            f"total={gt:,} ({gt/1e6:.2f}M), "
            f"trainable={gtr:,} ({gtr/1e6:.2f}M)"
        )

    # 你关心的 value_head 单独再报一遍
    vh_params = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("value_head")
    )
    print(
        f"[ValueNet] value_head params only: {vh_params:,} "
        f"({vh_params/1e6:.2f}M)"
    )

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

    # dataset
    loader, data_config = build_datasets(config)

    # ------------------------------
    # Build ValueNet
    # ------------------------------
    model_cfg = config.model
    model = ValueNetPytorch(model_cfg, num_bins=201).to(device)


    def load_pi05_prefix_only(model, ckpt_dir: str):
        """
        只从 Pi05 ckpt 里加载 PaliGemma 的前缀编码部分到 ValueNet 里：
        - 加载 paligemma_with_expert.paligemma.*
        - 不加载 value_head（本来就不存在）
        - 不加载 action_in_proj / time_mlp_* / 其他顶层东西
        """


        ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
        print(f"[ValueNet] Loading Pi05 prefix-only weights from: {ckpt_path}")

        from safetensors.torch import load_file
        # 1. 读 ckpt
        src_sd = load_file(ckpt_path, device="cpu")

        # 2. 取出当前模型的 state_dict（注意 DDP 的情况）
        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        target = model.module if is_ddp else model
        tgt_sd = target.state_dict()

        # 3. 只保留 paligemma_with_expert.paligemma.* 并且在当前模型中存在且 shape 一致的参数
        filtered_sd = {}
        for k, v in src_sd.items():
            # 只要 PaliGemma 主干
            if not k.startswith("paligemma_with_expert.paligemma."):
                continue

            # 当前 ValueNet 里必须也有同名参数，并且 shape 一致
            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                filtered_sd[k] = v

        print(f"[ValueNet] Will load {len(filtered_sd)} parameters into paligemma_with_expert.paligemma")

        # 4. 只用 filtered_sd 来覆盖（strict=False 可以忽略没有加载到的 value_head 等）
        missing, unexpected = target.load_state_dict(filtered_sd, strict=False)

        print(f"[ValueNet] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[ValueNet] missing keys (正常的通常只包括 value_head.*): {missing}")
        if unexpected:
            print(f"[ValueNet] unexpected keys from ckpt (已忽略): {unexpected}")
    
    # ⭐ 如果指定了 pytorch_weight_path，则从 Pi05 checkpoint 加载“前缀权重”
    if config.pytorch_weight_path is not None and dist.get_rank() == 0:
        load_pi05_prefix_only(model, config.pytorch_weight_path)
    dist.barrier()  # 确保所有 rank 在同一状态


    print_parameter_stats(model)


    if use_ddp:
        # find_unused_parameters=True 让 DDP 接受“有一部分参数没参与 loss / 没梯度”的情况（比如被冻结的 PaliGemma）
        model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)

    # ------------------------------
    # Optimizer & LR schedule
    # ------------------------------
    warmup = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def lr_sched(step):
        if step < warmup:
            init_lr = peak_lr / (warmup + 1)
            return init_lr + (peak_lr - init_lr) * (step / warmup)
        prog = min(1.0, (step - warmup) / (decay_steps - warmup))
        return end_lr + (peak_lr - end_lr) * 0.5 * (1 + np.cos(np.pi * prog))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if is_main:
        logging.info("🚀 Starting ValueNet training...")

    pbar = tqdm.tqdm(total=config.num_train_steps, disable=not is_main)

    # ------------------------------
    # Main training loop
    # ------------------------------
    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        # for observation, actions in loader:   # actions contains actions, rewards, dones, etc.
        for observation, actions, step_index, episode_length in loader:

            if global_step >= config.num_train_steps:
                break
            # print("rsluo======DEBUG observation type:", type(observation))

            import pdb; pdb.set_trace() #debug默认停的一个断点

            observation = jax.tree.map(lambda x: x.to(device), observation)

            B, T, _ = actions.shape  # actions: [B, T, act_dim]


            # 是 numpy array，转成 torch tensor 放到 device
            step_index_t = torch.as_tensor(step_index, device=device, dtype=torch.float32)        # [B]，范围 0..L-1
            episode_length_t = torch.as_tensor(episode_length, device=device, dtype=torch.float32)  # [B]，每个都是 L

            denom = torch.clamp(episode_length_t - 1.0, min=1.0)   # L=1 时防止除 0
            # 公式：R = -1 + (step_index_t - 1) / (L - 1)
            # Rt = -1.0 + (step_index_t - 1.0) / denom               # [B] ∈ [-1, 0]
            a = -1.0
            b = 0.0
            Rt = a + step_index_t / denom * (b - a) 

            # --------------------------------------------------
            # 3. Forward：ValueNet 输出 201-bin logits
            # --------------------------------------------------
            logits = model(observation)  # [B, 201]

            # --------------------------------------------------
            # 4. 把连续 value Rt ∈ [-1, 0] 映射成 201 个 bin 的离散 label
            #
            #    这里我们按「等宽分桶」：
            #        bin_edges: [num_bins+1]，覆盖 [-1, 0]
            #        第 i 个 bin 表示 [edge_i, edge_{i+1})
            #
            #    你之前说的：
            #       (-0.05, 0) → 第 201 类
            #       (-0.10, -0.05) → 第 200 类
            #       ...
            #    这个是 0.05 步长的例子（20 个 bin），
            #    我们现在用 201 个 bin，就是把 [-1, 0] 更细地均匀切成 201 份。
            # --------------------------------------------------
            num_bins = logits.shape[-1]  # 理论上 = 201
            v_min, v_max = -1.0, 0.0

            # edges: [num_bins+1]，比如 num_bins=201 → [202] 个边界   这是把区间  [−1,0]均匀切成 201 段：
            # 边界点个数 = num_bins + 1 = 202
            # 间隔宽度：1/201 =0.004975
            bin_edges = torch.linspace(v_min, v_max, num_bins + 1, device=device)

            # bucketize 返回 0..num_bins 之间的 index，表示落在哪两个 edge 之间
            # 减 1 后得到 0..num_bins-1 的 class index
            idx = torch.bucketize(Rt.contiguous(), bin_edges) - 1  # [B]
            idx = idx.clamp(0, num_bins - 1)

            # --------------------------------------------------
            # 5. CE loss：logits 对应 201 分类，target 是离散 bin index
            # --------------------------------------------------
            loss = torch.nn.functional.cross_entropy(logits, idx)

            # ---- Backward ----
            for pg in optimizer.param_groups:
                pg["lr"] = lr_sched(global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if is_main:
                wandb.log({"loss_value": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
            if is_main and global_step % 10 == 0:
                print(
                    f"[step {global_step}] "
                    f"loss={loss.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"Rt[:5]={Rt[:5]} "
                    f"idx[:5]={idx[:5]} "
                    f"step_index[:5]={None if step_index is None else step_index[:5]} "
                    f"episode_length[:5]={None if episode_length is None else episode_length[:5]} ",
                    flush=True,
                )
            save_checkpoint(model, optimizer, global_step, config, is_main, data_config)

            global_step += 1
            pbar.update(1)

    pbar.close()
    if is_main:
        wandb.finish()
    cleanup_ddp()


def main():
    

    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
