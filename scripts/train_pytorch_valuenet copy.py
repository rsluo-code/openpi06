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
from openpi.models_pytorch.pi06_pytorch_bak import PI06Pytorch

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


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None

    
# -----------------------------------------
# Training Loop (core replacement of PI0 → ValueNet)
# -----------------------------------------
def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)


    # Initialize checkpoint directory and wandb
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")



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
    model.gradient_checkpointing_disable() # 执行这个会显存up，推理fast

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    # if config.aux_pytorch_weight_path is not None and rank == 0:
    if config.pytorch_weight_path is not None :
        load_param_from_ckpt_dir(model, config.pytorch_weight_path,f"valuenet_{rank}")
    
    if use_ddp:
        dist.barrier()  # 确保所有 rank 在同一状态


    print_parameter_stats(model,"value_net")


    if use_ddp:
        # find_unused_parameters=True 让 DDP 接受“有一部分参数没参与 loss / 没梯度”的情况（比如被冻结的 PaliGemma）
        model = DistributedDataParallel(
            model,
            device_ids=[device.index], 
            find_unused_parameters=True)

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
        # for observation, _, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure, _ in loader:
        for observation, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure in loader:
            if global_step >= config.num_train_steps:
                break


            observation = jax.tree.map(lambda x: x.to(device), observation)

            B, T, _ = actions.shape  # actions: [B, T, act_dim]


            # 是 numpy array，转成 torch tensor 放到 device
            step_index_t = torch.as_tensor(step_index, device=device, dtype=torch.float32)        # [B]，范围 0..L-1
            episode_length_t = torch.as_tensor(episode_length, device=device, dtype=torch.float32)  # [B]，每个都是 L
            lang_max_len_t = torch.as_tensor(language_instruction_max_len, device=device, dtype=torch.float32)

            denom = torch.clamp(lang_max_len_t, min=1.0)
            # --------------------------------------------------
            # 核心修改：替换Rt的计算逻辑
            # 公式：ratio = (episode_length_t - step_index_t) / language_instruction_max_len
            # Rt = -ratio → 占比越大（ratio→1），Rt→-1；占比越小（ratio→0），Rt→0
            # --------------------------------------------------
            # 处理边界：防止language_instruction_max_len为0导致除0，最小设为1.0
            denom = torch.clamp(lang_max_len_t, min=1.0)  # 替换原来的denom计算
            # 计算核心占比：(episode_length - step_index) / language_instruction_max_len
            ratio = (episode_length_t - (step_index_t + 1)) / denom
            # 映射到[-1, 0]：ratio∈[0,1] → Rt∈[-1, 0]（如果ratio超出[0,1]，用clamp限制）
            Rt = -1* torch.clamp(ratio, min=0.0, max=1.0)   # 核心公式修改
            # 3. Forward：ValueNet 输出 201-bin logits
            logits = model(observation)  # [B, 201]

            # 4. 把连续 value Rt ∈ [-1, 0] 映射成 201 个 bin 的离散 label
            num_bins = logits.shape[-1]  # 理论上 = 201
            v_min, v_max = -1.0, 0.0

            # edges: [num_bins]，比如 num_bins=201 个类别：
            # 边界点个数 = num_bins = 201
            # 所以需要202个边界num_bins+1
            bin_edges = torch.linspace(v_min, v_max, num_bins+1, device=device)

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

           
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if is_main:
                wandb.log({"loss_value": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
            if is_main and global_step % 5 == 0:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)  # [B,201]

                    print(
                        f"[logits check step {global_step}] "
                        f"logits min/max = ({logits.min().item():.4f}, {logits.max().item():.4f})  "
                        f"probs min/max = ({probs.min().item():.6f}, {probs.max().item():.6f})"
                    )

                    print(
                        f"  probs sum (first 5 samples) = "
                        f"{probs[:5].sum(dim=-1).detach().cpu()}"
                    )
                    num_show =5
                    num_show = min(num_show, effective_batch_size)

                    # 在 batch 内随机抽样
                    rand_idx = torch.randperm(effective_batch_size)[:num_show]

                    topk = torch.topk(probs, k=5, dim=-1)

                    for i, idx_i in enumerate(rand_idx):
                        idx_i = idx_i.item()
                        print(
                            f" sample {i} (batch idx {idx_i}): "
                            f"true idx={idx[idx_i].item()}, "
                            f"topk idx={topk.indices[idx_i].tolist()}, "
                            f"topk prob={[round(p, 4) for p in topk.values[idx_i].tolist()]}"
                        )

                print(
                    f"[step {global_step}] "
                    f"loss={loss.item() if 'loss' in locals() else float('nan'):.4f} "
                    f"Rt.shape={tuple(Rt.shape)} idx.shape={tuple(idx.shape)} "
                    f"logits.shape={tuple(logits.shape)} "
                    f"step_index.shape={tuple(step_index_t.shape)} episode_length.shape={tuple(episode_length_t.shape)}\n"
                    f"Rt[:5]={Rt[:5].squeeze(-1).detach().cpu()} \n"
                    f"idx[:5]={idx[:5].squeeze(-1).detach().cpu()} \n"
                    f"step_index[:5]={step_index_t[:5].squeeze(-1).detach().cpu()} \n"
                    f"episode_length[:5]={episode_length_t[:5].squeeze(-1).detach().cpu()} \n"
                    f"lang_max_len[:5]={lang_max_len_t[:5].squeeze(-1).detach().cpu()} \n"
                    f"Rt.min/max=({Rt.min().item():.4f},{Rt.max().item():.4f}) "
                    f"idx.min/max=({idx.min().item()},{idx.max().item()})",
                    f"language_instruction_index[:5]=({language_instruction_index[:5].squeeze(-1).detach().cpu()})",
                    
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
