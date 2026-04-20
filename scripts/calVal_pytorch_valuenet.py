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


import torch
import matplotlib.pyplot as plt

class R012PlotterProgress:
    def __init__(self, v_min=-1.0, v_max=0.0, num_bins=201):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_bins = int(num_bins)

        # 用你描述的“201 个中心点覆盖 [-1,0]”
        # bin_width = 1/(num_bins-1) = 0.005
        self.bin_centers = torch.linspace(self.v_min, self.v_max, steps=self.num_bins)

        # step_index -> (R0, R1, R2)
        self.buffer = {}

    @torch.no_grad()
    def update(self, step_index, episode_length, logits=None, prob=None):
        """
        step_index:      [B] or scalar
        episode_length:  [B] or scalar
        logits:          [B,201] (optional)
        prob:            [B,201] (optional, if already softmaxed)
        """
        assert (logits is not None) or (prob is not None), "请提供 logits 或 prob"

        # ---- 统一到 CPU，便于收集与绘图 ----
        step_index = torch.as_tensor(step_index).detach().cpu().long().view(-1)
        B = step_index.numel()

        episode_length = torch.as_tensor(episode_length).detach().cpu().float()
        if episode_length.numel() == 1:
            episode_length = episode_length.expand(B)
        else:
            episode_length = episode_length.view(-1).float()
            assert episode_length.numel() == B, "episode_length 若为向量，必须与 step_index 同长度"

        if prob is None:
            prob = torch.softmax(logits.detach().cpu(), dim=-1)
        else:
            prob = prob.detach().cpu()

        assert prob.shape[0] == B and prob.shape[1] == self.num_bins, \
            f"prob 形状应为 [B,{self.num_bins}]，但得到 {tuple(prob.shape)}"

        bc = self.bin_centers  # [201] on CPU

        for i in range(B):
            s = int(step_index[i].item())
            if s in self.buffer:
                print(f"[dup step_index] {s} -> ignored")
                continue

            L = float(episode_length[i].item())
            if L <= 0:
                # 防止除0
                print(f"[bad episode_length] step {s}, L={L} -> ignored")
                continue

            # -------- R0：进度占比映射到 [-1,0] --------
            ratio = ( (L - float(s)) / L )
            ratio = max(0.0, min(1.0, ratio))  # clamp 到 [0,1]
            R0 = -ratio

            # -------- R2：argmax(prob) 的 bin center --------
            pred_idx = int(torch.argmax(prob[i]).item())
            R2 = float(bc[pred_idx].item())

            # -------- R1：期望 sum(prob * bin_center) --------
            R1 = float((prob[i] * bc).sum().item())

            self.buffer[s] = (R0, R1, R2)

    def save(self, out_path="r012.png"):
        assert len(self.buffer) > 0, "没有数据可画，请先 update()"

        steps = sorted(self.buffer.keys())
        R0 = [self.buffer[s][0] for s in steps]
        R1 = [self.buffer[s][1] for s in steps]
        R2 = [self.buffer[s][2] for s in steps]

        plt.figure(figsize=(14, 4))
        plt.plot(steps, R0, label="R0 = -(L-step)/L (target)", linewidth=2)
        plt.plot(steps, R1, label="R1 = E[value] = sum(p*center)", linewidth=2)
        plt.plot(steps, R2, label="R2 = center[argmax(p)]", linewidth=2)

        plt.xlabel("step_index")
        plt.ylabel("value in [-1,0]")
        plt.ylim(self.v_min, self.v_max)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[saved] {out_path}")

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
    model = ValueNetPytorch(model_cfg, num_bins=201).to(device)

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


    if use_ddp:
        model.module.training=False
    else:
        model.training=False

    plotter = R012PlotterProgress(v_min=-1.0, v_max=0.0, num_bins=201)

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))
        for observation, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure in loader:
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)
            B, T, _ = actions.shape  # actions: [B, T, act_dim]
            step_index_t = torch.as_tensor(step_index, device=device, dtype=torch.float32)        # [B]，范围 0..L-1
            episode_length_t = torch.as_tensor(episode_length, device=device, dtype=torch.float32)  # [B]，每个都是 L
            lang_max_len_t = episode_length_t
            denom = torch.clamp(lang_max_len_t-1, min=1.0)  # 替换原来的denom计算
            ratio = (episode_length_t - (step_index_t + 1)) / denom
            Rt = -1* torch.clamp(ratio, min=0.0, max=1.0)   # 核心公式修改
            logits = model(observation)  # [B, 201]
            num_bins = logits.shape[-1]  # 理论上 = 201
            v_min, v_max = -1.0, 0.0
            bin_edges = torch.linspace(v_min, v_max, num_bins+1, device=device)
            idx = torch.bucketize(Rt.contiguous(), bin_edges) - 1  # [B]
            idx = idx.clamp(0, num_bins - 1)
            with torch.no_grad():
                if use_ddp:
                    logits = model.module.forward(observation) 
                else:
                    logits = model.forward(observation) 
            probs = torch.softmax(logits, dim=-1)  # [B,201]
            # import pdb; pdb.set_trace()

            plotter.update(step_index=step_index_t,
                        episode_length=episode_length_t,
                        logits=logits)

            if is_main and global_step % 1 == 0:
                print(f"global_step{global_step}")
            global_step += 1
            pbar.update(1)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter.save(f"./z_experimental_result/valuenet_val_20260116mod_2b_11wstep_exemod_noeximg_香蕉_{ts}.png")

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
