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

from openpi.models_pytorch.pi06_pytorch import PI06Pytorch
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



    model_cfg = config.model
    model = PI06Pytorch(model_cfg)
    model= model.to(device)

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    

    enable_gradient_checkpointing = False
    model.gradient_checkpointing_disable()

    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        load_param_from_ckpt_dir(model,config.pytorch_weight_path,"PI06")
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    dist.barrier()  # 确保所有 rank 在同一状态

    print_parameter_stats(model,"pi06_total")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=True #world_size >= 8,  # Enable for 8+ GPUs
        )



    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    
    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if is_main:
        logging.info("🚀 Starting PI06 training...")


    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval


    pbar = tqdm.tqdm(total=config.num_train_steps, disable=not is_main)

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        # for observation, actions in loader:
        for observation,obs_N, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure,language_instruction_at_30precent in loader:
            import pdb; pdb.set_trace()

            # start_time = time.time()
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break
            # print("rsluo======DEBUG observation type:", type(observation))
            # import pdb; pdb.set_trace()
            
            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901
            # if local_rank == 0 or local_rank == 1:
            #     print("rank:", local_rank, "actions:", actions)
            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)


            losses = model(observation,obs_N, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure )
            # if local_rank == 0:
            #     print("forward_time:", time.time()-forward_time)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
                
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            # torch.cuda.synchronize()
            # opt_time = time.time()
            optim.step()
            optim.zero_grad(set_to_none=True)
            # if local_rank == 0:
            #     print("optimize_time:", time.time()-opt_time)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)


    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
