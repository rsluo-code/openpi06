"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
# 基础库
import torch
import os
import datetime  # 用于生成唯一的日志目录名
# TensorBoard核心库
from torch.utils.tensorboard import SummaryWriter

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

def init_tensorboard():
    # ===================== 初始化TensorBoard =====================
    # 1. 创建日志目录（按时间命名，避免不同训练的日志覆盖）
    # 日志根目录：runs（会自动创建在你运行脚本的目录下）
    log_root = "runs"
    os.makedirs(log_root, exist_ok=True)  # 确保目录存在，不存在则创建

    # 2. 生成唯一的实验名（时间戳）
    exp_name = f"flow_matching_awr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(log_root, exp_name)

    # 3. 初始化SummaryWriter（核心对象，负责写入日志）
    writer = SummaryWriter(log_dir=log_dir)

    # 打印日志目录（方便后续启动TensorBoard时确认路径）
    print(f"TensorBoard日志已保存到：{log_dir}")
    return writer

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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


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


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # 1. 初始化分布式环境（单机多卡/多机多卡通用）
    rank = 0
    if use_ddp:
        rank = dist.get_rank()  # 获取当前进程的rank
    print("多卡训练：" , rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
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

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    if rank==0:
        tensorboard_writer = init_tensorboard()
    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    # model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    from openpi.models_pytorch.valuenet_pytorch import ValueNetPytorch
    model = ValueNetPytorch(model_cfg, num_bins=201, image_keys=data_config.image_keys).to(device)


    enable_gradient_checkpointing = False
    model.gradient_checkpointing_disable() # 执行这个会显存up，推理fast

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=True #world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        from openpi.models_pytorch.some_func import print_parameter_stats,load_param_from_ckpt_dir
        load_param_from_ckpt_dir(model, config.pytorch_weight_path,f"valuenet_{rank}")
        print_parameter_stats(model,"value_net")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

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
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        # logging.info(f"Resumed training from step {global_step}")
        logging.info(f"Resumed training from step {global_step},path = {config.checkpoint_dir}/{global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        # 4. 把连续 value Rt ∈ [-1, 0] 映射成 201 个 bin 的离散 label
        num_bins = 201  # 理论上 = 201
        v_min, v_max = -1.0, 0.0
        bin_edges = torch.linspace(v_min, v_max, num_bins+1, device=device)
        # for observation, actions,_ ,_ in loader:
        for observation, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure in loader:
            
            # start_time = time.time()
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break
            
            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            
            B, T, _ = actions.shape  # actions: [B, T, act_dim]
            

            # 是 numpy array，转成 torch tensor 放到 device
            step_index_t = torch.as_tensor(step_index, device=device, dtype=torch.float32)        # [B]，范围 0..L-1
            episode_length_t = torch.as_tensor(episode_length, device=device, dtype=torch.float32)  # [B]，每个都是 L
            # lang_max_len_t = episode_length_t
            lang_max_len_t = torch.as_tensor(language_instruction_max_len, device=device, dtype=torch.float32)
            
            # denom = torch.clamp(lang_max_len_t, min=1.0)
            # --------------------------------------------------
            # 核心修改：替换Rt的计算逻辑
            # 公式：ratio = (episode_length_t - step_index_t) / language_instruction_max_len
            # Rt = -ratio → 占比越大（ratio→1），Rt→-1；占比越小（ratio→0），Rt→0
            # --------------------------------------------------
            # 处理边界：防止language_instruction_max_len为0导致除0，最小设为1.0


            denom = torch.clamp(lang_max_len_t-1, min=1.0)  # 替换原来的denom计算

            # 计算核心占比：(episode_length - step_index) / language_instruction_max_len
            ratio = (episode_length_t - (step_index_t + 1)) / denom
            # 映射到[-1, 0]：ratio∈[0,1] → Rt∈[-1, 0]（如果ratio超出[0,1]，用clamp限制）
            Rt = -1* torch.clamp(ratio, min=0.0, max=1.0)   # 核心公式修改

            one_forward_start_time = time.time()
            # 3. Forward：ValueNet 输出 201-bin logits
            logits = model(observation)  # [B, 201]
            one_forward_time = time.time() - one_forward_start_time
            
            

            # edges: [num_bins]，比如 num_bins=201 个类别：
            # 边界点个数 = num_bins = 201
            # 所以需要202个边界num_bins+1
            

            # bucketize 返回 0..num_bins 之间的 index，表示落在哪两个 edge 之间
            # 减 1 后得到 0..num_bins-1 的 class index
            idx = torch.bucketize(Rt.contiguous(), bin_edges) - 1  # [B]
            idx = idx.clamp(0, num_bins - 1)

            # --------------------------------------------------
            # 5. CE loss：logits 对应 201 分类，target 是离散 bin index
            # --------------------------------------------------
            
            loss = torch.nn.functional.cross_entropy(logits, idx)

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            

            one_lossbackward_start_time = time.time()
            loss.backward()
            one_lossbackward_time = time.time() - one_lossbackward_start_time
            
            # if local_rank == 0:
            #     print("backward_time:", time.time()-backward_time)

            # Log memory usage after backward pass


            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
            
            # Optimizer step

            optim.step()
            optim.zero_grad(set_to_none=True)
            
            # if local_rank == 0:
            #     print("optimize_time:", time.time()-opt_time)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None


            if rank==0:
                tensorboard_writer.add_scalar("Loss/total_loss", loss, global_step)
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
                
                log_memory_usage(device, global_step, "after_backward")
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)  # [B,201]
                    print(
                        f"[logits check step {global_step}] "
                        f"logits min/max = ({logits.min().item():.4f}, {logits.max().item():.4f})  "
                        f"probs min/max = ({probs.min().item():.6f}, {probs.max().item():.6f})"
                        f"probs sum (first 5 samples) = {probs[:5].sum(dim=-1).detach().cpu()}"
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

                pbar.update(int(config.log_interval))
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
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} {int(config.log_interval)}step_use_time={elapsed:.1f}s one_forward_time={one_forward_time:.1f}s one_lossbackward_time={one_lossbackward_time:.1f}s"
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

            # Update progress bar
            # if pbar is not None:
            #     pbar.update(1)
            #     pbar.set_postfix(
            #         # {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
            #         {"time": time.time()-start_time}
            #     )

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
