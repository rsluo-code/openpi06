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
import csv
import gc
import logging
import os
import platform
import pathlib
import shutil
import sys
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
# 基础库
import torch
import os
import datetime  # 用于生成唯一的日志目录名
# TensorBoard核心库
from torch.utils.tensorboard import SummaryWriter
try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

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


@dataclasses.dataclass
class MetricsArtifacts:
    csv_path: pathlib.Path
    png_path: pathlib.Path
    rows: list[dict[str, float | int | str]]
    fieldnames: list[str]


def init_metrics_artifacts(config: _config.TrainConfig) -> MetricsArtifacts:
    metrics_dir = config.checkpoint_dir / "monitoring"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = metrics_dir / f"training_metrics_{run_tag}.csv"
    png_path = metrics_dir / f"training_metrics_{run_tag}.png"
    fieldnames = [
        "timestamp",
        "global_step",
        "interval_steps",
        "elapsed_sec",
        "time_per_step_sec",
        "loss",
        "learning_rate",
        "grad_norm",
        "gpu_mem_allocated_gb",
        "gpu_mem_reserved_gb",
        "gpu_mem_peak_allocated_gb",
        "gpu_mem_peak_reserved_gb",
        "cpu_percent",
        "cpu_mem_used_gb",
        "cpu_mem_total_gb",
        "cpu_mem_percent",
        "process_rss_gb",
    ]
    rows: list[dict[str, float | int | str]] = []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    logging.info(f"Metrics CSV: {csv_path}")
    logging.info(f"Metrics PNG: {png_path}")
    print(f"[metrics] csv_path={csv_path}")
    print(f"[metrics] png_path={png_path}")
    return MetricsArtifacts(csv_path=csv_path, png_path=png_path, rows=rows, fieldnames=fieldnames)


def append_metrics_row(artifacts: MetricsArtifacts, row: dict[str, float | int | str]) -> None:
    artifacts.rows.append(row)
    with artifacts.csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=artifacts.fieldnames)
        writer.writerow(row)


def render_metrics_plot(artifacts: MetricsArtifacts) -> None:
    if plt is None or not artifacts.rows:
        return

    steps = [int(row["global_step"]) for row in artifacts.rows]
    metric_specs = [
        ("loss", "Loss", [("loss", "loss")]),
        ("learning_rate", "Learning Rate", [("learning_rate", "lr")]),
        ("grad_norm", "Grad Norm", [("grad_norm", "grad_norm")]),
        (
            "gpu_mem_allocated_gb",
            "GPU Memory (GB)",
            [
                ("gpu_mem_allocated_gb", "allocated"),
                ("gpu_mem_reserved_gb", "reserved"),
                ("gpu_mem_peak_allocated_gb", "peak_alloc"),
                ("gpu_mem_peak_reserved_gb", "peak_reserved"),
            ],
        ),
        ("cpu_percent", "CPU Percent", [("cpu_percent", "cpu%")]),
        ("cpu_mem_percent", "CPU Memory", [("cpu_mem_percent", "mem%"), ("cpu_mem_used_gb", "used_gb")]),
        ("process_rss_gb", "Process RSS (GB)", [("process_rss_gb", "rss_gb")]),
    ]

    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(16, 24), sharex=True)
    fig.suptitle("PI06 Training Metrics", fontsize=16)

    for ax, (_, title, series_specs) in zip(axes, metric_specs, strict=True):
        has_any_series = False
        for field, label in series_specs:
            values = []
            valid = False
            for row in artifacts.rows:
                value = row.get(field, "")
                if value == "":
                    values.append(float("nan"))
                else:
                    values.append(float(value))
                    valid = True
            if valid:
                ax.plot(steps, values, label=label, linewidth=1.5)
                has_any_series = True
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if has_any_series and len(series_specs) > 1:
            ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Global Step")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(artifacts.png_path, dpi=150)
    plt.close(fig)

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

        logging.info(f"Loading {safetensors_path}")
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


def get_system_metrics(device):
    metrics = {}
    if torch.cuda.is_available():
        metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        metrics["gpu_mem_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        metrics["gpu_mem_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
    if psutil is not None:
        vm = psutil.virtual_memory()
        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        metrics["cpu_mem_used_gb"] = (vm.total - vm.available) / 1e9
        metrics["cpu_mem_total_gb"] = vm.total / 1e9
        metrics["cpu_mem_percent"] = vm.percent
        proc = psutil.Process(os.getpid())
        metrics["process_rss_gb"] = proc.memory_info().rss / 1e9
    return metrics


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # 1. 初始化分布式环境（单机多卡/多机多卡通用）
    rank = local_rank


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
        metrics_artifacts = init_metrics_artifacts(config)
        if plt is None:
            logging.warning("matplotlib is not available; metrics PNG rendering is disabled")
    else:
        metrics_artifacts = None
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

    from openpi.models_pytorch.pi06_pytorch import PI06Pytorch
    model = PI06Pytorch(model_cfg, image_keys=data_config.image_keys).to(device)

    # if hasattr(model, "gradient_checkpointing_enable"):
    #     enable_gradient_checkpointing = True
    #     model.gradient_checkpointing_enable()
    #     logging.info("Enabled gradient checkpointing for memory optimization")
    # else:
    #     enable_gradient_checkpointing = False
    #     logging.info("Gradient checkpointing is not supported for this model")

    # model.paligemma_with_expert.gemma_expert.eval()
    # for params in model.paligemma_with_expert.gemma_expert.parameters():
    #     params.requires_grad = False
    # print("*" * 50)
    # print("*" * 50)
    # print("Warnning! gemma_expert is requires_grad = False")
    # print("*" * 50)
    # print("*" * 50)

    enable_gradient_checkpointing = False
    model.gradient_checkpointing_disable()

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
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        # safetensors.torch.load_model(
        #     (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        # )
        from safetensors.torch import load_file
        state_dict = load_file(model_path, device=str(device))
        missing_keys, unexpected_keys = (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).load_state_dict(
            state_dict,
            strict=False
        )
        if missing_keys:
            print("[load_state_dict] missing keys:")
            for k in missing_keys:
                print("  ", k)
        if unexpected_keys:
            print("[load_state_dict] unexpected keys:")
            for k in unexpected_keys:
                print("  ", k)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # with torch.no_grad():
    #     (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).success_token_linear.weight.zero_()
    #     if (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).success_token_linear.bias is not None:
    #         (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).success_token_linear.bias.zero_()

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
        logging.info(f"Resumed training from step {global_step}")

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
    show_pbar = is_main and sys.stderr.isatty()
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not show_pbar)
        if show_pbar
        else None
    )
    if is_main and not show_pbar:
        logging.info("Progress bar disabled because stderr is not a TTY; using log_interval summaries only")

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        # for observation, actions,_ ,_ in loader:
        for observation,obs_N, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure,language_instruction_at_30precent in loader:
        # for observation, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure in loader:

            # start_time = time.time()
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            language_instruction_at_30precent = language_instruction_at_30precent.to(device)
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            obs_N = jax.tree.map(lambda x: x.to(device), obs_N)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901
            # if local_rank == 0 or local_rank == 1:
            #     print("rank:", local_rank, "actions:", actions)
            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            # torch.cuda.synchronize()
            # forward_time = time.time()
            losses = model(observation,obs_N, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure,language_instruction_at_30precent )
            # losses = model(observation, actions)
            # if local_rank == 0:
            #     print("forward_time:", time.time()-forward_time)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
            # Backward pass
            # torch.cuda.synchronize()
            # backward_time = time.time()
            loss.backward()
            # if local_rank == 0:
            #     print("backward_time:", time.time()-backward_time)

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

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

            # 写入TensorBoard指标（核心步骤）
            # 写入所有train_info里的指标
            if rank==0:
                tensorboard_writer.add_scalar("Loss/total_loss", float(loss), global_step)
                tensorboard_writer.add_scalar("Train/learning_rate", optim.param_groups[0]["lr"], global_step)
                tensorboard_writer.add_scalar(
                    "Train/grad_norm",
                    float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    global_step,
                )
            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            global_step += 1

            if pbar is not None:
                pbar.update(1)

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                interval_steps = len(infos)

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

                system_metrics = get_system_metrics(device)
                if is_main and system_metrics:
                    metrics_msg = ", ".join(
                        [
                            f"gpu_alloc={system_metrics['gpu_mem_allocated_gb']:.2f}GB",
                            f"gpu_reserved={system_metrics['gpu_mem_reserved_gb']:.2f}GB",
                        ]
                        + (
                            [
                                f"cpu={system_metrics['cpu_percent']:.1f}%",
                                f"cpu_mem={system_metrics['cpu_mem_percent']:.1f}%",
                                f"rss={system_metrics['process_rss_gb']:.2f}GB",
                            ]
                            if "cpu_percent" in system_metrics
                            else []
                        )
                    )
                    logging.info(f"step={global_step} system {metrics_msg}")
                    if rank == 0:
                        for key, value in system_metrics.items():
                            tensorboard_writer.add_scalar(f"System/{key}", value, global_step)

                if metrics_artifacts is not None:
                    metrics_row = {
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "global_step": global_step,
                        "interval_steps": interval_steps,
                        "elapsed_sec": elapsed,
                        "time_per_step_sec": elapsed / max(1, interval_steps),
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm if avg_grad_norm is not None else "",
                        "gpu_mem_allocated_gb": system_metrics.get("gpu_mem_allocated_gb", ""),
                        "gpu_mem_reserved_gb": system_metrics.get("gpu_mem_reserved_gb", ""),
                        "gpu_mem_peak_allocated_gb": system_metrics.get("gpu_mem_peak_allocated_gb", ""),
                        "gpu_mem_peak_reserved_gb": system_metrics.get("gpu_mem_peak_reserved_gb", ""),
                        "cpu_percent": system_metrics.get("cpu_percent", ""),
                        "cpu_mem_used_gb": system_metrics.get("cpu_mem_used_gb", ""),
                        "cpu_mem_total_gb": system_metrics.get("cpu_mem_total_gb", ""),
                        "cpu_mem_percent": system_metrics.get("cpu_mem_percent", ""),
                        "process_rss_gb": system_metrics.get("process_rss_gb", ""),
                    }
                    append_metrics_row(metrics_artifacts, metrics_row)
                    render_metrics_plot(metrics_artifacts)

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / max(1, interval_steps),
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    log_payload.update(system_metrics)
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection
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
