import os
import numpy as np
import matplotlib.pyplot as plt


def lr_sched_v1(step: int, warmup: int, decay_steps: int, peak_lr: float, end_lr: float) -> float:
    if step < warmup:
        init_lr = peak_lr / (warmup + 1)
        return init_lr + (peak_lr - init_lr) * (step / warmup)
    prog = min(1.0, (step - warmup) / (decay_steps - warmup))
    return end_lr + (peak_lr - end_lr) * 0.5 * (1.0 + np.cos(np.pi * prog))


def lr_sched_v2(step: int, warmup_steps: int, decay_steps: int, peak_lr: float, end_lr: float) -> float:
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cos


def plot_lr_schedules_to_png(
    save_path: str,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    peak_lr: float,
    end_lr: float,
    num_plot_points: int = 4000,
):
    """
    在服务器环境安全运行：
    - 不 show
    - 直接保存 png

    save_path: e.g. "./lr_schedule_compare.png"
    """

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ========= 下采样 =========
    stride = max(1, total_steps // num_plot_points)
    steps = np.arange(0, total_steps + 1, stride, dtype=np.int64)

    lr1 = np.array([lr_sched_v1(int(s), warmup_steps, decay_steps, peak_lr, end_lr) for s in steps])
    lr2 = np.array([lr_sched_v2(int(s), warmup_steps, decay_steps, peak_lr, end_lr) for s in steps])

    # ========= 打印关键信息 =========
    print("=== LR Schedule Plot ===")
    print(f"save_path        = {save_path}")
    print(f"total_steps      = {total_steps}")
    print(f"warmup_steps     = {warmup_steps}")
    print(f"decay_steps      = {decay_steps}")
    print(f"num_plot_points  ≈ {len(steps)}")
    print(f"stride           = {stride}")
    print()
    print(f"v1 lr(step=0)        = {lr_sched_v1(0, warmup_steps, decay_steps, peak_lr, end_lr):.8e}")
    print(f"v2 lr(step=0)        = {lr_sched_v2(0, warmup_steps, decay_steps, peak_lr, end_lr):.8e}")
    print(f"v1 lr(step=warmup)   = {lr_sched_v1(warmup_steps, warmup_steps, decay_steps, peak_lr, end_lr):.8e}")
    print(f"v2 lr(step=warmup)   = {lr_sched_v2(warmup_steps, warmup_steps, decay_steps, peak_lr, end_lr):.8e}")

    # ========= 绘图 =========
    plt.figure(figsize=(8, 5))
    plt.plot(steps, lr1, label="lr_sched_v1")
    plt.plot(steps, lr2, label="lr_sched_v2")
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.title("Learning Rate Schedule Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()   # ⭐ 非常重要：服务器环境必须 close

    print(f"PNG saved to: {save_path}")


if __name__ == "__main__":
    plot_lr_schedules_to_png(
        save_path="./lr_schedule_compare.png",
        total_steps=400_001,
        warmup_steps=1_000,
        decay_steps=1_000_000,
        peak_lr=5e-5,
        end_lr=5e-5,
        num_plot_points=4000,   # 绘制点数
    )




