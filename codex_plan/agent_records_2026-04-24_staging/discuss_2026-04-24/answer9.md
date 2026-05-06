# 2026-04-24 PI06 训练接通 ValueNet

本轮按用户要求，把 `PI06_pretrain` 这条训练链改成真正启用 ValueNet，并确保 PI06 数据链显式带上 `episode_first_head_img`。

## 实际修改

- `src/openpi/training/config.py`
  - `PI06_pretrain` 当前已是：

```python
if_use_valuenet=True
```

  - 在 `PI06_pretrain.data=RLDSLindenPI06DataConfig(...)` 中显式新增：

```python
image_keys=(
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "episode_first_head_img",
)
```

- `src/openpi/models_pytorch/pi06_pytorch.py`
  - 将内部加载的 ValueNet checkpoint 从旧路径：

```text
/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260413/80000
```

  - 改为：

```text
/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260423/80000
```

## 说明

- 其中第 1 条用户要求在当前代码里已经是完成状态，`PI06_pretrain` 本来就已经是 `if_use_valuenet=True`，所以本轮没有再重复改它。
- 本轮真正补的是：
  - PI06 训练输入里显式带第四路 `episode_first_head_img`
  - PI06 内部 ValueNet 路径切到最新 checkpoint

## 验证

- `python3 -m py_compile src/openpi/training/config.py src/openpi/models_pytorch/pi06_pytorch.py`

已通过。
