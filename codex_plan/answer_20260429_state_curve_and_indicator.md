# 2026-04-29: validation state 曲线维度可配置 + PI06 indicator 尾部强制 true

本轮做了两件事。

## 1. validation 的 state 曲线不再写死第 8 维

之前：

- `validation_1_hand_valuenet.py` 的 `val` 图固定画 `states[:, 7]`
- 这等于把“右手第 8 维夹爪 state”写死了

现在改成：

- `validation_1_hand_valuenet.py` 新增参数：

```python
plot_state_index: int = 7
```

- `024validation_1_hand_valuenet.sh`
- `025validation_1_hand_valuenet_multi.sh`

新增 bash 变量：

```bash
PLOT_STATE_INDEX="7"
```

然后传给 Python：

```bash
--plot-state-index
```

行为：

- `PLOT_STATE_INDEX="7"` 时，单手 8 维 state 会画第 8 维
- 若当前是右手，这通常就是右手夹爪
- 若当前是左手，这通常就是左手夹爪
- 若改成别的值，就画对应维度

说明：

- 当前 `validation_1_hand_valuenet.py` 仍然只支持单手模式；`use_left == use_right` 仍会报错
- 所以这里的 state 曲线维度配置，是在“单手 8 维 state”内部选第几维

## 2. PI06 中 indicator 增加尾部 50 帧强制 true

文件：

- `src/openpi/models_pytorch/pi06_pytorch.py`

之前逻辑：

- 若 `self.value_net is not None`
- 直接：

```python
indicator = self.value_net._model.forward_cal_indicator(...)
```

现在增加：

```python
tail_window = (episode_length - step_index) <= 50
indicator = torch.logical_or(indicator.bool(), tail_window.bool()).to(dtype=indicator.dtype)
```

也就是：

- 先按原本 ValueNet 算 indicator
- 再把处于倒数 50 帧的位置强制置为 true

这样尾部 50 帧会稳定进入 advantage rewrite 的正例逻辑。

## 影响范围

### validation 改动影响：

- `validation_1_hand_valuenet.py`
- `024validation_1_hand_valuenet.sh`
- `025validation_1_hand_valuenet_multi.sh`

### PI06 indicator 改动影响：

- `src/openpi/models_pytorch/pi06_pytorch.py`

不影响：

- ValueNet 训练
- server image_keys 逻辑
- dataset 逻辑
