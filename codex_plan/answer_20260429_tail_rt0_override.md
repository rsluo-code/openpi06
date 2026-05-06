# 2026-04-29: validation 增加尾部 30 帧 Rt0 直接覆盖模式

本轮目标：

- 给 `024validation_1_hand_valuenet.sh`
- 给 `025validation_1_hand_valuenet_multi.sh`
- 给 `validation_1_hand_valuenet.py`

增加一个定制化开关。

开启后：

- 轨迹最后 `N` 帧不再发给 server 做 infer
- 直接令：

```python
Rt1 = Rt0
Rt2 = Rt0
```

默认关闭。

## 这次改了什么

### 1. validation_1_hand_valuenet.py 增加两个参数

新增：

```python
custom_tail_direct_rt0: bool = False
custom_tail_length: int = 30
```

### 2. Rt 计算顺序调整

现在每个 timestep 的顺序是：

1. 先计算 `Rt0`
2. 判断当前是否命中最后 `custom_tail_length` 帧
3. 如果：
   - `custom_tail_direct_rt0 == True`
   - 并且当前步在最后 `custom_tail_length` 帧

则：

```python
Rt1 = Rt0
Rt2 = Rt0
```

并且：

- 不构造 request element
- 不调用 `client.infer(...)`

4. 否则仍然正常走 infer，按 logits 算 `Rt1/Rt2`

### 3. 024 / 025 bash 增加开关

新增变量：

```bash
CUSTOM_TAIL_DIRECT_RT0="false"
CUSTOM_TAIL_LENGTH="30"
```

并传给 Python：

```bash
--custom-tail-direct-rt0 / --no-custom-tail-direct-rt0
--custom-tail-length
```

## 默认行为

默认：

```bash
CUSTOM_TAIL_DIRECT_RT0="false"
CUSTOM_TAIL_LENGTH="30"
```

所以和旧逻辑完全一致，不会改变结果。

## 开启方式

若想启用最后 30 帧直接覆盖：

```bash
CUSTOM_TAIL_DIRECT_RT0="true"
CUSTOM_TAIL_LENGTH="30"
```

## 影响范围

只影响：

- `024validation_1_hand_valuenet.sh`
- `025validation_1_hand_valuenet_multi.sh`
- `validation_1_hand_valuenet.py`

不影响：

- server
- 训练
- dataset
- config

## 说明

这项定制化会降低尾部 30 帧的 websocket 推理请求数。

但也意味着：

- 最后 30 帧的 `Rt1/Rt2`
- 不再来自模型
- 而是人为设为 `Rt0`

所以这是一种有意的估计规则，不是模型真实性能评估。 
