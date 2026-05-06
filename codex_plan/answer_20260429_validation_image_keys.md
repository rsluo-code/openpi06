# 2026-04-29: 024/025 validation 增加 image_keys 配置

本轮目标：

- 给 `024validation_1_hand_valuenet.sh`
- 给 `025validation_1_hand_valuenet_multi.sh`

增加 bash 侧可配置的 `IMAGE_KEYS`，避免 validation 客户端总是把不需要的图像都发给 server，增加 websocket 传输压力。

## 结论

现在 `024/025 -> validation_1_hand_valuenet.py -> websocket server` 这条链路，客户端实际发送哪些图像，由 bash 中的 `IMAGE_KEYS` 决定。

默认配置为三路：

```bash
IMAGE_KEYS=(
  "base_0_rgb"
  "left_wrist_0_rgb"
  "right_wrist_0_rgb"
  # "episode_first_head_img"
)
```

如果 server 端当前 `config.image_keys` 是四路，就把：

```bash
# "episode_first_head_img"
```

打开。

## 这次改了什么

### 1. 024/025 bash 增加 IMAGE_KEYS

文件：

- `024validation_1_hand_valuenet.sh`
- `025validation_1_hand_valuenet_multi.sh`

增加了 `IMAGE_KEYS=(...)`，并通过命令行参数传给：

```bash
--image-keys ...
```

### 2. validation_1_hand_valuenet.py 只发送被配置的图像字段

文件：

- `validation_1_hand_valuenet.py`

新增：

- `Args.image_keys`
- `_build_request_element(...)`

行为变为：

- 只在 `image_keys` 包含 `base_0_rgb` 时发送 `observation/image`
- 只在 `image_keys` 包含 `left_wrist_0_rgb` 时发送 `observation/wrist_image_left`
- 只在 `image_keys` 包含 `right_wrist_0_rgb` 时发送 `observation/wrist_image_right`
- 只在 `image_keys` 包含 `episode_first_head_img` 时发送 `observation/episode_first_head_img`

因此，当 bash 中把 `episode_first_head_img` 注释掉后，这一路不会再通过 websocket 发送。

### 3. server 端 ValueNet 输入解析也改成按 image_keys 动态要求字段

文件：

- `src/openpi/policies/linden_valuenet_inoutput.py`

之前的问题：

- server 端虽然对 `episode_first_head_img` 已经是条件要求
- 但对 `observation/image`
- `observation/wrist_image_left`
- `observation/wrist_image_right`

还是写死要求

现在改成：

- 只对 `config.image_keys` 中出现的 key 要求对应字段
- 并且只解析这些图像

这样 client/server 两边就一致了。

## 使用说明

如果 server 当前配置是三路：

```bash
IMAGE_KEYS=(
  "base_0_rgb"
  "left_wrist_0_rgb"
  "right_wrist_0_rgb"
)
```

如果 server 当前配置是四路：

```bash
IMAGE_KEYS=(
  "base_0_rgb"
  "left_wrist_0_rgb"
  "right_wrist_0_rgb"
  "episode_first_head_img"
)
```

## 风险提示

这套配置必须和 server 端当前加载的 `TrainConfig.data.image_keys` 完全一致。

否则 server 会在 `LindenInputs` 阶段直接报缺字段错误。
