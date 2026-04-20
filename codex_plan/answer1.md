# 修改方案：ValueNet 训练增加 `episode_first_head_img`

目标：执行 `001train_torch_valuenet.sh`，走 `value_pretrain_16dim` 配置训练 ValueNet 时，输入图像从当前 3 路增加为可配置的多路图像，例如加入 `"episode_first_head_img"`。

重要结论：

1. 只改 `src/openpi/models/model.py` 的 `IMAGE_KEYS` 不够。
2. `scripts/train_pytorch_valuenet.py` 使用的是 PyTorch ValueNet，实际预处理入口是 `src/openpi/models_pytorch/valuenet_pytorch.py`，里面调用 `src/openpi/models_pytorch/preprocessing_pytorch.py::preprocess_observation_pytorch()`。
3. 所以 PyTorch 训练真正会检查图像 key 的地方是 `src/openpi/models_pytorch/preprocessing_pytorch.py` 里的 `IMAGE_KEYS`，不是只看 `src/openpi/models/model.py`。
4. `src/openpi/policies/linden_valuenet_inoutput.py::LindenInputs` 当前把图像 names 写死为：
   - `"base_0_rgb"`
   - `"left_wrist_0_rgb"`
   - `"right_wrist_0_rgb"`
   这里也必须改，否则 `observation.images` 里根本不会有 `"episode_first_head_img"`。
5. RLDS dataset 当前 `src/openpi/training/linden_rlds_valunet_dataset.py` 只输出当前帧的 head / left wrist / right wrist，也要额外输出 episode 第一帧 head 图像。

推荐做法：不要把新图像 key 写死在多个文件里，而是把 image keys 做成 `TrainConfig -> DataConfig -> policy transform -> model preprocessing` 的配置项。

## 需要修改的文件

### 1. `src/openpi/training/config.py`

在 `DataConfig` 增加字段：

```python
image_keys: Sequence[str] = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
```

在 `RLDSLindenValueNetDataConfig` 增加同名字段：

```python
image_keys: Sequence[str] = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
```

在 `RLDSLindenValueNetDataConfig.create()` 里，把 `image_keys` 传给 `LindenInputs`：

```python
inputs=[
    linden_valuenet_inoutput.LindenInputs(
        model_type=model_config.model_type,
        use_left=self.use_left,
        use_right=self.use_right,
        train_or_infer=self.train_or_infer,
        image_keys=self.image_keys,
    )
]
```

并且在返回的 `DataConfig` 里保存：

```python
image_keys=self.image_keys,
```

最后在 `value_pretrain_16dim` 的 `data=RLDSLindenValueNetDataConfig(...)` 里配置：

```python
image_keys=(
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "episode_first_head_img",
),
```

这样以后继续加图像 key，只需要改这个 config。

### 2. `src/openpi/policies/linden_valuenet_inoutput.py`

在 `LindenInputs` dataclass 增加字段：

```python
image_keys: tuple[str, ...] = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
```

在 `__call__()` 里多读取一路输入：

```python
episode_first_head_img = _parse_image(data["observation/episode_first_head_img"])
```

把当前写死的：

```python
names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
images = (base_image, wrist_image_left, wrist_image_right)
```

改成先构造映射，再按 `self.image_keys` 取：

```python
image_by_key = {
    "base_0_rgb": base_image,
    "left_wrist_0_rgb": wrist_image_left,
    "right_wrist_0_rgb": wrist_image_right,
    "episode_first_head_img": episode_first_head_img,
}

mask_by_key = {
    "base_0_rgb": np.True_,
    "left_wrist_0_rgb": image_masks[1],
    "right_wrist_0_rgb": image_masks[2],
    "episode_first_head_img": np.True_,
}

inputs = {
    "state": state,
    "image": {key: image_by_key[key] for key in self.image_keys},
    "image_mask": {key: mask_by_key[key] for key in self.image_keys},
    ...
}
```

注意：`use_left=False/use_right=True` 时，当前代码会把 left wrist mask 设为 False。上面保留这个逻辑即可；`episode_first_head_img` 建议 mask 始终 True。

如果担心配置写错，可以加显式检查：

```python
missing = set(self.image_keys) - set(image_by_key)
if missing:
    raise ValueError(f"Unsupported image_keys: {sorted(missing)}")
```

### 3. `src/openpi/training/config.py` 的 repack transform

在 `RLDSLindenValueNetDataConfig.create()` 的 `RepackTransform` 映射里增加：

```python
"observation/episode_first_head_img": "observation/episode_first_head_img",
```

否则 `LindenInputs` 读取不到 `data["observation/episode_first_head_img"]`。

### 4. `src/openpi/training/linden_rlds_valunet_dataset.py`

在 `restructure(traj)` 里构造 episode 第一帧 head 图：

```python
episode_first_head_img = tf.repeat(
    traj["observation"]["image_head"][0:1],
    tf.shape(traj["observation"]["image_head"])[0],
    axis=0,
)
```

然后在返回的 `"observation"` 里增加：

```python
"episode_first_head_img": episode_first_head_img,
```

在 `chunk_actions(traj)` 里同步裁剪：

```python
traj["observation"]["episode_first_head_img"] = traj["observation"]["episode_first_head_img"][:num_chunks]
```

在 `decode_images(traj)` 里同步 decode：

```python
traj["observation"]["episode_first_head_img"] = tf.io.decode_image(
    traj["observation"]["episode_first_head_img"],
    expand_animations=False,
    dtype=tf.uint8,
)
```

说明：这里假设 RLDS 的 `traj["observation"]["image_head"]` 是 encoded image string 序列，和当前 `"image"` / wrist image 的处理方式一致。

### 5. `src/openpi/models_pytorch/preprocessing_pytorch.py`

ValueNet PyTorch 训练实际会走这里。

有两个选择：

推荐选择 A：让 ValueNet 把配置传进来，不依赖全局默认 `IMAGE_KEYS`。

把 `ValueNetPytorch` 改成持有 `image_keys`，并调用：

```python
obs = _preprocessing.preprocess_observation_pytorch(
    observation,
    train=train,
    image_keys=self.image_keys,
)
```

这需要配合 `scripts/train_pytorch_valuenet.py` 构造模型时传入 `data_config.image_keys`。

备选选择 B：直接把 `preprocessing_pytorch.py` 的 `IMAGE_KEYS` 也加上：

```python
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "episode_first_head_img",
)
```

但这个做法不满足“以后在 TrainConfig 中配置更多图像 key”的目标，而且会影响所有使用这个默认值的 PyTorch 模型。

### 6. `src/openpi/models_pytorch/valuenet_pytorch.py`

推荐把 `ValueNetPytorch.__init__()` 增加参数：

```python
def __init__(self, config, num_bins=201, image_keys=None):
    ...
    self.image_keys = tuple(image_keys or _preprocessing.IMAGE_KEYS)
```

然后 `_preprocess_observation()` 改为：

```python
obs = _preprocessing.preprocess_observation_pytorch(
    observation,
    train=train,
    image_keys=self.image_keys,
)
```

这样 `list(obs.images.values())` 的顺序就由 `self.image_keys` 控制。

### 7. `scripts/train_pytorch_valuenet.py`

现在代码先 `build_datasets(config)` 得到 `data_config`，再创建模型：

```python
loader, data_config = build_datasets(config)
model = ValueNetPytorch(model_cfg, num_bins=201).to(device)
```

改成：

```python
model = ValueNetPytorch(
    model_cfg,
    num_bins=201,
    image_keys=data_config.image_keys,
).to(device)
```

这样 `value_pretrain_16dim` 的 `TrainConfig.data.image_keys` 就能真正控制 ValueNet 使用哪些图像。

### 8. `src/openpi/models/model.py`

如果 JAX 路径、通用文档或其他策略也需要知道默认图像 key，可以同步把这里的 `IMAGE_KEYS` 加上 `"episode_first_head_img"`。

但对 `001train_torch_valuenet.sh -> scripts/train_pytorch_valuenet.py -> ValueNetPytorch` 这条 PyTorch 训练链路来说，它不是唯一关键点，甚至不是实际生效的预处理默认值。

建议：

```python
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "episode_first_head_img",
)
```

同时最好也同步 `src/openpi/models_pytorch/preprocessing_pytorch.py`，或者按上面的推荐方案由配置显式传入。

## 最小可行修改顺序

1. `linden_rlds_valunet_dataset.py`：从 RLDS 轨迹里产出 `observation/episode_first_head_img`，并在 chunk/decode 阶段保留。
2. `config.py`：`DataConfig` 和 `RLDSLindenValueNetDataConfig` 增加 `image_keys`，`value_pretrain_16dim` 配置加入 `"episode_first_head_img"`。
3. `config.py`：`RepackTransform` 增加 `"observation/episode_first_head_img"` 映射，并把 `image_keys` 传给 `LindenInputs` 和最终 `DataConfig`。
4. `linden_valuenet_inoutput.py`：`LindenInputs` 按 `image_keys` 动态生成 `inputs["image"]` / `inputs["image_mask"]`。
5. `valuenet_pytorch.py` + `train_pytorch_valuenet.py`：把 `data_config.image_keys` 传给模型预处理。
6. 视需要同步 `model.py` 和 `preprocessing_pytorch.py` 的默认 `IMAGE_KEYS`。

## 验证建议

改完后先不要直接长训，建议加临时打印或断点确认一个 batch：

```python
loader, data_config = build_datasets(config)
batch = next(iter(loader))
print(data_config.image_keys)
print(batch["image"].keys())
print({k: v.shape for k, v in batch["image"].items()})
print(batch["image_mask"].keys())
```

期望看到：

```text
("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "episode_first_head_img")
dict_keys(["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "episode_first_head_img"])
```

然后再确认模型侧没有报：

```text
ValueError: images dict missing keys
```

## 额外注意

每多一路图像，PaliGemma prefix 的 image tokens 会增加一整路，显存和计算都会上涨。`batch_size=80` 可能需要下调，尤其是 `paligemma_variant="gemma_2b"` 时。
