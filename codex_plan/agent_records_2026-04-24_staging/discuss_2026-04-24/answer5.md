# 2026-04-24 validation 命名逻辑解耦

本轮把 `validation_1_hand_valuenet.py` 里输出命名对全局变量的依赖解开了，避免 `025` 继续被旧的 `NAME_SAVE / LEIBIE / USE_LEFT / USE_RIGHT` 这些模块级默认值绑死。

## 修改点

- `validation_1_hand_valuenet.py`
  - `Args` 新增：
    - `output_base_dir`
    - `name_save`
    - `model_time`
    - `model_step`
    - `model_dim`
    - `prompt_type`
  - 新增 `_build_name_save(args, max_len)`，默认会生成：

```text
valuenet_{model_time}_step{model_step}_{arm_mode}_{model_dim}_maxlen{max_len}_{prompt_type}
```

  - 其中 `arm_mode` 自动根据：
    - `use_left=True,use_right=False -> left`
    - `use_left=False,use_right=True -> right`
    - `use_left=True,use_right=True -> both`
  - `prompt_type` 优先用 `args.prompt_type`，否则从 `prompt_map` 反查，最后兜底用 `args.prompt`。
  - `_eval_single_episode(...)` 里的左右手视频选择逻辑，也改成真正读取 `args.use_left/use_right`，不再误用模块级 `USE_LEFT/USE_RIGHT`。

- `025validation_1_hand_valuenet_multi.sh`
  - 新增 bash 内部配置项：
    - `OUTPUT_BASE_DIR`
    - `NAME_SAVE`
    - `MODEL_TIME`
    - `MODEL_STEP`
    - `MODEL_DIM`
    - `PROMPT_TYPE`
    - `PROMPT`
    - `USE_LEFT`
    - `USE_RIGHT`
  - shell 会把这些参数传给 `validation_1_hand_valuenet_multi.py`

## 现在的默认命名

按当前 `025` 配置，输出名前缀会接近：

```text
valuenet_20260423_step80000_right_8dim_maxlen700_sf包裹
```

再拼上具体 episode tag，形成每个 episode 的独立输出文件名。

## 验证

- `python3 -m py_compile validation_1_hand_valuenet.py validation_1_hand_valuenet_multi.py`
- `bash -n 025validation_1_hand_valuenet_multi.sh`

都已通过。
