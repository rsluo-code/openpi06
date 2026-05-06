# 2026-04-24 prompt_type 逐 episode 配置

本轮把 `025` 的 prompt 配置改成更合理的形式：

- 不再在 bash 里手填 `PROMPT=""`
- 改成由 bash 维护 `PROMPT_TYPES=(...)`
- 每个 `episode_dir` 对应一个 `prompt_type`
- 真正传给 server 的 `prompt` 文本，从 `validation_1_hand_valuenet.py` 中已有的 `prompt_map` 自动取

## 修改点

- `025validation_1_hand_valuenet_multi.sh`
  - 删除单个：
    - `PROMPT_TYPE`
    - `PROMPT`
  - 改为：

```bash
PROMPT_TYPES=(
  "sf包裹"
  ...
)
```

- `validation_1_hand_valuenet_multi.py`
  - 新增 `prompt_types: list[str]`
  - 执行时检查：
    - `len(prompt_types) == len(episode_dirs)`
  - 每轮评估前，会按当前 episode 的 `prompt_type`：
    - 校验是否存在于 `single_eval.prompt_map`
    - 自动设置：
      - `prompt_type`
      - `prompt = prompt_map[prompt_type]`

## 结果

- 现在支持“一条 episode 一个 prompt_type”
- 名字里的 `prompt_type` 和实际发给 server 的 `prompt` 会保持一致
- prompt 的英文文本只在 `validation_1_hand_valuenet.py` 的 `prompt_map` 里维护一份

## 验证

- `python3 -m py_compile validation_1_hand_valuenet_multi.py validation_1_hand_valuenet.py`
- `bash -n 025validation_1_hand_valuenet_multi.sh`

都已通过。
