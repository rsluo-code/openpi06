# 2026-04-24 调整 025 使用方式

本轮没有改 Python 逻辑，只改了 `025validation_1_hand_valuenet_multi.sh` 的使用方式，让多 episode 配置写在 bash 脚本内部，而不是要求命令行传参。

## 修改点

在 `025validation_1_hand_valuenet_multi.sh` 中新增三种可配置来源：

- `EPISODE_DIRS=(...)`
- `EPISODE_DIRS_FILE=""`
- `EPISODE_GLOB=""`

脚本优先级：

1. `EPISODE_DIRS_FILE`
2. `EPISODE_GLOB`
3. `EPISODE_DIRS`

最后由 shell 组装 `PY_ARGS`，再调用：

```bash
python validation_1_hand_valuenet_multi.py "${PY_ARGS[@]}"
```

## 当前使用方式

用户现在只需要编辑 bash 脚本顶部配置，再直接执行：

```bash
bash 025validation_1_hand_valuenet_multi.sh
```

## 验证

- `bash -n 025validation_1_hand_valuenet_multi.sh`

已通过。
