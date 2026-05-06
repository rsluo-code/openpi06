# 2026-04-24 新增多 episode validation 入口

本轮没有改旧的 `024validation_1_hand_valuenet.sh` 和 `validation_1_hand_valuenet.py` 行为，而是新增了一套独立入口：

- `025validation_1_hand_valuenet_multi.sh`
- `validation_1_hand_valuenet_multi.py`

## 设计

- 旧入口继续保持原状，避免影响已有使用方式。
- 新 Python 脚本直接复用旧脚本中的核心单 episode 评估函数 `_eval_single_episode(...)`。
- 新增的只是“多 episode_dir 解析层”：
  - `episode_dirs: list[str]`
  - `episode_dirs_file: str | None`
  - 兼容旧的 `episode_glob`
  - 如果都不传，则回退到旧的单 `episode_dir`

## 新脚本支持的输入方式

- 显式多个目录：

```bash
bash 025validation_1_hand_valuenet_multi.sh \
  --episode_dirs /path/ep1 /path/ep2 /path/ep3
```

- 从文件读取多个目录：

```bash
bash 025validation_1_hand_valuenet_multi.sh \
  --episode_dirs_file /path/episode_dirs.txt
```

- 用 glob：

```bash
bash 025validation_1_hand_valuenet_multi.sh \
  --episode_glob '/path/to/episode*'
```

## 验证

- `python3 -m py_compile validation_1_hand_valuenet_multi.py`
- `bash -n 025validation_1_hand_valuenet_multi.sh`

都已通过。

补充：

- 用系统 `python3` 直接 import 时缺 `tyro`，这是因为它没进 `pi06_env_old`。
- 实际运行 `025validation_1_hand_valuenet_multi.sh` 时会先 `conda activate pi06_env_old`，这条链路是对的。
