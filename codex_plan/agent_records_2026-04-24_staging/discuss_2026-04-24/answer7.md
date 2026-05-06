# 2026-04-24 025 bash 注释补充

本轮没有改执行逻辑，只给 `025validation_1_hand_valuenet_multi.sh` 增加了就地注释，明确说明每个变量的含义、修改后的效果和优先级。

## 已补充说明的变量

- `EPISODE_DIRS`
- `EPISODE_DIRS_FILE`
- `EPISODE_GLOB`
- `OUTPUT_BASE_DIR`
- `NAME_SAVE`
- `MODEL_TIME`
- `MODEL_STEP`
- `MODEL_DIM`
- `PROMPT_TYPES`
- `USE_LEFT`
- `USE_RIGHT`

## 注释内容覆盖

- episode 来源三种方式的优先级：
  - `EPISODE_DIRS_FILE > EPISODE_GLOB > EPISODE_DIRS`
- 每种 episode 配置改动后会产生什么效果
- 输出目录和输出名的作用
- 自动命名依赖哪些元信息
- `PROMPT_TYPES` 必须和 `EPISODE_DIRS` 一一对应
- `PROMPT_TYPES` 如何从 `prompt_map` 映射到真实 prompt
- 左手 / 右手配置对评估数据选择和输出命名的影响

## 验证

- `bash -n 025validation_1_hand_valuenet_multi.sh`

已通过。
