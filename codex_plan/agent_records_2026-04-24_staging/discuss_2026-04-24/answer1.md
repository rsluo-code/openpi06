# 2026-04-24 记录更新

本轮按用户要求，把今天在 `openpi06` 上围绕 ValueNet、`episode_first_head_img`、`calAt/calVal`、validation/server 链路、以及最新日志结论的工作整理到 `agent_records`。

## 今日已完成工作摘要

- 修复 `scripts/train_pytorch_valuenet.py` 的 resume 逻辑：
  - 新增 checkpoint 加载逻辑。
  - `config.resume=True` 时会恢复 `global_step`、optimizer、metadata。
  - 修正保存步数顺序，避免 resume 后又从 `0` 保存。
- 检查并打通 `021server_valuenet.sh -> websocket server -> validation_1_hand_valuenet.py` 对 `episode_first_head_img` 的支持：
  - validation 客户端会发送 `observation/episode_first_head_img`。
  - server 端 `LindenInputs(image_keys=...)` 按 `config.py` 中的 `image_keys` 决定是否要求该字段。
- 增强 server 端字段缺失报错：
  - 当 `config.image_keys` 中要求 `episode_first_head_img`，但客户端没发时，服务端会抛出明确中文错误。
  - websocket client 收到错误字符串后会 `raise RuntimeError(...)`，validation 会立刻停下。
- 修改 `validation_1_hand_valuenet.py`：
  - `Rt0` 改为“当前剩余长度 / 最长长度”的定义，即 `-(L - i) / max_len`。
  - `At1` 长度为完整轨迹 `L`。
  - `At2` 长度为 `L - N`。
  - 支持 `EPISODE_GLOB` 批量匹配多个 `episode*` 目录并分别生成可视化。
- 修正 `002wx_cal_torch_calAt.sh` 对应链路：
  - `calAt` 仍走 PI06 数据链，保留 `observation_tN`。
  - PI06 数据链已补上 `episode_first_head_img` 和 `episode_first_head_img_N`。
  - `calVal` 走 ValueNet 数据链，也已支持 `episode_first_head_img`。
- 统一 `calAt` / `calVal` 的 `pytorch_weight_path` 到最新 ValueNet checkpoint：
  - `/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260423/80000`

## 今日关键代码文件

- `scripts/train_pytorch_valuenet.py`
- `validation_1_hand_valuenet.py`
- `024validation_1_hand_valuenet.sh`
- `src/openpi/policies/linden_valuenet_inoutput.py`
- `src/openpi/policies/linden_pi06_inoutput.py`
- `src/openpi/training/linden_rlds_pi06_dataset.py`
- `src/openpi/training/config.py`

## 今日日志结论

- `debug/cal_valuenet_At_20260424_112409.txt`
  - 报错不是权重问题，而是 dataloader 返回项数量不匹配：`expected 9, got 7`。
  - 根因是 `calAt` 错走了纯 ValueNet 数据链，缺少 `observation_tN`。
  - 后续已修回 PI06 数据链，同时保留 `episode_first_head_img`。
- `debug/cal_valuenet_At_20260424_113345.txt`
  - 报错为 CUDA OOM。
  - 说明四路图像接通后，`calAt` 显存压力明显变大。
- `debug/cal_valuenet_At_20260424_114117.txt`
  - 运行已推进到至少 `24/3341`。
  - 稳定单步耗时约 `23.9s - 24.1s/it`。
  - 按该速度估算，全量跑完约 `22.3` 小时。

## 当前残留风险

- `calAt` 当前可跑，但整轮耗时很长，接近一天。
- 四路图像比旧三路更吃显存；后续如果 batch 再升大，仍可能重新触发 OOM。
- server 端现在按 `config.image_keys` 严格校验字段；客户端和配置一旦不一致会立即失败，这是有意行为。

## 接手建议

- 后续继续看 `calAt` 时，先确认：
  - 当前 run 实际使用的 `config name`
  - `pytorch_weight_path`
  - `image_keys`
  - 是否仍沿用 PI06 数据链
- 再看日志时，优先区分三类问题：
  - 字段缺失 / 配置不一致
  - dataloader 返回结构不匹配
  - 显存不足或纯性能瓶颈
