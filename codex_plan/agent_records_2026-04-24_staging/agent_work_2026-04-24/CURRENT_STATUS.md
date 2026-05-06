# Current Status 2026-04-24

更新时间：2026-04-24 CST。

## 当前状态

- 工作目录：
  `/home/rsluo/codes/openpi06`
- 本轮核心目标：
  - ValueNet 训练支持从 checkpoint 正确 resume。
  - 四路图像输入新增 `episode_first_head_img`。
  - `021server_valuenet.sh` 和 `024validation_1_hand_valuenet.sh` 继续可用。
  - `002wx_cal_torch_calAt.sh` / `calVal` 也适配四路图像。
- 当前 `calAt` 使用的最新 ValueNet checkpoint：
  `/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260423/80000`

## 已完成

- 已修复 `scripts/train_pytorch_valuenet.py` 的 resume 机制：
  - 可加载最新数字 checkpoint。
  - 恢复 `optimizer.pt` 和 `metadata.pt`。
  - `global_step` 恢复后继续训练，不会重新从 `0` 计数保存。
- 已确认 `021server_valuenet.sh -> websocket server -> validation_1_hand_valuenet.py` 链路支持 `episode_first_head_img`。
- 已把服务端缺字段报错改为显式中文错误，并包含：
  - 缺少哪些字段
  - 当前 `config.image_keys`
  - 客户端实际发送字段列表
- 已确认 websocket client 会把服务端错误抛成 `RuntimeError`，validation 会及时停下。
- 已修改 `validation_1_hand_valuenet.py`：
  - `Rt0 = -(L - i) / max_len`
  - `At1` 用轨迹长度 `L`
  - `At2` 用轨迹长度 `L - N`
  - `It1/It2` 也按同样长度约定
  - 支持 `episode_glob` / `EPISODE_GLOB` 批量跑多个 `episode*`
- 已修正 `calAt` 数据链：
  - 不能用纯 ValueNet dataloader，因为要 `observation_tN`
  - 已改回 PI06 数据链
  - 同时给 PI06 数据链补上 `episode_first_head_img`
- 已统一 `calAt` / `calVal` 的 `pytorch_weight_path` 到：
  `/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260423/80000`
- 已新增 bucket 汇总脚本：
  - `z_bucket_csvs/merge_bucket_csvs.py`
  - `z_bucket_csvs/run_merge_bucket_csvs.sh`
- 已在 `z_bucket_csvs/20260204_5item_8dim/total_rank` 生成 8 卡合并结果和 top 10%-90% 阈值汇总。
- 已新增多 episode validation 入口：
  - `025validation_1_hand_valuenet_multi.sh`
  - `validation_1_hand_valuenet_multi.py`
  - 不改旧 `024` 的行为
  - `025` 现在改为在 bash 脚本内部配置 `EPISODE_DIRS / EPISODE_DIRS_FILE / EPISODE_GLOB`
  - 直接执行 `bash 025validation_1_hand_valuenet_multi.sh` 即可
  - 输出名现在由参数自动生成，不再硬依赖旧的全局 `NAME_SAVE`
  - 默认命名包含：`valuenet 时间 / step / left-right-both / dim / maxlen / prompt_type`
  - `025` 现在支持 `PROMPT_TYPES=(...)`，按 episode 一一对应
  - 真正传给 server 的 `prompt` 从 `validation_1_hand_valuenet.py` 的 `prompt_map` 自动取
  - `025` 已补充详细中文注释，说明变量含义、优先级和修改效果
  - `025` 现在支持从 bash 配置 `VIDEO_PANEL_WIDTH/VIDEO_PANEL_HEIGHT`
  - mp4 布局已改成 3 行 2 列：`head|out_png`、`left|out_png_At`、`right|out_png_It`

## 已修改文件

- `scripts/train_pytorch_valuenet.py`
- `validation_1_hand_valuenet.py`
- `024validation_1_hand_valuenet.sh`
- `src/openpi/policies/linden_valuenet_inoutput.py`
- `src/openpi/policies/linden_pi06_inoutput.py`
- `src/openpi/training/linden_rlds_pi06_dataset.py`
- `src/openpi/training/config.py`
- `z_bucket_csvs/merge_bucket_csvs.py`
- `z_bucket_csvs/run_merge_bucket_csvs.sh`
- `025validation_1_hand_valuenet_multi.sh`
- `validation_1_hand_valuenet_multi.py`
- `validation_1_hand_valuenet.py`

## 日志结论

- `debug/train_valuenet_20260423_142824.txt`
  - 可验证 `get_latest_checkpoint_step` 之前确实只拿到 `35000`，但原脚本没有真正恢复后续训练状态。
- `debug/cal_valuenet_At_20260424_112409.txt`
  - 报错：`ValueError: not enough values to unpack (expected 9, got 7)`
  - 原因：`calAt` 错走纯 ValueNet loader，缺 `observation_tN`
- `debug/cal_valuenet_At_20260424_113345.txt`
  - 报错：`torch.OutOfMemoryError`
  - 原因：四路图像下 `calAt` 显存压力上升
- `debug/cal_valuenet_At_20260424_114117.txt`
  - 已正常推进到至少 `24/3341`
  - 单步约 `24s/it`
  - 全量估算时长约 `22.3h`
- `z_bucket_csvs/20260204_5item_8dim/total_rank/top_percent_thresholds.txt`
  - 已生成合并后的 top 10%-90% 阈值表
  - 当前结果：
    - 10%: `22, 42, -80, -105`
    - 20%: `12, 21, -115, -125`
    - 30%: `2, 9, -155, -145`
    - 40%: `-3, 1, -185, -175`
    - 50%: `-13, -4, -230, -225`
    - 60%: `-18, -10, -270, -270`
    - 70%: `-28, -21, -320, -325`
    - 80%: `-43, -36, -375, -375`
    - 90%: `-118, -69, -490, -480`

## 当前风险

- `calAt` 不是“跑不通”，而是“跑得很久”。
- 只要 `config.image_keys` 要求 `episode_first_head_img`，客户端就必须发，否则会立即失败。
- 现在 `calAt` 需要 PI06 数据链；后续不要误切回纯 ValueNet 数据链。
- `PI06_pretrain` 现在已显式启用四路 `image_keys`，包含 `episode_first_head_img`。
- `pi06_pytorch.py` 内部加载的 ValueNet checkpoint 已切到 `20260423/80000`。

## 下次接手建议

- 继续处理 `calAt` 前，先确认：
  - `value_pretrain_16dim_calAt` 是否仍是 `RLDSLindenPI06DataConfig`
  - `image_keys` 是否仍含 `episode_first_head_img`
  - `pytorch_weight_path` 是否仍指向用户想要的 checkpoint
- 如果用户继续看运行时长或性能，优先从日志估算，不要直接重跑。
