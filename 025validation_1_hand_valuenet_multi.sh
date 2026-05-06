eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old
chmod +x /home/rsluo/miniconda3/envs/pi06_env_old/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas
echo "hello"

# episode 来源，下面支持三种方式。
# 优先级是：
#   1) EPISODE_DIRS_FILE
#   2) EPISODE_GLOB
#   3) EPISODE_DIRS
#
# 修改后的效果：
# - EPISODE_DIRS：手动指定要测试的 episode 目录。
# - EPISODE_DIRS_FILE：从 txt 文件中逐行读取 episode 目录。
# - EPISODE_GLOB：用 glob 规则批量匹配 episode 目录。
#
# 一般只维护其中一种，其他保持为空即可。
EPISODE_DIRS=(
  # Old paths:
  # "/data0/origin_datas_cut/sf_packages/data_11_04/1104_12_快递_LD_165/bg_tablecloth1arm_rightobject_1/episode_2025-11-04_110709_991_part_2"
  # "/data0/origin_datas_cut/sf_packages/data_11_07/1107_04_抓放包裹_LD_199/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_110951_569_part_1"
  # "/data0/origin_datas_cut/sf_packages/data_11_07/1107_15_抓放包裹_LD_196/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_100116_104_part_1"
  # "/data0/origin_datas_cut/sf_packages/data_12_15/1215_抓快递_LD_r9_242/bg_tablecloth1arm_rightobject_1/episode_2025-12-15_143931_599_part_2"
  # "/data0/origin_datas_cut/sf_packages/data_03_19/0319_抓快递_r9_268/bg_tablecloth1arm_rightobject_1/episode_2026-03-19_160645_222_part_1"
  # "/data0/origin_datas_cut/sf_packages/data_10_31/1031_04_抓放包裹_LD_236/bg_tablecloth1arm_rightobject_1/episode_2025-10-31_134318_563_part_0"
  # "/data0/origin_datas_cut/sf_packages/data_11_03/1103_12_快递_LD_155/bg_tablecloth1arm_rightobject_1/episode_2025-11-03_100909_257_part_5"
  # "/data0/origin_datas_cut/sf_packages/data_03_18/0318_抓快递_r3_136/bg_tablecloth1arm_rightobject_1/episode_2026-03-18_160324_447_part_2"
  # "/data0/origin_datas_cut/sf_packages/data_11_04/1104_06_快递_LD_163/bg_tablecloth1arm_rightobject_1/episode_2025-11-04_111100_547_part_0"
  # "/data0/origin_datas_cut/sf_packages/data_11_05/1105_15_抓放包裹_LD_220/bg_tablecloth1arm_rightobject_1/episode_2025-11-05_160020_555_part_0"

  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_07/1107_04_抓放包裹_LD_199/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_095835_939_part_0_part_2_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_10/1110_12_快递_LD_208/bg_tablecloth1arm_rightobject_1/episode_2025-11-10_093707_238_part_3_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_06/1106_12_抓放包裹_LD_182/bg_tablecloth1arm_rightobject_1/episode_2025-11-06_024030_400_part_2_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_07/1107_12_抓放包裹_LD_209/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_111541_402_part_3_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_12_10/1210_09_快递_LD_195/bg_tablecloth1arm_rightobject_1/episode_2025-12-10_101528_427_part_2_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_03_18/0318_抓快递_r3_136/bg_tablecloth1arm_rightobject_1/episode_2026-03-18_155847_351_part_3_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_04/1104_09_快递_LD_182/bg_tablecloth1arm_rightobject_1/episode_2025-11-04_155022_900_part_0_part_0_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_07/1107_09_抓放包裹_LD_180/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_094930_517_part_2_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_10_31/1031_06_抓放包裹_LD_220/bg_tablecloth1arm_rightobject_1/episode_2025-10-31_154920_068_part_1_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_03_18/0318_抓快递_r3_136/bg_tablecloth1arm_rightobject_1/episode_2026-03-18_155558_416_part_2_part_0"
)
EPISODE_DIRS_FILE=""
EPISODE_GLOB=""

# 输出目录，生成的 png/mp4 都会写到这里。
# 修改后的效果：
# - 所有可视化输出都会落到这个目录下。
OUTPUT_BASE_DIR="/home/rsluo/codes/openpi06/z_valn_output"

# 可选的自定义输出名前缀。
# 修改后的效果：
# - 如果非空，直接使用这个值作为输出名前缀。
# - 如果为空，python 会根据 MODEL_TIME / MODEL_STEP /
#   MODEL_DIM / USE_LEFT / USE_RIGHT / max_len / prompt_type 自动生成名字。
NAME_SAVE=""

# 自动命名时使用的元信息。
# 修改后的效果：
# - MODEL_TIME：体现在输出名里，一般写训练日期或版本。
# - MODEL_STEP：体现在输出名里，一般写 checkpoint step。
# - MODEL_DIM：体现在输出名里，例如 8dim / 16dim。
MODEL_TIME="20260429_30Rt"
MODEL_STEP="75000"
MODEL_DIM="8dim"

# 视频每个 panel 的宽高。
# 修改后的效果：
# - mp4 里的每一格都会被 resize 到这个大小。
# - 现在视频布局是 3 行 2 列：
#   第 1 行：head | out_png
#   第 2 行：left | out_png_At
#   第 3 行：right | out_png_It
# - 原图过小会放大，过大也会缩小到这个尺寸。
VIDEO_PANEL_WIDTH="640"
VIDEO_PANEL_HEIGHT="360"

# 每个 episode 对应一个 prompt_type。
# 修改后的效果：
# - PROMPT_TYPES 的数量必须和 EPISODE_DIRS 数量一致。
# - 第 i 个 episode 使用第 i 个 prompt_type。
# - 真正发给 server 的英文 prompt，会根据这个 prompt_type
#   从 validation_1_hand_valuenet.py 里的 prompt_map 自动查出来。
# - 输出名里也会体现这个 prompt_type。
PROMPT_TYPES=(
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
  "sf包裹"
)

# 选择测试左手、右手还是双手。
# 修改后的效果：
# - USE_LEFT=true,  USE_RIGHT=false：测试左手 state/image。
# - USE_LEFT=false, USE_RIGHT=true ：测试右手 state/image。
# - USE_LEFT=true,  USE_RIGHT=true ：输出名会标成 both，图像布局也会按双手理解；
#   但当前 python 里的 state 选择逻辑是否完全支持双手，还要看你的数据和模型是否真支持。
# - USE_LEFT=false, USE_RIGHT=false：非法配置，python 会直接报错。
USE_LEFT="false"
USE_RIGHT="true"

# 控制 validation 客户端实际发送给 server 的图像 key。
# 必须和 server 端当前 config.image_keys 保持一致，否则会立即报错。
# 默认这里用三路图像；如果 server 配的是四路，再把最后一行打开。
IMAGE_KEYS=(
  "base_0_rgb"
  "left_wrist_0_rgb"
  "right_wrist_0_rgb"
  # "episode_first_head_img"
)

# val 图里额外叠加的 state 维度，0-based。
# 右手默认 7 表示右手第 8 维，也就是夹爪。
PLOT_STATE_INDEX="7"

# 定制化开关：最后 N 帧不走 server infer，直接令 Rt1=Rt0, Rt2=Rt0。
# 默认关闭；打开时仍然会对前面的帧正常走 infer。
CUSTOM_TAIL_DIRECT_RT0="true"
CUSTOM_TAIL_LENGTH="30"

PY_ARGS=()
if [ -n "$EPISODE_DIRS_FILE" ]; then
  PY_ARGS+=(--args.episode-dirs-file "$EPISODE_DIRS_FILE")
elif [ -n "$EPISODE_GLOB" ]; then
  PY_ARGS+=(--args.episode-glob "$EPISODE_GLOB")
elif [ "${#EPISODE_DIRS[@]}" -gt 0 ]; then
  PY_ARGS+=(--args.episode-dirs "${EPISODE_DIRS[@]}")
fi

PY_ARGS+=(--args.output-base-dir "$OUTPUT_BASE_DIR")
PY_ARGS+=(--args.model-time "$MODEL_TIME")
PY_ARGS+=(--args.model-step "$MODEL_STEP")
PY_ARGS+=(--args.model-dim "$MODEL_DIM")
PY_ARGS+=(--args.video-panel-width "$VIDEO_PANEL_WIDTH")
PY_ARGS+=(--args.video-panel-height "$VIDEO_PANEL_HEIGHT")
if [ "${#IMAGE_KEYS[@]}" -gt 0 ]; then
  PY_ARGS+=(--args.image-keys "${IMAGE_KEYS[@]}")
fi
PY_ARGS+=(--args.plot-state-index "$PLOT_STATE_INDEX")
if [ "$CUSTOM_TAIL_DIRECT_RT0" = "true" ]; then
  PY_ARGS+=(--args.custom-tail-direct-rt0)
else
  PY_ARGS+=(--args.no-custom-tail-direct-rt0)
fi
PY_ARGS+=(--args.custom-tail-length "$CUSTOM_TAIL_LENGTH")

if [ "${#PROMPT_TYPES[@]}" -gt 0 ]; then
  PY_ARGS+=(--args.prompt-types "${PROMPT_TYPES[@]}")
fi

if [ -n "$NAME_SAVE" ]; then
  PY_ARGS+=(--args.name-save "$NAME_SAVE")
fi

if [ "$USE_LEFT" = "true" ]; then
  PY_ARGS+=(--args.use-left)
else
  PY_ARGS+=(--args.no-use-left)
fi

if [ "$USE_RIGHT" = "true" ]; then
  PY_ARGS+=(--args.use-right)
else
  PY_ARGS+=(--args.no-use-right)
fi

python validation_1_hand_valuenet_multi.py "${PY_ARGS[@]}"
