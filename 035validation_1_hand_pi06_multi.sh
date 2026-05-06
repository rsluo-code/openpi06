eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old
export TORCHINDUCTOR_DISABLE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-openpi06}"
mkdir -p "$MPLCONFIGDIR"
chmod +x /home/rsluo/miniconda3/envs/pi06_env_old/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas 2>/dev/null || true
echo "hello"

# episode 来源，优先级：
#   1) EPISODE_DIRS_FILE
#   2) EPISODE_GLOB
#   3) EPISODE_DIRS
EPISODE_DIRS=(
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_07/1107_04_抓放包裹_LD_199/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_095835_939_part_0_part_2_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_10/1110_12_快递_LD_208/bg_tablecloth1arm_rightobject_1/episode_2025-11-10_093707_238_part_3_part_0"
  "/data0/origin_datas_cut_only_pick/sf_packages/data_11_06/1106_12_抓放包裹_LD_182/bg_tablecloth1arm_rightobject_1/episode_2025-11-06_024030_400_part_2_part_0"
)
EPISODE_DIRS_FILE=""
EPISODE_GLOB=""

OUTPUT_BASE_DIR="/home/rsluo/codes/openpi06/z_pi06_output"
NAME_SAVE=""
MODEL_TIME="20260506"
MODEL_STEP="100000"
MODEL_DIM="8dim"

PROMPT_TYPES=(
  "sf包裹"
  "sf包裹"
  "sf包裹"
)

USE_LEFT="false"
USE_RIGHT="true"

PY_ARGS=()
if [ -n "$EPISODE_DIRS_FILE" ]; then
  PY_ARGS+=(--episode-dirs-file "$EPISODE_DIRS_FILE")
elif [ -n "$EPISODE_GLOB" ]; then
  PY_ARGS+=(--episode-glob "$EPISODE_GLOB")
elif [ "${#EPISODE_DIRS[@]}" -gt 0 ]; then
  PY_ARGS+=(--episode-dirs "${EPISODE_DIRS[@]}")
fi

PY_ARGS+=(--output-base-dir "$OUTPUT_BASE_DIR")
PY_ARGS+=(--model-time "$MODEL_TIME")
PY_ARGS+=(--model-step "$MODEL_STEP")
PY_ARGS+=(--model-dim "$MODEL_DIM")

if [ "${#PROMPT_TYPES[@]}" -gt 0 ]; then
  PY_ARGS+=(--prompt-types "${PROMPT_TYPES[@]}")
fi

if [ -n "$NAME_SAVE" ]; then
  PY_ARGS+=(--name-save "$NAME_SAVE")
fi

if [ "$USE_LEFT" = "true" ]; then
  PY_ARGS+=(--use-left)
else
  PY_ARGS+=(--no-use-left)
fi

if [ "$USE_RIGHT" = "true" ]; then
  PY_ARGS+=(--use-right)
else
  PY_ARGS+=(--no-use-right)
fi

python validation_1_hand_pi06_imgN_multi.py "${PY_ARGS[@]}"
