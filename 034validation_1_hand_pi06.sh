eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old
export TORCHINDUCTOR_DISABLE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-openpi06}"
mkdir -p "$MPLCONFIGDIR"
chmod +x /home/rsluo/miniconda3/envs/pi06_env_old/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas 2>/dev/null || true
echo "hello"

EPISODE_DIR="/data0/origin_datas_cut_only_pick/sf_packages/data_11_07/1107_04_抓放包裹_LD_199/bg_tablecloth1arm_rightobject_1/episode_2025-11-07_095835_939_part_0_part_2_part_0"
OUTPUT_BASE_DIR="/home/rsluo/codes/openpi06/z_pi06_output"
NAME_SAVE=""
MODEL_TIME="20260506"
MODEL_STEP="100000"
MODEL_DIM="8dim"
PROMPT_TYPE="sf包裹"
USE_LEFT="false"
USE_RIGHT="true"

PY_ARGS=(
  --episode-dir "$EPISODE_DIR"
  --output-base-dir "$OUTPUT_BASE_DIR"
  --model-time "$MODEL_TIME"
  --model-step "$MODEL_STEP"
  --model-dim "$MODEL_DIM"
  --prompt-type "$PROMPT_TYPE"
)

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

python validation_1_hand_pi06_imgN.py "${PY_ARGS[@]}"
