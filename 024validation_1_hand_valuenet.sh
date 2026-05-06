eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old
chmod +x /home/rsluo/miniconda3/envs/pi06_env_old/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas
echo "hello"

# 这里控制 validation 客户端实际发送给 server 的图像 key。
# 注意必须和 server 端当前 config.image_keys 保持一致，否则 server 会直接报缺字段。
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
CUSTOM_TAIL_DIRECT_RT0="false"
CUSTOM_TAIL_LENGTH="30"

PY_ARGS=()
if [ "${#IMAGE_KEYS[@]}" -gt 0 ]; then
  PY_ARGS+=(--image-keys "${IMAGE_KEYS[@]}")
fi
PY_ARGS+=(--plot-state-index "$PLOT_STATE_INDEX")
if [ "$CUSTOM_TAIL_DIRECT_RT0" = "true" ]; then
  PY_ARGS+=(--custom-tail-direct-rt0)
else
  PY_ARGS+=(--no-custom-tail-direct-rt0)
fi
PY_ARGS+=(--custom-tail-length "$CUSTOM_TAIL_LENGTH")

if [ -n "${EPISODE_GLOB:-}" ]; then
  python validation_1_hand_valuenet.py --episode_glob "${EPISODE_GLOB}" "${PY_ARGS[@]}"
else
  python validation_1_hand_valuenet.py "${PY_ARGS[@]}"
fi
