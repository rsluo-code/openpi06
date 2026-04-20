task_id=${1}
echo ${task_id}
NAME="comput_norm_stats"
TS="$(date '+%Y%m%d_%H%M%S')"  
LOGStdout="logs/${NAME}_${TS}.sh_logStdout.log"
LOGStderr="logs/${NAME}_${TS}.sh_logStderr.log"

/wx-opt/dls_cli/ky exp submit \
  -a yuanzhang10 \
  -n train-${task_id} \
  -d train-${task_id} \
  -e "compute_norm_state.sh" \
  -l ${LOGStdout} \
  -o ${LOGStderr} \
  -i reg.deeplearning.cn/ky/shell-ubuntu-dlp:22.04 \
  --experimentName train-${task_id} \
  --modelName train-${task_id}\
  --modelPath log/1224${task_id} \
  --noCheckStuck \
  --iXunFeiOnStatus \
  -k HopperH200-NVLINK-141GB  \
  -r wxdlp3-sppro-cogllm-reserved\
  --proID 1365 \
  --useGpu \
  -g 1 \
  -w 1 \
  -c 19 \
  -m 370 \
  -t PtJob \

# -w 选择多少个机器，一般一个机器8卡
# -m 多少内存
# -c 多少核心
# --proID 项目id