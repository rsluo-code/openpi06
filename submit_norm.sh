task_id=${1}
echo ${task_id}

/opt/dls_cli/ky exp submit \
  -a jfni3 \
  -n train-${task_id} \
  -d train-${task_id} \
  -e "compute_norm.sh" \
  -l logs/norm.sh_logStdout.log \
  -o logs/norm.sh_logStderr.log \
  -i reg.deeplearning.cn/dlaas/cv_dist_openmpi:0.3 \
  --modelName compute_norm\
  --modelPath /b3-mix03/sppro/permanent/jfni3/pi05_checkpoints \
  --modelVersion compute_norm\
  --faultRetry \
  --iXunFeiOnStatus \
  -k TeslaA800-NVLINK-80GB  \
  -r dlp3-sppro-cogllm-reserved\
  --proID 1365 \
  --useGpu \
  -g 8 \
  -w 1 \
  -c 100 \
  -m 1920 \
  -t PtJob \