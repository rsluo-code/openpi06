task_id=${1}
echo ${task_id}

/opt/dls_cli/ky exp submit \
  -a jfni3 \
  -n train-${task_id} \
  -d train-${task_id} \
  -e "train_torch.sh" \
  -l logs/sl_v4_1013.sh_logStdout.log \
  -o logs/sl_v4_1013.sh_logStderr.log \
  -i reg.deeplearning.cn/ky/nvidia-cuda:11.5.0-cudnn8-devel-centos7-ofed5.7-gdr-20230109 \
  --modelName train-pi0-torch2\
  --modelPath /b3-mix03/sppro/permanent/jfni3/pi05_checkpoints/linden_torch \
  --modelVersion train-pi0-torch2\
  --faultRetry \
  --iXunFeiOnStatus \
  -k TeslaA800-NVLINK-80GB  \
  -r dlp3-sppro-cogllm-reserved\
  --proID 1365 \
  --useGpu \
  -g 8 \
  -w 2 \
  -c 200 \
  -m 1920 \
  -t PtJob \