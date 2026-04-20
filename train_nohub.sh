#!/bin/bash
# start_training.sh

SCRIPT_NAME="train_torch.sh"
LOG_FILE="/DATA/disk2/1024_grasp_packages/code/pi0_code/openpi06/logs/training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="training.pid"

echo "Starting training at $(date)" | tee -a $LOG_FILE

# 启动任务
nohup bash $SCRIPT_NAME > $LOG_FILE 2>&1 &

# 获取PID
TRAINING_PID=$!
echo $TRAINING_PID > $PID_FILE

# disown 进程
disown $TRAINING_PID

echo "Training started with PID: $TRAINING_PID" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "PID file: $PID_FILE" | tee -a $LOG_FILE