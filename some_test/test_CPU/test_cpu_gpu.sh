#!/bin/bash
# Linux系统CPU/内存/GPU持续监控脚本（5秒/次，单/多GPU自适应，时间戳日志）
# 修复：GPU字段解析错乱 + 去掉set -e避免小错误直接退出
set -uo pipefail  # 去掉-e，保留-u和pipefail

# 【可自定义配置项】
INTERVAL=5
LOG_FILE="./sys_resource_continuous_$(date +'%Y%m%d_%H%M%S').log"
TMP_DATA="./.tmp_resource_data.tmp"
TMP_GPU="./.tmp_gpu.tmp"

# 初始化
> $TMP_DATA
echo -e "=====================================" >> $LOG_FILE
echo "【监控开始】$(date +'%Y-%m-%d %H:%M:%S') | 采集频率：${INTERVAL}秒/次 | 按Ctrl+C停止并自动统计平均值" >> $LOG_FILE
echo -e "-------------------------------------\n" >> $LOG_FILE


# 核心采集循环
echo -e "📌 系统资源监控中...（单/多GPU自动适配）"
echo -e "📌 采集频率：${INTERVAL}秒/次 | 按 [Ctrl+C] 停止并自动统计平均值"
echo -e "📌 本次监控独立日志：$LOG_FILE"
while true; do
    CURRENT_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "【采集时间】$CURRENT_TIME" >> $LOG_FILE

    # 1. CPU采集（容错处理，避免top命令异常导致退出）
    CPU_INFO=$(top -bn1 2>/dev/null | grep -E '^%Cpu' | awk '{printf "%.1f,%.1f", $2, $4}' || echo "0.0,0.0")
    CPU_US=$(echo $CPU_INFO | cut -d',' -f1)
    CPU_SY=$(echo $CPU_INFO | cut -d',' -f2)
    echo -e "CPU信息：用户态占用=${CPU_US}% | 系统态占用=${CPU_SY}%" >> $LOG_FILE

    # 2. 内存采集
    MEM_TOTAL_B=$(free -b 2>/dev/null | grep 内存： | awk '{print $2}' || echo 1)
    MEM_USED_B=$(free -b 2>/dev/null | grep 内存： | awk '{print $3}' || echo 0)
    MEM_HUMAN=$(free -h 2>/dev/null | grep 内存： | awk '{print "总="$2", 已用="$3", 空闲="$4}' || echo "总=0, 已用=0, 空闲=0")
    echo -e "内存信息：${MEM_HUMAN} " >> $LOG_FILE

    # 3. GPU采集（核心修复：兼容字段格式，容错处理）
    GPU_MEM_USED_STAT=$(nvidia-smi 2>/dev/null | grep Default  || echo 1)

    # 4. 写入临时文件
    echo "显存信息： ${GPU_MEM_USED_STAT}" >> $LOG_FILE

    echo -e "-------------------------------------\n" >> $LOG_FILE
    sleep $INTERVAL
done
