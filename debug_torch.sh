ulimit -u 200000
ulimit -a
source /home5/cv5/jfni3/.bashrc
conda activate /home5/cv5/jfni3/.conda/envs/pi_torch
export NCCL_NET_GDR_LEVEL=2
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=6

GPUS_PER_NODE=1
NNODES=1
MASTER_PORT=${MASTER_PORT:-22345}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    scripts/train_pytorch.py linden_torch --exp_name sl_v4_pose