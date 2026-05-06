ulimit -u 200000
ulimit -a
export NCCL_NET_GDR_LEVEL=2
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

GPUS_PER_NODE=8
NNODES=1
MASTER_PORT=${MASTER_PORT:-22345}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"

conda activate pi06_env_old

echo "PYTHON  = $(which python)"
echo "TORCHRUN = $(which torchrun)"
cd pdb_test
python3 ./01test_dlimp.py
cd ..

# torchrun --nproc_per_node ${GPUS_PER_NODE} \
#          --nnodes ${NNODES} \
#          --node_rank ${RANK} \
#          --master_addr ${MASTER_ADDR} \
#          --master_port ${MASTER_PORT} \
#          scripts/train_pytorch_valuenet.py value_pretrain --exp_name value_pretrain_1205
# 替换原有的torchrun命令
NAME="train_valuenet"

SDIR="/home/rsluo/codes/openpi06/"
echo "$SDIR"
export PYTHONPATH="$SDIR/dlimp:$SDIR/src:${PYTHONPATH:-}"
mkdir -p "$SDIR/debug"

TS="$(date '+%Y%m%d_%H%M%S')"  
LOG="$SDIR/debug/${NAME}_${TS}.txt"

python -m torch.distributed.run \
            --nproc_per_node ${GPUS_PER_NODE} \
            --nnodes ${NNODES} \
            --node_rank ${RANK} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            scripts/train_pytorch_valuenet.py value_pretrain_16dim --exp_name sf_packages_rightarm_20260428 \
            2>&1 \
            | tee "$LOG"
         
# conda activate pi06_env
# python3 scripts/train_pytorch_valuenet.py value_pretrain --exp_name value_pretrain_1205
