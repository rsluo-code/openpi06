ulimit -u 200000
export CUDA_VISIBLE_DEVICES=5,6,7
# export CUDA_VISIBLE_DEVICES=3,4,5,7
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
GPUS_PER_NODE=1
NNODES=1
MASTER_PORT=27345
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
export JAX_TRACEBACK_FILTERING=off

source scripts/env_CUDA.bashrc

eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"

# source  /wx-home/sppro/rsluo/.bashrc

# /wx-home/sppro/rsluo/anaconda3/bin/conda init
# /wx-home/sppro/rsluo/anaconda3/bin/conda activate pi06_env
conda activate pi06_env

echo "PYTHON  = $(which python)"
echo "TORCHRUN = $(which torchrun)"
echo "JAX_TRACEBACK_FILTERING = $JAX_TRACEBACK_FILTERING"
# cd pdb_test
# python3 ./01test_dlimp.py
# cd ..

# torchrun --nproc_per_node ${GPUS_PER_NODE} \
#          --nnodes ${NNODES} \
#          --node_rank ${RANK} \
#          --master_addr ${MASTER_ADDR} \
#          --master_port ${MASTER_PORT} \
#          scripts/train_pytorch_pi06.py PI06_pretrain --exp_name pi06_train_1231_2b_16dim

# 替换原有的torchrun命令
NAME="PI05_bc"

SDIR="./"
echo "$SDIR"
mkdir -p "$SDIR/debug"

TS="$(date '+%Y%m%d_%H%M%S')"  
LOG="$SDIR/debug/${NAME}_${TS}.txt"

python -u -m torch.distributed.run \
         --nproc_per_node ${GPUS_PER_NODE} \
         --nnodes ${NNODES} \
         --node_rank ${RANK} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         scripts/train_bc_pytorch.py PI05_bc --exp_name pi05bc_train_20260128_16dim_noexchange \
            2>&1 \
            | tee "$LOG"



# conda activate pi06_env
# python3 scripts/train_pytorch_valuenet.py PI06_pretrain --exp_name pi06_train_1231_2b_16dim