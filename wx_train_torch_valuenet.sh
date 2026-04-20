ulimit -u 200000
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=2,3,4,5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "hello1"
GPUS_PER_NODE=8
NNODES=1
MASTER_PORT=26345
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
echo "hello2"

source scripts/env_CUDA.bashrc
echo "hello3"

eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
echo "hello4"

# source  /wx-home/sppro/rsluo/.bashrc

# /wx-home/sppro/rsluo/anaconda3/bin/conda init
# /wx-home/sppro/rsluo/anaconda3/bin/conda activate pi06_env
conda activate pi06_env
echo "hello5"

echo "PYTHON  = $(which python)"
echo "TORCHRUN = $(which torchrun)"
# cd pdb_test
# python3 ./01test_dlimp.py
# cd ..
echo "hello6"

# torchrun --nproc_per_node ${GPUS_PER_NODE} \
#          --nnodes ${NNODES} \
#          --node_rank ${RANK} \
#          --master_addr ${MASTER_ADDR} \
#          --master_port ${MASTER_PORT} \
#          scripts/train_pytorch_valuenet.py value_pretrain_16dim --exp_name valuenet_train_20260106_300m_16dim_exchangsize


NAME="valuenet_pretrain"

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
         scripts/train_pytorch_valuenet_2.py value_pretrain_right_arm_8dim_sf_packages --exp_name valuenet_20260331_sf_packages \
            2>&1 \
            | tee "$LOG"
         
# conda activate pi06_env
# python3 scripts/train_pytorch_valuenet.py value_pretrain --exp_name value_pretrain_1205