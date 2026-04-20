eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

python ./compute_norm_stats.py --config-name norm_state_16dim
# python ./compute_norm_stats_two_arm_exchange_size.py --config-name norm_state_16dim