eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python ./scripts/compute_norm_stats.py --config-name norm_state_16dim