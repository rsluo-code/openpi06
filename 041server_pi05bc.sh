eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env

export TORCHINDUCTOR_DISABLE=1
chmod +x /home/rsluo/miniconda3/envs/pi06_env/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas

export CUDA_VISIBLE_DEVICES=5

python scripts/serve_policy.py policy:checkpoint \
        --policy.config=PI05_bc \
        --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/PI05_bc/pi05bc_train_20260128_16dim_noexchange/60000
