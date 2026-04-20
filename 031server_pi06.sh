eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old

export TORCHINDUCTOR_DISABLE=1
chmod +x /home/rsluo/miniconda3/envs/pi06_env_old/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas

export CUDA_VISIBLE_DEVICES=4

# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=PI06_pretrain_validation \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/PI06_pretrain/pi06_train_20260109_16dim/30000


python scripts/serve_policy.py policy:checkpoint \
        --policy.config=PI06_pretrain_validation \
        --policy.dir=/data0/rsluo/pi06_torch/PI06_pretrain/sf_packages_rightarm_20260413/55000

        