eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env_old

export TORCHINDUCTOR_DISABLE=1
chmod +x /home/rsluo/miniconda3/envs/pi06_env_old/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas

export CUDA_VISIBLE_DEVICES=1
# # 330m
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_train_1229_300m_16dim/55000

## 20260109 2b 7w
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260109_2b_16dim_exchangsize/70000
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260109_2b_16dim_exchangsize/65000


## 20260110 2b  base7w+10w
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260110_2b_16dim_exchangsize_base_7w/100000


# # 20260113 2b 16w
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260113_2b_16dim_exchangsize/160000
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260113_2b_16dim_exchangsize/115000
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260113_2b_16dim_exchangsize/130000
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260113_2b_16dim_exchangsize/50000

# # 20260116 2b w
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260116_2b_16dim_exchangimg/315000
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260116_2b_16dim_exchangimg/110000
        # --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260116_2b_16dim_exchangimg/25000


# # 20260128 no exchange 8dim valbase itself
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260128_2b_8dim_noexchangimg_only5item/45000


# 20260203 no exchange 8dim valbase 90maxlen
# python scripts/serve_policy.py policy:checkpoint \
#         --policy.config=value_pretrain_16dim \
#         --policy.dir=/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260203_2b_8dim_noexchangimg_only5item_valbase90maxlen/100000



# 20260420 no exchange 8dim valbase 90maxlen
python scripts/serve_policy.py policy:checkpoint \
        --policy.config=value_pretrain_16dim \
        --policy.dir=/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260413/80000