
MY_CONDA_PATH="/home/rsluo/miniconda3/bin/conda"
eval "$($MY_CONDA_PATH shell.bash hook)"
python3 check_pi06_value_model_diff.py PI06_pretrain --exp_name PI06_pretrain_check