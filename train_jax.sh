ulimit -u 100000
source /home5/cv5/jfni3/.bashrc
conda activate /work2/cv5/jfni3/miniconda3/envs/pi05

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS="--xla_gpu_autotune_level=0" python scripts/train.py linden_jfni3 --exp-name=sl_v3_1003 --overwrite