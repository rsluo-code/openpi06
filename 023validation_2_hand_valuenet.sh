eval "$("/home/rsluo/miniconda3/bin/conda" shell.bash hook)"
conda activate pi06_env
export TORCHINDUCTOR_DISABLE=1
chmod +x /home/rsluo/miniconda3/envs/pi06_env/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas
echo "hello"
python validation_2_hand_valuenet.py 