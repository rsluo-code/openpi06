1. PI06_pretrain.model.if_use_valuenet=False -> True
  2. PI06_pretrain.data.image_keys 显式加上 "episode_first_head_img"
  3. pi06_pytorch.py 里的 valuenet_checkpoint_dir 改到 20260423/80000  帮我改好
