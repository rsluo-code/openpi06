import torch
import os
def count_parameters(model: torch.nn.Module):
    """返回 (总参数量, 可训练参数量)。"""
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable

def print_parameter_stats(model: torch.nn.Module,model_name:str):
    """打印整体参数量"""

    # 如果是 DDP，拿里面的真实模型
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    total, trainable = count_parameters(model)
    print(f"[{model_name}] Total params: {total:,} "
                f"({total/1e6:.2f}M, {total/1e9:.3f}B)")
    print(f"[{model_name}] Trainable params: {trainable:,} "
                f"({trainable/1e6:.2f}M, {trainable/1e9:.3f}B)")


def load_param_from_ckpt_dir(model, ckpt_dir: str,model_name:str,_strict=False):

    ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
    print(f"[{model_name}] Loading Pi05 prefix-only weights from: {ckpt_path}")

    from safetensors.torch import load_file
    # 1. 读 ckpt
    src_sd = load_file(ckpt_path, device="cpu")

    # 2. 取出当前模型的 state_dict（注意 DDP 的情况）
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    target = model.module if is_ddp else model
    tgt_sd = target.state_dict()

    # 3. 只保留 paligemma_with_expert.paligemma.* 并且在当前模型中存在且 shape 一致的参数
    filtered_sd = {}
    for k, v in src_sd.items():
        # 当前 model_name 里必须也有同名参数，并且 shape 一致
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            filtered_sd[k] = v

    print(f"[{model_name}] Will load {len(filtered_sd)} parameters into itself")

    # 4. 只用 filtered_sd 来覆盖（strict=False 可以忽略没有加载到的 value_head 等）
    missing, unexpected = target.load_state_dict(filtered_sd, strict=_strict)

    print(f"[{model_name}] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[{model_name}] missing keys: {missing}")
    if unexpected:
        print(f"[{model_name}] unexpected keys from ckpt (已忽略): {unexpected}")




from dataclasses import dataclass

@dataclass(frozen=True)
class TaskInfo:
    zh: str
    en: str
    max_len: int
    at_30precent: float
AT_30PRECENT= -0.0050
TASKS = {
    1: TaskInfo("纸巾", "pick up a pack of tissues and put it on the plate", 757, 0.030 ),
    2: TaskInfo("魔方", "pick up the Rubik's cube and put it on the plate", 898, -0.0070 ),
    3: TaskInfo("苹果", "pick up the apple and put it on the plate", 761, 0.0100 ),
    4: TaskInfo("香蕉", "pick up the banana and put it on the plate", 809, 0.008 ),
    5: TaskInfo("碗", "pick up the bowl and place it on the plate", 717, 0.028 ),
    6: TaskInfo("可乐", "pick up the cola and put it on the plate", 833, -0.007 ),
    7: TaskInfo("手电筒", "pick up the flashlight and put it on the plate", 644, AT_30PRECENT ),
    8: TaskInfo("猕猴桃", "pick up the kiwi and put it on the plate", 801, 0.008 ),
    9: TaskInfo("柠檬", "pick up the lemon and put it on the plate", 744, -0.0090 ),
    10: TaskInfo("芒果", "pick up the mango and put it on the plate", 805, 0.005 ),
    11: TaskInfo("山竹", "pick up the mangosteen and put it on the plate", 815, -0.0015 ),
    12: TaskInfo("卷尺", "pick up the measure and put it on the plate", 690, -0.0015 ),
    13: TaskInfo("面包", "pick up the packaged snack cakes and put it on the plate", 694, 0.001 ),
    14: TaskInfo("杯子", "pick up the paper cup and put it on the plate", 694, -0.0015 ),
    15: TaskInfo("桃子", "pick up the peach and put it on the plate", 986, -0.0090 ),
    16: TaskInfo("布绒玩具", "pick up the plush toy and put it on the plate", 837, -0.0035 ),
    17: TaskInfo("饼干", "pick up the rectangular box-packed biscuits and put it on the plate", 1140, 0.011 ),
    18: TaskInfo("螺丝刀", "pick up the screwdriver and put it on the plate", 776, AT_30PRECENT ),
    19: TaskInfo("盲盒", "pick up the square paper box and put it on the plate", 881, 0.016 ),
    20: TaskInfo("茶饮", "pick up the tea and put it on the plate", 865, -0.0120 ),
    21: TaskInfo("毛巾", "pick up the towel and put it on the plate", 631, 0.0030 ),
    22: TaskInfo("矿泉水", "pick up the water and put it on the plate", 792, 0.012 ),
    23: TaskInfo("翻转包裹", "If the label is face-up, flip the package and put it in the basket.", 1200, 0.012 ),
    24: TaskInfo("SF_快递抓取", "Pick and Place package to the right part", 700, 0.007 ),

}
def build_en_to_task(tasks: dict[int, TaskInfo]) -> dict[str, tuple[int, int]]:
    """
    构建反向索引：en -> (index, max_len)

    注意：
    - 这个表是从 TASKS 自动派生出来的，不需要你手写维护
    - 如果 en 有重复，会直接报错，避免悄悄覆盖
    """
    en_to_task: dict[str, tuple[int, int]] = {}
    for idx, info in tasks.items():
        en = info.en
        if en in en_to_task:
            raise ValueError(f"Duplicate en instruction found: {en}")
        en_to_task[en] = (idx, info.max_len,info.at_30precent)
    return en_to_task

# 程序启动时构建一次，后面训练循环里直接查
EN_TO_TASK = build_en_to_task(TASKS)

import tensorflow as tf
# 1) 把 Python dict 里的 keys/values 做成 TF 常量
_EN_KEYS = tf.constant(list(EN_TO_TASK.keys()), dtype=tf.string)

# index 和 max_len 分别做成两个 value tensor
_EN_INDEX_VALS = tf.constant([v[0] for v in EN_TO_TASK.values()], dtype=tf.int32)
_EN_MAXLEN_VALS = tf.constant([v[1] for v in EN_TO_TASK.values()], dtype=tf.int32)
_EN_At30_VALS = tf.constant([v[2] for v in EN_TO_TASK.values()], dtype=tf.float32)

# 2) 构建两个静态 HashTable（也可以合成一个表，但两个表最直观）
_INDEX_TABLE = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(_EN_KEYS, _EN_INDEX_VALS),
    default_value=tf.constant(-1, dtype=tf.int32),   # 查不到时返回 -1
)

_MAXLEN_TABLE = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(_EN_KEYS, _EN_MAXLEN_VALS),
    default_value=tf.constant(-1, dtype=tf.int32),
)

_AT30_TABLE = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(_EN_KEYS, _EN_At30_VALS),
    default_value=tf.constant(AT_30PRECENT, dtype=tf.float32),
)

def get_index_and_max_len(en_instruction):
    """
    支持两种输入：
    - Python str（你在普通代码里调用）
    - tf.Tensor(tf.string)（你在 tf.data map 里调用）
    返回：
    - index: tf.int32 或 int
    - max_len: tf.int32 或 int
    """
    # tf.data/map 场景：en_instruction 是 tf.Tensor
    if tf.is_tensor(en_instruction):
        idx = _INDEX_TABLE.lookup(en_instruction)
        max_len = _MAXLEN_TABLE.lookup(en_instruction)
        at_30precent = _AT30_TABLE.lookup(en_instruction)
        return idx, max_len,at_30precent

    # 普通 Python 场景：en_instruction 是 str
    idx, max_len,at_30precent = EN_TO_TASK[en_instruction]
    return idx, max_len,at_30precent
