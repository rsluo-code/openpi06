import os
import json
import numpy as np
import glob
import tensorflow_datasets as tfds
import tensorflow as tf
import sys
from collections import defaultdict

# 添加 dlimp 路径
# base_path = "/home/rsluo/codes/rlds_factory/rlds_vis/src/dlimp"
# sys.path.append(base_path)
import dlimp as dl

# ===================== 配置项 =====================
# 根目录（两级目录下有 part 子目录）
ROOT_DATA_DIR = "/wx-mix01/sppro/permanent/yuanzhang10/rlds_data/for_rsluo/sppro_linden_zhuawupin_v4/0930_1114/"
# 数据集名称
DATASET_NAME = "rlds_example_dataset"
# ==================== 工具函数 ====================
def safe_decode_bytes(obj, encoding="utf-8", errors="strict"):
    """轻量版解码函数：仅处理 str/bytes/numpy 数组"""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj).decode(encoding, errors=errors)
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            def _dec(x):
                if isinstance(x, (bytes, bytearray, memoryview)):
                    return bytes(x).decode(encoding, errors=errors)
                return x
            vdec = np.vectorize(_dec, otypes=[object])
            return vdec(obj)
        if obj.dtype.kind == 'S':
            try:
                return np.char.decode(obj, encoding=encoding)
            except Exception:
                return np.char.decode(obj, encoding=encoding, errors='ignore')
        return obj
    return obj

# ==================== 核心统计逻辑 ====================
def count_language_instructions():
    # 初始化统计字典：key=指令文本，value=出现次数
    instr_counter = defaultdict(int)
    # 遍历所有 part 目录（匹配 ROOT_DATA_DIR 下两级的 part 目录）
    part_dir_pattern = os.path.join(ROOT_DATA_DIR, "*", "*part*")  # 匹配两级下的 part 目录
    part_dirs = glob.glob(part_dir_pattern)
    
    if not part_dirs:
        print(f"⚠️ 未找到任何 part 目录，匹配路径：{part_dir_pattern}")
        return
    
    print(f"📌 找到 {len(part_dirs)} 个 part 目录：")
    for p in part_dirs:
        print(f"   - {p}")
    
    # 遍历每个 part 目录
    for part_dir in part_dirs:
        # 检查数据集是否存在
        dataset_path = os.path.join(part_dir, DATASET_NAME, "1.0.0")
        if not os.path.exists(dataset_path):
            print(f"⚠️ {part_dir} 未完成转换，跳过")
            continue
        
        # 加载数据集
        try:
            builder = tfds.builder(DATASET_NAME, data_dir=part_dir, version="1.0.0")
            dataset = dl.DLataset.from_rlds(
                builder,
                split="train",
                shuffle=False,
                num_parallel_reads=tf.data.AUTOTUNE,
            )
            dataset = dataset.with_ram_budget(1)
            N = len(dataset)
            print(f"\n📊 处理 {part_dir}：共 {N} 条轨迹")
            
            # 遍历数据集样本
            for idx, item in enumerate(dataset.as_numpy_iterator()):
                # 解码 language_instruction
                decoded_instr = safe_decode_bytes(item["language_instruction"], errors="ignore")
                # 取第一条指令（确保是原生字符串）
                first_instr = decoded_instr[0] if isinstance(decoded_instr, (list, np.ndarray)) else decoded_instr
                if isinstance(first_instr, np.generic):
                    first_instr = first_instr.item()
                first_instr = str(first_instr).strip()  # 去除首尾空格
                
                # 统计次数
                instr_counter[first_instr] += 1
                
        except Exception as e:
            print(f"❌ 处理 {part_dir} 出错：{e}")
            continue
    
    # ==================== 输出统计结果 ====================
    print("\n" + "="*50)
    print("📈 language_instruction 统计结果（按出现次数降序）：")
    print("="*50)
    # 按出现次数排序
    sorted_instr = sorted(instr_counter.items(), key=lambda x: x[1], reverse=True)
    if not sorted_instr:
        print("   无有效指令数据")
    else:
        for idx, (instr, count) in enumerate(sorted_instr, 1):
            print(f"{idx:2d}. 指令：{instr}")
            print(f"    次数：{count}")
            print("-"*30)
    
    # 可选：保存统计结果到文件
    stats_file = os.path.join(ROOT_DATA_DIR, "language_instruction_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(dict(instr_counter), f, ensure_ascii=False, indent=4)
    print(f"\n💾 统计结果已保存到：{stats_file}")

if __name__ == "__main__":
    count_language_instructions()