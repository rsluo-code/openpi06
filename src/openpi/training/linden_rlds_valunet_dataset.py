"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path
import tqdm

import openpi.shared.download as download
import os
import glob
import torch.distributed as dist
import re

from openpi.models_pytorch.some_func import get_index_and_max_len


DEFAULT_EPISODE_BLACKLIST_TXT_PATHS = [
    "/home/rsluo/codes/cut_episode_for_pi06/debug/20260415_153549_need_recut_episodes.txt",
    "/home/rsluo/codes/cut_episode_for_pi06/debug/20260415_153549_short_episodes_lt_100.txt",
]


def _load_episode_blacklist(data_config) -> list[str]:
    blacklist = list(getattr(data_config, "rlds_episode_blacklist", []) or [])
    blacklist_paths = getattr(data_config, "rlds_episode_blacklist_paths", None)
    if blacklist_paths is None:
        blacklist_paths = getattr(data_config, "rlds_episode_blacklist_path", None)
    if blacklist_paths is None:
        blacklist_paths = DEFAULT_EPISODE_BLACKLIST_TXT_PATHS
    if isinstance(blacklist_paths, str):
        blacklist_paths = [
            path.strip()
            for item in blacklist_paths.split(",")
            for path in item.split(":")
            if path.strip()
        ]
    if blacklist_paths:
        for blacklist_path in blacklist_paths:
            with open(blacklist_path, "r", encoding="utf-8") as file_obj:
                blacklist.extend(
                    line.strip()
                    for line in file_obj
                    if line.strip() and not line.lstrip().startswith("#")
                )
    print(f"读取到episdoe的bliacklist:{len(blacklist)}")
    return sorted(set(os.path.abspath(path) for path in blacklist))


def _build_file_path_blacklist_regex(blacklist: list[str]) -> str:
    escaped_paths = [re.escape(path) for path in blacklist]
    # sppro_rlds_dataset_builder.py stores keys as:
    #   f"{episode_dir_str}_part{part_idx}"
    # Therefore a blacklist entry for the original episode path should also
    # reject all generated RLDS chunks with the trailing _partN suffix.
    return r"^(?:" + "|".join(escaped_paths) + r")(?:_part[0-9]+)?$"


class LindenRldsValueNetDataset:
    def __init__(
        self,
        # data_dir: str,
        data_config,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 30,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 5000, #250_000,
        num_parallel_reads: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # use_left: bool = False,
        # use_right: bool = False,
        # left_size_joint_num: int = 8,
        # right_size_joint_num: int = 8,
    ):
        data_dir = data_config.rlds_data_dir
        use_left = data_config.use_left
        use_right = data_config.use_right
        left_size_joint_num = data_config.left_size_joint_num
        right_size_joint_num = data_config.right_size_joint_num
        image_keys = tuple(getattr(data_config, "image_keys", ()))
        use_episode_first_head_img = "episode_first_head_img" in image_keys
        episode_blacklist = _load_episode_blacklist(data_config)
        
        if use_left == False and use_right == False:
            raise RuntimeError(f"LindenRldsValueNetDataset use_left and use_right must one be True")
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        print("="*25,"rsluo_test LindenRldsValueNetDataset","="*25)
        print("dlimp at:", dl.__file__)
        print("Has DLataset?", hasattr(dl, "DLataset"))
        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")


        # 获取当前分布式环境的状态：卡号 (rank) 和总卡数 (world_size)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_data_dirs = glob.glob(f"{data_dir}/package_sorting_*")
        
        # 定义黑名单列表，可以放多个路径或部分匹配
        blacklist = [
            # os.path.join(data_dir, "data_11_05", "package_sorting_*"),
            # os.path.join(data_dir, "data_11_06", "package_sorting_*"),
            # os.path.join(data_dir, "data_11_07", "package_sorting_*"),
        ]
        if len(blacklist)>0:
            # 过滤掉黑名单
            filtered_data_dirs = [
                d for d in all_data_dirs
                if not any(glob.fnmatch.fnmatch(d, b) for b in blacklist)
            ]
            # 之后使用 filtered_data_dirs
            all_data_dirs = filtered_data_dirs    

        print(f"world_size={world_size}")
        print(f"all_data_dirs has ={len(all_data_dirs)} part")
        print(f"episode_blacklist has ={len(episode_blacklist)} paths")

        # ========== 优化后的分片逻辑 ==========
        def split_data_dirs(all_data_dirs, rank, world_size,reverse=False):
            """
            优化的目录分片逻辑，适配目录数少于进程数的场景
            """
            # 第一步：处理空列表的极端情况
            if not all_data_dirs:
                return []
            
            all_data_dirs.sort(reverse=reverse)  # 保持原排序逻辑
            total_dirs = len(all_data_dirs)
            print(f"total_dirs={total_dirs}")
            
            # 第二步：判断目录数是否少于进程数
            # 如果目录总数少于总显卡数，让每张卡都加载所有目录
            if total_dirs < world_size:
                data_dirs = all_data_dirs
            else:
                # 场景2：目录数 > 进程数 → 沿用原均匀分片逻辑
                shard_size = total_dirs // world_size
                start_idx = rank * shard_size
                # 最后一个rank处理剩余所有目录
                end_idx = start_idx + shard_size if rank < world_size - 1 else total_dirs
                data_dirs = all_data_dirs[start_idx:end_idx]
            
            return data_dirs
        
        data_dirs = split_data_dirs(all_data_dirs, rank, world_size,reverse = False)
        print(f"rank{rank} data_dirs = {data_dirs}")      

        # 定义过滤函数：只保留需要统计的键，过滤掉多余键
        def filter_unwanted_keys(example):
            # 仅保留第一个数据集的核心键（跳过不需要统计的3个键）
            keep_keys = [
                'action_pose', 'cut_mark', 'is_terminal', 'is_last', 'is_first',
                'observation', 'language_instruction', 'action_joint', 'overlap',
                'traj_metadata', '_len', '_traj_index', '_frame_index',"step_index","episode_length","success_or_failure",
            ]
            # 遍历保留的键，构建新字典（避免键不存在时报错）
            filtered_example = {}
            for k in keep_keys:
                if k in example:
                    filtered_example[k] = example[k]
                else:
                    print(f"{k} not in ")
            
            return filtered_example

        # 原代码逻辑 + 过滤修改
        datasets = []
        for data in data_dirs:
            if not os.path.exists(f"{data}/rlds_example_dataset/1.0.0"):
                print(data, " does not finish convertion")
                continue
            builder = tfds.builder("rlds_example_dataset", data_dir=data, version="1.0.0")
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=num_parallel_reads)
            
            # ========== 关键修改：过滤多余的键 ==========
            # 用 map 操作过滤，开启并行加速
            dataset = dataset.map(
                filter_unwanted_keys,
                num_parallel_calls=tf.data.AUTOTUNE  # 提升处理效率
            )
            
            datasets.append(dataset)

        # 拼接数据集（此时所有数据集结构已统一）
        if datasets:  # 增加非空判断，避免空列表报错
            dataset = datasets[0]
            for i in range(1, len(datasets)):
                dataset = dataset.concatenate(datasets[i])
        else:
            raise ValueError(f"No valid datasets found in {data_dir}!")

        if episode_blacklist:
            file_path_blacklist_regex = tf.constant(_build_file_path_blacklist_regex(episode_blacklist))

            def not_in_episode_blacklist(traj):
                file_path = traj["traj_metadata"]["episode_metadata"]["file_path"]
                file_path = tf.reshape(file_path, [-1])[0]
                return tf.logical_not(tf.strings.regex_full_match(file_path, file_path_blacklist_regex))

            dataset = dataset.filter(not_in_episode_blacklist)

        # # ##################用于从RLDS中筛选原本保存的绝对路径################
        # def string_contains(target, all_patterns):
        #     """
        #     自定义函数：判断patterns中的每个关键词是否在target字符串中
        #     参数：
        #         target: 单个字符串张量
        #         patterns: 多个关键词组成的张量
        #     返回：和patterns长度相同的布尔张量（True=包含，False=不包含）
        #     """
        #     # 构造正则表达式：匹配"包含关键词"的任意字符串（.*表示任意字符）
        #     contains_mask = False
        #     for patterns in all_patterns:
        #         patterns = tf.constant(patterns)
        #         regex_patterns = tf.strings.join([tf.constant(".*"), patterns, tf.constant(".*")])
        #         # 用正则全匹配判断是否包含关键词
        #         # contains_mask = tf.logical_not(tf.strings.regex_full_match(target, regex_patterns))
        #         contains_mask = contains_mask or tf.strings.regex_full_match(target, regex_patterns)
        #     return contains_mask
            
        # 1. 定义要匹配的多个目标子串（可自由添加/删除）
        # 先合并再筛选，适合筛选任务
        
        # TARGET_SUBSTR_LIST = ["猕猴桃", "矿泉水", "香蕉","盲盒","碗","纸巾"]
        # dataset = dataset.filter(
        #     lambda x: (tf.reduce_any(string_contains(
        #                 x["traj_metadata"]["episode_metadata"]["file_path"][0], TARGET_SUBSTR_LIST))
        #     )
        # )
        # ##################用于从RLDS中筛选原本保存的绝对路径################

        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            actions_left = tf.concat([
                traj["observation"]["state_joint"][:,0:(left_size_joint_num-1)],
                traj["action_joint"][:,(left_size_joint_num-1):left_size_joint_num]
                ],axis=1)
            actions_right = tf.concat([
                traj["observation"]["state_joint"][:,left_size_joint_num :(left_size_joint_num + right_size_joint_num -1)],
                traj["action_joint"][:,(left_size_joint_num + right_size_joint_num -1):(left_size_joint_num + right_size_joint_num )]
                ],axis=1)
            # print("traj",traj["observation"]["state_joint"].shape)
            # actions= actions_right

            # gripper_left = tf.cast(traj["action_joint"][:, 6::14] > 0.03, tf.float32)
            # gripper_right = tf.cast(traj["action_joint"][:, 13::14] > 0.03, tf.float32)  
            # zeros_copy = tf.zeros(tf.shape(gripper_left), dtype=tf.float32)    
            # actions = tf.concat((traj["action_joint"], zeros_copy, zeros_copy), axis=-1)        
            # actions = traj["action_pose"]     

            head_img = traj["observation"]["image_head"]
            right_wrist_img = traj["observation"]["right_wrist_image"]
            left_wrist_img = traj["observation"]["left_wrist_image"]
            instruction = traj["language_instruction"]
            if use_left ==True and use_right==True:
                joint_position = traj["observation"]["state_joint"][:,0:(left_size_joint_num + right_size_joint_num )] #双手任务需要去掉[:,8:16]
                actions=tf.concat([actions_left,actions_right],axis=1)
            elif use_left ==True and use_right==False:
                joint_position = traj["observation"]["state_joint"][:,0:left_size_joint_num ] #双手任务需要去掉[:,8:16]
                actions=actions_left
            elif use_left ==False and use_right==True:
                joint_position = traj["observation"]["state_joint"][:,left_size_joint_num:(left_size_joint_num + right_size_joint_num ) ] #双手任务需要去掉[:,8:16]
                actions=actions_right
            # joint_position = traj["observation"]["state_joint"][:,8:16] #双手任务需要去掉[:,8:16]
            # state_left = tf.cast(traj["observation"]["state_joint"][:, 6::14] > 0.03, tf.float32)
            # state_right = tf.cast(traj["observation"]["state_joint"][:, 13::14] > 0.03, tf.float32)
            # joint_position = tf.concat((traj["observation"]["state_joint"][:,:6], state_left, traj["observation"]["state_joint"][:,7:13], state_right), axis=-1)   
            # joint_position = tf.concat((traj["observation"]["state_joint"], zeros_copy, zeros_copy), axis=-1)     
            # joint_position = traj["observation"]["state_pose"]


            # 用某个时间维度推 episode 长度，这里用 joint_position 的长度
            traj_len = tf.shape(joint_position)[0]

            # 如果已有字段，就直接用；没有就按长度生成
            if "step_index" in traj:
                step_index = traj["step_index"]
            else:
                raise RuntimeError("no step index")


            # language_instruction_index, language_instruction_max_len = get_index_and_max_len(instruction)
            language_instruction_index, language_instruction_max_len,language_instruction_at_30precent = get_index_and_max_len(instruction)

            traj["language_instruction_index"] = language_instruction_index
            traj["language_instruction_max_len"] = language_instruction_max_len
            traj["language_instruction_at_30precent"] = language_instruction_at_30precent

            if "episode_length" in traj:
                episode_length = traj["episode_length"]
            else:
                episode_length = tf.fill([traj_len], traj_len)           # [traj_len, traj_len, ...]
                traj["episode_length"] = episode_length

            if "success_or_failure" in traj:
                success_or_failure = traj["success_or_failure"]
            else:
                success_or_failure = tf.ones([traj_len])          # [1, 1, ...]
                traj["success_or_failure"] = success_or_failure

                
            observation = {
                "image": head_img,
                "right_wrist_image": right_wrist_img,
                "left_wrist_image": left_wrist_img,
                "joint_position": joint_position,
            }
            if use_episode_first_head_img:
                observation["episode_first_head_img"] = tf.repeat(
                    traj["observation"]["image_head"][0:1],
                    tf.shape(traj["observation"]["image_head"])[0],
                    axis=0,
                )

            return {
                "actions": actions,
                "observation": observation,
                "prompt": instruction,
                "step_index": step_index,
                "episode_length": episode_length,
                "language_instruction_index": language_instruction_index,
                "language_instruction_max_len": language_instruction_max_len,
                "language_instruction_at_30precent":language_instruction_at_30precent,
                "success_or_failure":success_or_failure,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)


        def chunk_actions(traj):
            """Splits episode into action chunks.
                把一条轨迹（trajectory）按滑动窗口（sliding window）方式切成多个 action chunk。
                若 action_chunk_size = 4

                轨迹长度 traj_len = 7

                那么生成：

                chunk0 = actions[0:4]
                chunk1 = actions[1:5]
                chunk2 = actions[2:6]
                chunk3 = actions[3:7]
            """
            # 读取轨迹长度
            traj_len = tf.shape(traj["actions"])[0]
            # num_chunks = tf.cond(
            #     traj_len >= 2 * action_chunk_size,
            #     lambda: traj_len - 2 * action_chunk_size + 1,
            #     lambda: traj_len
            # )
            # 计算 chunk 个数
            num_chunks = traj_len - action_chunk_size + 1

            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],   # tf.range(action_chunk_size)形状: (action_chunk_size,)  tf.range(action_chunk_size)[None]添加 batch 维度 → (1, action_chunk_size)
                [num_chunks, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(num_chunks)[:, None],        # tf.range(num_chunks)  shape: (num_chunks,)    tf.range(num_chunks)[:, None]  # 添加列维度 → (num_chunks, 1)
                [num_chunks, action_chunk_size],
            )
            # 如果 chunk 超出了轨迹末尾，就重复最后一个动作。
            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            # 按action_chunk_indices索引 gather 出动作 chunk，traj["actions"]的shape = (num_chunks, action_chunk_size, action_dim)

            # traj["observation"]["image_N"] = traj["observation"]["image"][action_chunk_size-1:traj_len]
            # traj["observation"]["right_wrist_image_N"] = traj["observation"]["right_wrist_image"][action_chunk_size-1:traj_len]
            # traj["observation"]["left_wrist_image_N"] = traj["observation"]["left_wrist_image"][action_chunk_size-1:traj_len]
            # traj["observation"]["joint_position_N"] = traj["observation"]["joint_position"][action_chunk_size-1:traj_len]


            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            traj["observation"]["image"] = traj["observation"]["image"][:num_chunks]
            traj["observation"]["right_wrist_image"] = traj["observation"]["right_wrist_image"][:num_chunks]
            traj["observation"]["left_wrist_image"] = traj["observation"]["left_wrist_image"][:num_chunks]
            if use_episode_first_head_img:
                traj["observation"]["episode_first_head_img"] = traj["observation"]["episode_first_head_img"][:num_chunks]
            traj["observation"]["joint_position"] = traj["observation"]["joint_position"][:num_chunks]
            traj["prompt"] = traj["prompt"][:num_chunks]


            traj["step_index"] = traj["step_index"][:num_chunks]
            traj["episode_length"] = traj["episode_length"][:num_chunks]
            traj["language_instruction_index"] = traj["language_instruction_index"][:num_chunks]
            traj["language_instruction_max_len"] = traj["language_instruction_max_len"][:num_chunks]
            traj["language_instruction_at_30precent"] = traj["language_instruction_at_30precent"][:num_chunks]
            traj["success_or_failure"] = traj["success_or_failure"][:num_chunks]
            


            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )

            traj["observation"]["right_wrist_image"] = tf.io.decode_image(
                traj["observation"]["right_wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["left_wrist_image"] = tf.io.decode_image(
                traj["observation"]["left_wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            if use_episode_first_head_img:
                traj["observation"]["episode_first_head_img"] = tf.io.decode_image(
                    traj["observation"]["episode_first_head_img"], expand_animations=False, dtype=tf.uint8
                )

            
            # traj["observation"]["image_N"] = tf.io.decode_image(
            #     traj["observation"]["image_N"], expand_animations=False, dtype=tf.uint8
            # )
            # traj["observation"]["right_wrist_image_N"] = tf.io.decode_image(
            #     traj["observation"]["right_wrist_image_N"], expand_animations=False, dtype=tf.uint8
            # )
            # traj["observation"]["left_wrist_image_N"] = tf.io.decode_image(
            #     traj["observation"]["left_wrist_image_N"], expand_animations=False, dtype=tf.uint8
            # )
            return traj
        dataset = dataset.frame_map(decode_images, num_parallel_calls)


        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)



        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # import pdb; pdb.set_trace()
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 25995127
