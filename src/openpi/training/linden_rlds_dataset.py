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

class LindenRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 30,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 60000, #250_000,
        num_parallel_reads: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        is_joint = False
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # builder = tfds.builder("rlds_example_dataset", data_dir=data_dir, version="1.0.0")
        # dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        raw_dir_path = Path(data_dir)
        all_data_dirs = []
        all_data_dirs.extend([d for d in raw_dir_path.rglob("general_pick_place*") if d.is_dir()])
        
        # all_data_dirs = glob.glob(f"{data_dir}/folding_1004*") + glob.glob(f"{data_dir}/folding_1005*")
        all_data_dirs.sort(reverse=True)  
        shard_size = len(all_data_dirs) // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else len(all_data_dirs)
        data_dirs = all_data_dirs[start_idx:end_idx]

        # data_dirs = glob.glob(f"{data_dir}/folding_09*")
        datasets = []
        for data in data_dirs:
            if not os.path.exists(f"{data}/rlds_example_dataset/1.0.0"):
                print(data, " does not finish convertion")
                continue
            builder = tfds.builder("rlds_example_dataset", data_dir=data, version="1.0.0")
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=num_parallel_reads)
            datasets.append(dataset)
        dataset = datasets[0]
        for i in range(1, len(datasets)):
            dataset = dataset.concatenate(datasets[i])

        def string_contains(target, all_patterns):
            """
            自定义函数：判断patterns中的每个关键词是否在target字符串中
            参数：
                target: 单个字符串张量
                patterns: 多个关键词组成的张量
            返回：和patterns长度相同的布尔张量（True=包含，False=不包含）
            """
            # 构造正则表达式：匹配"包含关键词"的任意字符串（.*表示任意字符）
            contains_mask = False
            for patterns in all_patterns:
                patterns = tf.constant(patterns)
                regex_patterns = tf.strings.join([tf.constant(".*"), patterns, tf.constant(".*")])
                # 用正则全匹配判断是否包含关键词
                # contains_mask = tf.logical_not(tf.strings.regex_full_match(target, regex_patterns))
                contains_mask = contains_mask or tf.strings.regex_full_match(target, regex_patterns)
            return contains_mask
            
        # 1. 定义要匹配的多个目标子串（可自由添加/删除）
        TARGET_SUBSTR_LIST = ["猕猴桃", "矿泉水", "香蕉","盲盒","碗","纸巾"]

        # filter truncation episode
        dataset = dataset.filter(
            lambda x: (tf.reduce_any(string_contains(
                        x["traj_metadata"]["episode_metadata"]["file_path"][0], TARGET_SUBSTR_LIST))
            )
        )
        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # actions = traj["action_pose"][:,0:8]
            # actions = traj["observation"]["state_pose"][:,0:8]
            ##----------------------------------------------------归一化注释----------------------------------------------------
            head_img = traj["observation"]["image_head"]
            right_wrist_img = traj["observation"]["right_wrist_image"]
            left_wrist_img = traj["observation"]["left_wrist_image"]
            ##----------------------------------------------------归一化注释----------------------------------------------------
            
            instruction = traj["language_instruction"] 
            # joint_position = traj["observation"]["state_pose"][:,0:8]
            # joint_position = traj["observation"]["state_pose"][:,0:8]

            if is_joint:
                actions = tf.concat([traj["observation"]["state_joint"][:,0:7], traj["action_joint"][:,7:8]], axis=1)
                joint_position = traj["observation"]["state_joint"][:,0:8]
            else:
                actions = tf.concat([traj["observation"]["state_pose"][:,0:7], traj["action_pose"][:,7:8]], axis=1)
                joint_position = traj["observation"]["state_pose"][:,0:8]
            # print(actions)
            # import pdb;pdb.set_trace()

            return {
                "actions": actions,
                "observation": {
             ##----------------------------------------------------归一化注释----------------------------------------------------
                    "image": head_img,
                    "right_wrist_image": right_wrist_img,
                    "left_wrist_image": left_wrist_img,
             ##----------------------------------------------------归一化注释----------------------------------------------------
                    "joint_position": joint_position,
                },
                "prompt": instruction,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]
            # num_chunks = tf.cond(
            #     traj_len >= 2 * action_chunk_size,
            #     lambda: traj_len - 2 * action_chunk_size + 1,
            #     lambda: traj_len
            # )
            num_chunks = traj_len - action_chunk_size + 1
            # if (tf.reduce_max(traj["actions"])>3) or (tf.reduce_min(traj["actions"])<-3):  # exclude wrong values
            #     num_chunks = 0
            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [num_chunks, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(num_chunks)[:, None],
                [num_chunks, action_chunk_size],
            )

            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            ##----------------------------------------------------归一化注释----------------------------------------------------
            traj["observation"]["image"] = traj["observation"]["image"][:num_chunks]
            traj["observation"]["right_wrist_image"] = traj["observation"]["right_wrist_image"][:num_chunks]
            traj["observation"]["left_wrist_image"] = traj["observation"]["left_wrist_image"][:num_chunks]
            ##----------------------------------------------------归一化注释----------------------------------------------------
            traj["observation"]["joint_position"] = traj["observation"]["joint_position"][:num_chunks]
            traj["prompt"] = traj["prompt"][:num_chunks]
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            ##----------------------------------------------------归一化注释----------------------------------------------------
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["right_wrist_image"] = tf.io.decode_image(
                traj["observation"]["right_wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["left_wrist_image"] = tf.io.decode_image(
                traj["observation"]["left_wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            ##----------------------------------------------------归一化注释----------------------------------------------------
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
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it. #22786722
        return 24100000


class LindenRewardRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 30,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 60000, #250_000,
        num_parallel_reads: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        is_joint = False
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # builder = tfds.builder("rlds_example_dataset", data_dir=data_dir, version="1.0.0")
        # dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        raw_dir_path = Path(data_dir)
        all_data_dirs = []
        all_data_dirs.extend([d for d in raw_dir_path.rglob("general_pick_place*") if d.is_dir()])
        # import pdb; pdb.set_trace()
        
        # all_data_dirs = glob.glob(f"{data_dir}/folding_1004*") + glob.glob(f"{data_dir}/folding_1005*")
        all_data_dirs.sort(reverse=True)
        shard_size = len(all_data_dirs) // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else len(all_data_dirs)
        data_dirs = all_data_dirs[start_idx:end_idx]
        shard_remainder = len(all_data_dirs) % world_size
        if rank < shard_remainder:
            data_dirs.append(all_data_dirs[shard_size*world_size+rank])

        import random
        random.shuffle(data_dirs)

        # data_dirs = glob.glob(f"{data_dir}/folding_09*")
        datasets = []
        for data in data_dirs:
            if not os.path.exists(f"{data}/rlds_example_dataset/1.0.0"):
                print(data, " does not finish convertion")
                continue
            builder = tfds.builder("rlds_example_dataset", data_dir=data, version="1.0.0")
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=num_parallel_reads)
            datasets.append(dataset)
        dataset = datasets[0]
        for i in range(1, len(datasets)):
            dataset = dataset.concatenate(datasets[i])

        def string_contains(target, all_patterns):
            """
            自定义函数：判断patterns中的每个关键词是否在target字符串中
            参数：
                target: 单个字符串张量
                patterns: 多个关键词组成的张量
            返回：和patterns长度相同的布尔张量（True=包含，False=不包含）
            """
            # 构造正则表达式：匹配"包含关键词"的任意字符串（.*表示任意字符）
            contains_mask = False
            for patterns in all_patterns:
                patterns = tf.constant(patterns)
                regex_patterns = tf.strings.join([tf.constant(".*"), patterns, tf.constant(".*")])
                # 用正则全匹配判断是否包含关键词
                # contains_mask = tf.logical_not(tf.strings.regex_full_match(target, regex_patterns))
                contains_mask = contains_mask or tf.strings.regex_full_match(target, regex_patterns)
            return contains_mask
            
        # 1. 定义要匹配的多个目标子串（可自由添加/删除）
        TARGET_SUBSTR_LIST = ["猕猴桃", "矿泉水", "香蕉","盲盒","碗","纸巾"]

        # filter truncation episode
        dataset = dataset.filter(
            lambda x: (tf.reduce_any(string_contains(
                        x["traj_metadata"]["episode_metadata"]["file_path"][0], TARGET_SUBSTR_LIST))
            )
        )
        print(TARGET_SUBSTR_LIST)
        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            # traj_len = tf.shape(traj["action_joint"])[0]
            # random_drop_num_frames = tf.cast(30 * (3 + tf.random.uniform([]) * 0.25), dtype=tf.int32)
            # valid_indices = tf.range(random_drop_num_frames, limit=traj_len - random_drop_num_frames, dtype=tf.int32)
            # actions = traj["action_pose"][:,0:8]
            # actions = traj["observation"]["state_pose"][:,0:8]
            ##----------------------------------------------------归一化注释----------------------------------------------------
            head_img = traj["observation"]["image_head"]
            right_wrist_img = traj["observation"]["right_wrist_image"]
            left_wrist_img = traj["observation"]["left_wrist_image"]
            ##----------------------------------------------------归一化注释----------------------------------------------------
            # import pdb; pdb.set_trace()
            instruction = traj["language_instruction"] 
            # joint_position = traj["observation"]["state_pose"][:,0:8]
            # joint_position = traj["observation"]["state_pose"][:,0:8]

            if is_joint:
                actions = tf.concat([traj["observation"]["state_joint"][:,0:7], traj["action_joint"][:,7:8]], axis=1)
                joint_position = traj["observation"]["state_joint"][:,0:8]

                # zeros_pad = traj["action_joint"][:,6:7] * 0.
                # actions = tf.concat([traj["observation"]["state_joint"][:,0:6], zeros_pad, traj["action_joint"][:,6:7]], axis=1)
                # joint_position = tf.concat([traj["observation"]["state_joint"][:,0:6], zeros_pad, traj["observation"]["state_joint"][:,6:7]], axis=1)
            else:
                actions = tf.concat([traj["observation"]["state_pose"][:,0:7], traj["action_pose"][:,7:8]], axis=1)
                joint_position = traj["observation"]["state_pose"][:,0:8]

            # 用某个时间维度推 episode 长度，这里用 joint_position 的长度
            traj_len = tf.shape(joint_position)[0]

            # 如果已有字段，就直接用；没有就按长度生成
            if "step_index" in traj:
                step_index = traj["step_index"]
            else:
                step_index = tf.range(traj_len, dtype=tf.int32)          # 0,1,...,traj_len-1
                traj["step_index"] = step_index
            if "episode_length" in traj:
                episode_length = traj["episode_length"]
            else:
                episode_length = tf.fill([traj_len], traj_len)           # [traj_len, traj_len, ...]
                traj["episode_length"] = episode_length
            # import pdb; pdb.set_trace()

            return {
                # "actions": tf.gather(actions, valid_indices),
                # "observation": {
                # ##----------------------------------------------------归一化注释----------------------------------------------------
                #     "image": tf.gather(head_img, valid_indices),
                #     "right_wrist_image": tf.gather(right_wrist_img, valid_indices),
                #     "left_wrist_image": tf.gather(left_wrist_img, valid_indices),
                # ##----------------------------------------------------归一化注释----------------------------------------------------
                #     "joint_position": tf.gather(joint_position, valid_indices),
                # },
                # "prompt": tf.gather(instruction, valid_indices),               
                "actions": actions,
                "observation": {
                ##----------------------------------------------------归一化注释----------------------------------------------------
                    "image": head_img,
                    "right_wrist_image": right_wrist_img,
                    "left_wrist_image": left_wrist_img,
                ##----------------------------------------------------归一化注释----------------------------------------------------
                    "joint_position": joint_position,
                },
                "prompt": instruction,
                "step_index": step_index,
                "episode_length": episode_length,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks."""
            traj_len = tf.shape(traj["actions"])[0]
            # num_chunks = tf.cond(
            #     traj_len >= 2 * action_chunk_size,
            #     lambda: traj_len - 2 * action_chunk_size + 1,
            #     lambda: traj_len
            # )
            num_chunks = traj_len - action_chunk_size + 1
            # if (tf.reduce_max(traj["actions"])>3) or (tf.reduce_min(traj["actions"])<-3):  # exclude wrong values
            #     num_chunks = 0
            # For each step in the trajectory, construct indices for the next n actions
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [num_chunks, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(num_chunks)[:, None],
                [num_chunks, action_chunk_size],
            )
            # import pdb; pdb.set_trace()
            # Cap to length of the sequence --> final chunks will repeat the last action
            # This makes sense, since we are using absolute joint + gripper position actions
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # Gather the actions for each chunk
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            ##----------------------------------------------------归一化注释----------------------------------------------------
            traj["observation"]["image"] = traj["observation"]["image"][:num_chunks]
            traj["observation"]["right_wrist_image"] = traj["observation"]["right_wrist_image"][:num_chunks]
            traj["observation"]["left_wrist_image"] = traj["observation"]["left_wrist_image"][:num_chunks]
            ##----------------------------------------------------归一化注释----------------------------------------------------
            traj["observation"]["joint_position"] = traj["observation"]["joint_position"][:num_chunks]
            traj["prompt"] = traj["prompt"][:num_chunks]

            # Only RL
            episode_len = tf.shape(traj["actions"])[0]
            # task_progress = tf.expand_dims(tf.range(episode_len - 1, -1, delta=-1), axis=1)
            # rewards = -tf.ones((episode_len, 1), dtype=tf.float32)
            # returns = -tf.cast(task_progress, dtype=tf.float32)
            # import pdb; pdb.set_trace()
            traj["step_index"] = traj["step_index"][:num_chunks]
            traj["episode_length"] = traj["episode_length"][:num_chunks]
            # traj["observation"]["start_image"] = traj["episode_length"][:num_chunks]

            # traj["task_progress"] = task_progress
            # traj["rewards"] = rewards
            # traj["returns"] = returns

            # traj["observation"]["start_image"] = tf.repeat(traj["observation"]["image"][0:1], repeats=episode_len, axis=0)
            
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # Decode images: RLDS saves encoded images, only decode now for efficiency
        def decode_images(traj):
            ##----------------------------------------------------归一化注释----------------------------------------------------
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["right_wrist_image"] = tf.io.decode_image(
                traj["observation"]["right_wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["left_wrist_image"] = tf.io.decode_image(
                traj["observation"]["left_wrist_image"], expand_animations=False, dtype=tf.uint8
            )

            # traj["observation"]["start_image"] = tf.io.decode_image(
            #     traj["observation"]["start_image"], expand_animations=False, dtype=tf.uint8
            # )
            ##----------------------------------------------------归一化注释----------------------------------------------------
            return traj
        dataset = dataset.frame_map(decode_images, num_parallel_calls)


        # Shuffle, batch
        dataset = dataset.shuffle(shuffle_buffer_size)#(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it. #22786722
        return 24100000
