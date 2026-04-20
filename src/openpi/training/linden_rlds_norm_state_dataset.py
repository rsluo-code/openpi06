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



class LindenRldNormStateDataset:
    all_rlds_data_len = 0
    # all_rlds_data_len = 20208055
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 30,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 5000, #250_000,
        num_parallel_reads: int = 8,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = 8  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        print("="*25,"rsluo_test LindenRldNormStateDataset","="*25)
        print("dlimp at:", dl.__file__)
        print("Has DLataset?", hasattr(dl, "DLataset"))
        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # builder = tfds.builder("rlds_example_dataset", data_dir=data_dir, version="1.0.0")
        # dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # all_data_dirs = glob.glob(f"{data_dir}/data_03_12/package_sorting_*")
        all_data_dirs = glob.glob(f"{data_dir}/*/package_sorting_*")
        
        # all_data_dirs = glob.glob(f"{data_dir}/folding_1004*") + glob.glob(f"{data_dir}/folding_1005*")
        all_data_dirs.sort(reverse=True)  
        shard_size = len(all_data_dirs) // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else len(all_data_dirs)
        data_dirs = all_data_dirs[start_idx:end_idx]

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


        # data_dirs = glob.glob(f"{data_dir}/folding_09*")
        datasets = []
        if_compute_all_rlds_data_len =False
        traj_lengths = []  # 用来保存每条 trajectory 的长度
        if self.all_rlds_data_len != 0 :
            if_compute_all_rlds_data_len=False
        for data in data_dirs:
            if not os.path.exists(f"{data}/rlds_example_dataset/1.0.0"):
                print(data, " does not finish convertion")
                continue
            print(data, " start convertion")
            
            builder = tfds.builder("rlds_example_dataset", data_dir=data, version="1.0.0")
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=num_parallel_reads)
            if if_compute_all_rlds_data_len:
                for idx, item in enumerate(dataset.as_numpy_iterator()):

                    traj_length = item["observation"]["state_joint"].shape[0]  # 第一维长度
                    traj_lengths.append(traj_length)
                    self.all_rlds_data_len += traj_length

            # ========== 关键修改：过滤多余的键 ==========
            # 用 map 操作过滤，开启并行加速
            dataset = dataset.map(
                filter_unwanted_keys,
                num_parallel_calls=tf.data.AUTOTUNE  # 提升处理效率
            )
            datasets.append(dataset)
        print(f"cal steps len finish, self.all_rlds_data_len = {self.all_rlds_data_len}")
        if(self.all_rlds_data_len == 0):
            raise RuntimeError(f"do not found rlds_example_dataset in {data_dir}")

        # 打印统计信息
        if if_compute_all_rlds_data_len:
            print("="*30)
            print(f"Total all_rlds_data_len = {self.all_rlds_data_len}")
            if traj_lengths:
                print(f"Number of trajectories: {len(traj_lengths)}")
                print(f"Trajectory length - min: {min(traj_lengths)}, max: {max(traj_lengths)}, mean: {sum(traj_lengths)/len(traj_lengths):.2f}")
            print("="*30)

        # dataset = datasets[0]
        for i in range(1, len(datasets)):
            dataset = dataset.concatenate(datasets[i])

        # # Repeat dataset so we never run out of data.
        dataset = dataset.repeat()

        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            actions_left = tf.concat([traj["observation"]["state_joint"][:,0:7],traj["action_joint"][:,7:8]],axis=1)
            actions_right = tf.concat([traj["observation"]["state_joint"][:,8:15],traj["action_joint"][:,15:16]],axis=1)
            # print("traj",traj["observation"]["state_joint"].shape)
            actions=tf.concat([actions_left,actions_right],axis=1)

            instruction = traj["language_instruction"]

            joint_position = traj["observation"]["state_joint"][:,0:16] #双手任务需要去掉[:,8:16]
            # print(f"joint_position.shape={joint_position.shape} ")


            return {
                "actions": actions,
                "observation": {
                    "joint_position": joint_position,
                },
                "prompt": instruction,


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
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            traj["observation"]["joint_position"] = traj["observation"]["joint_position"][:num_chunks]
            traj["prompt"] = traj["prompt"][:num_chunks]
  

            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)


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
        # Easier to hardcode than to iterate through the dataset and compute it.
        print(f"using dataset __len__={self.all_rlds_data_len}")
        return self.all_rlds_data_len
        return 25995127
