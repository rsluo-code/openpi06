"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw_dir /path/to/raw/data --repo_id <org>/<dataset-name>
python examples/aloha_real/convert_process_video.py  --repo_id jfni3/test
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import multiprocessing
from functools import partial
import cv2
import pandas as pd
import sys

import h5py
# from pathlib import Path
# LEROBOT_HOME = Path.home() / ".cache" / "lerobot"
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import itertools


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_yaw",
        "right_wrist_pitch",
        "right_gripper",
    ]
    cameras = [
        "cam_head_images",
        "cam_right_images"
    ]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"{camera}"][:]
        else:
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"{camera}"]:
                imgs_array.append(cv2.imdecode(data, 1))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["robot_state"][:,7:15])
        action = torch.from_numpy(ep["robot_action"][:,7:15])

        # binary gripper
        # action[:,7] = torch.where(action[:,7] >= 0.65, 1.0, -1.0)

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            ["cam_head_images", "cam_right_images",],
            # ["head_camera_images", "left_camera_images", "right_camera_images",],
        )

        prompt = ep['prompt'][()].decode()
    return imgs_per_cam, state, action, velocity, effort, prompt


def process_episode(episode_dir, failed_log):
    """处理单个episode文件夹"""
    # 定义数据文件路径
    head_video_path = episode_dir / "cam_head/cam_head.mp4"
    right_video_path = episode_dir / "cam_right/cam_right.mp4"
    csv_path = episode_dir / "robot_data.csv"
    
    # 加载动作和状态数据
    df = pd.read_csv(
        csv_path,
        header=0,  # 使用第一行作为列名
        index_col=False,  # 禁止将首列作为索引
        usecols=lambda col: not col.startswith('Unnamed')  # 过滤自动生成的索引列
    )
    actions = df[[f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
    states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)

    
    # 加载视频帧
    right_cap = cv2.VideoCapture(str(right_video_path))
    right_frames = []
    frame_count = 0
    while right_cap.isOpened():
        ret, frame = right_cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        right_frames.append(rgb_frame)
        frame_count += 1
    right_cap.release()

    head_cap = cv2.VideoCapture(str(head_video_path))
    head_frames = []
    frame_count = 0
    while head_cap.isOpened():
        ret, frame = head_cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head_frames.append(rgb_frame)
        frame_count += 1
    head_cap.release()

    # print(f"\nProcessing {episode_dir.name}:")
    # print(f"CSV数据帧数: {len(states)} (动作: {len(actions)})")
    # print(f"右摄像头视频帧数: {len(right_frames)}")
    # print(f"头摄像头视频帧数: {len(head_frames)}")

    if len(states) != len(right_frames) or len(states) != len(head_frames):
        print(f"❌ 数据长度不匹配，跳过episode: {episode_dir.name}")
        print(f"  状态帧数: {len(states)}, 右摄像头: {len(right_frames)}, 头摄像头: {len(head_frames)}")
        with open(failed_log, 'a') as f:
            f.write(f"{episode_dir.name}\n")
        return [], None
    
    stop_reason = None
    info_path = episode_dir / "info.txt"
    with open(info_path, 'r') as f:
        for line in f:
            if line.startswith("StopReason"):
                stop_reason = line.split('=')[1].strip()
                break
    if stop_reason != "USER_STOP":
        print(f"❌ StopReason={stop_reason}不符合要求，跳过episode: {episode_dir.name}")
        with open(failed_log, 'a') as f:
            f.write(f"{episode_dir.name} (StopReason={stop_reason})\n")
        return [], None


    # 构建数据帧
    frames_list = []
    for i in range(len(states)):
        frame_data = {
            "observation.state": torch.from_numpy(states[i]),
            "action": torch.from_numpy(actions[i]),
            "observation.images.cam_right_images": right_frames[i],
            "observation.images.cam_head_images": head_frames[i],
        }
        frames_list.append(frame_data)
    
    return frames_list, "pick apple in pallet"

def populate_dataset(
    dataset: LeRobotDataset,
    episode_dirs: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(episode_dirs))

    failed_log = "failed_episodes.log"
    if Path(failed_log).exists():
        Path(failed_log).unlink()

    # 改为顺序处理每个episode
    for episode_dir in episode_dirs:
        frames, prompt = process_episode(episode_dir, failed_log)
        if not frames:
            continue
        # 写入数据
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode(task=prompt)

    return dataset


def port_aloha(
    # raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)


    raw_dirs=["/b3-mix03/sppro/permanent/jfni3/apple0815-1200pcs/0814—apple/bg_tablecloth1:arm_right:object_apple:base_0.200_15.000/"]
    episode_dirs = []
    for raw_dir in raw_dirs:
        raw_dir_path = Path(raw_dir)
        episode_dirs.extend([d for d in raw_dir_path.rglob("episode_2025-08-15_10:0*") if d.is_dir()])

    episode_dirs=episode_dirs[:]
    size = len(episode_dirs)
    print("Size of episode_dirs:", size)

    dataset_path = Path(repo_id)
    
    # if dataset_path.exists():
    #     # 加载现有数据集时明确指定参数
    #     dataset = LeRobotDataset(
    #         repo_id="my_data", 
    #         root=dataset_path,
    #         local_files_only=True
    #     )
    #     print(dataset)
    #     print(f"Loaded existing dataset at {dataset_path}")
    # else:
        # 创建新数据集
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    print(f"Created new dataset at {LEROBOT_HOME/repo_id}")

    # 后续处理保持不变
    dataset = populate_dataset(
        dataset,
        episode_dirs,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(port_aloha)