import collections
import dataclasses
import math
import pathlib

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import argparse
import torch
import cv2
import einops
import time
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import glob



"""Rest everything else."""

ISAAC_DUMMY_ACTION = [0.0] * 7 + [1.0]

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10


def eval_isaac(args: Args) -> None:

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    data_dir = "/b3-mix03/sppro/permanent/yuanzhang10/资源部传输数据/叠衣服/SL-叠衣服-全检结束-铲夹起/SL-短袖1001_全检结束/1001_短袖_04_短袖_SL_1件"
    episode_dirs = glob.glob(f"{data_dir}/episode_2025100109*")
    total_episodes, total_successes = 0, 0
    all_err = []
    for episode_dir in episode_dirs:
        # Initialize LIBERO environment and task description
        task_description = _get_isaac_env()

        # Start episodes
        task_episodes, task_successes = 0, 0
        action_success = 0
        done_flag = False
        head_video_path = episode_dir + "/cam_head/cam_head.mp4"
        left_video_path = episode_dir + "/cam_left/cam_left.mp4"
        right_video_path = episode_dir + "/cam_right/cam_right.mp4"
        csv_path = episode_dir + "/robot.csv"
        
        # 加载动作和状态数据
        df = pd.read_csv(
            csv_path,
            header=0,  # 使用第一行作为列名
            index_col=False,  # 禁止将首列作为索引
            usecols=lambda col: not col.startswith('Unnamed')  # 过滤自动生成的索引列
        )

        actions = df[[f"l_a_{i}" for i in range(7)] + [f"r_a_{i}" for i in range(7)]].values.astype(np.float32)
        states = df[[f"l_s_{i}" for i in range(7)] + [f"r_s_{i}" for i in range(7)]].values.astype(np.float32)
        
        # 加载视频帧
        left_cap = cv2.VideoCapture(str(left_video_path))
        left_frames = []
        frame_count = 0
        while left_cap.isOpened():
            ret, frame = left_cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left_frames.append(rgb_frame)
            frame_count += 1
        left_cap.release()

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

        print(f"\nTask: {task_description}")

        # Reset environment
        action_plan = collections.deque()

        # Setup
        t = 0
        err_list = []

        print(f"Starting episode {task_episodes+1}...")
        for i in range(states.shape[0]):
            try:
                img = head_frames[i]
                right_wrist_img = right_frames[i]
                left_wrist_img = left_frames[i]

                img = einops.rearrange(img, "h w c -> c h w")
                right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
                left_wrist_img = einops.rearrange(left_wrist_img, "h w c -> c h w")

                # Save preprocessed image for replay video
                # replay_images.append(img)
                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "observation/image": img,
                        "observation/wrist_image_right": right_wrist_img,
                        "observation/wrist_image_left":left_wrist_img,
                        "observation/joint_position": states[i],
                        "prompt": str(task_description),
                    }
                    # Query model to get action
                    # start=time.time()
                    action_chunk = client.infer(element)["actions"]
                    # end=time.time()
                    # print("time:", end-start, action_chunk.shape)
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[:args.replan_steps])
                    
                # if len(action_plan) == args.replan_steps:
                #     print(action_plan)
                action = action_plan.popleft().tolist()
                action_error = np.mean(np.abs(action[:14] - actions[i][:14]))
                err_list.append(action_error)

            except Exception as e:
                print(f"Caught exception: {e}")
                break
        all_err.append(np.mean(err_list))
        print(np.mean(err_list))
        task_episodes += 1
        total_episodes += 1
    print("mean:", np.mean(all_err))




def _get_isaac_env():
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = "fold the T-shirt neatly"

    return task_description


if __name__ == "__main__":
    tyro.cli(eval_isaac)
