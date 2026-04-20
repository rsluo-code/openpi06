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



"""Rest everything else."""
episode_dir = "/DATA/disk1/1024_grasp_packages/data/origin_data_v2/data_10_24/data_10_24_r1/1024_15_快递_LD_247/bg_tablecloth1arm_allobject_1/episode_2025-10-24_024207_528"
@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10

    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    # video_out_path: str = "data/isaac/videos"  # Path to save videos

def eval_isaac(args: Args) -> None:

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Initialize LIBERO environment and task description
    task_description = _get_isaac_env()

    # Start episodes
    task_episodes, task_successes = 0, 0
    action_success = 0
    done_flag = False
    head_video_path = episode_dir + "/cam_head/cam_head.mp4"
    left_video_path = episode_dir + "/cam_left/cam_left.mp4"
    right_video_path = episode_dir + "/cam_right/cam_right.mp4"
    csv_path = episode_dir + "/robot_data.csv"
    
    # 加载动作和状态数据
    df = pd.read_csv(
        csv_path,
        header=0,  # 使用第一行作为列名
        index_col=False,  # 禁止将首列作为索引
        usecols=lambda col: not col.startswith('Unnamed')  # 过滤自动生成的索引列
    )

    actions = df[[f"l_a_{i}" for i in range(8)] + [f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
    states = df[[f"l_s_{i}" for i in range(8)] + [f"r_s_{i}" for i in range(8)]].values.astype(np.float32)
    
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

    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        print(f"\nTask: {task_description}")

        # Reset environment
        action_plan = collections.deque()

        # Setup
        t = 0
        err_list = []
        joint1 = []
        joint2 = []
        joint3 = []
        joint4 = []
        joint5 = []
        joint6 = []
        joint7 = []
        joint8 = []
        joint9 = []
        joint10 = []
        joint11 = []
        joint12 = []
        joint13 = []
        joint14 = []
        joint15 = []
        joint16 = []


        label1 = []
        label2 = []
        label3 = []
        label4 = []
        label5 = []
        label6 = []
        label7 = []
        label8 = []
        label9 = []
        label10 = []
        label11 = []
        label12 = []
        label13 = []
        label14 = []
        label15 = []
        label16 = []

        print(f"Starting episode {task_episodes+1}...")
        for i in range(states.shape[0]):
            try:
                img = head_frames[i]
                right_wrist_img = right_frames[i]
                left_wrist_img = left_frames[i]

                # image = obs['image_recording']["image_head"].cpu().numpy().squeeze()
                # wrist_image = obs['image_recording']["image_arm"].cpu().numpy().squeeze()

                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # left_wrist_image = cv2.cvtColor(left_wrist_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("1.png", image)

                # img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                # )
                # left_wrist_img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(left_wrist_img, args.resize_size, args.resize_size)
                # )
                # right_wrist_img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(right_wrist_img, args.resize_size, args.resize_size)
                # )

                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # left_wrist_img = cv2.cvtColor(left_wrist_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("head.png", img)
                # cv2.imwrite("left.png", left_wrist_img)

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
                action_error = np.mean(np.abs(action[:16] - actions[i][:16]))
                err_list.append(action_error)
                joint1.append(action[0])
                joint2.append(action[1])
                joint3.append(action[2])
                joint4.append(action[3])
                joint5.append(action[4])
                joint6.append(action[5])
                joint7.append(action[6])
                joint8.append(action[7])
                joint9.append(action[8])
                joint10.append(action[9])
                joint11.append(action[10])
                joint12.append(action[11])
                joint13.append(action[12])
                joint14.append(action[13])
                joint15.append(action[14])
                joint16.append(action[15])

                label1.append(actions[i][0])
                label2.append(actions[i][1])
                label3.append(actions[i][2])
                label4.append(actions[i][3])
                label5.append(actions[i][4])
                label6.append(actions[i][5])
                label7.append(actions[i][6])
                label8.append(actions[i][7])
                label9.append(actions[i][8])
                label10.append(actions[i][9])
                label11.append(actions[i][10])
                label12.append(actions[i][11])
                label13.append(actions[i][12])
                label14.append(actions[i][13])
                label15.append(actions[i][14])
                label16.append(actions[i][15])





            except Exception as e:
                print(f"Caught exception: {e}")
                break
        print(np.mean(err_list))
        task_episodes += 1
        total_episodes += 1

        all_joints = [joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8, joint9, joint10, joint11, joint12, joint13, joint14, joint15, joint16]
        all_labels = [label1, label2, label3, label4, label5, label6, label7, label8, label9, label10, label11, label12, label13, label14,label15, label16]
        timesteps = np.arange(len(joint1)) 

        fig, axes = plt.subplots(4, 4, figsize=(12, 16))
        fig.suptitle('Joint Angles', fontsize=16, y=1.02)

        for i, (ax, joint_data, label_data) in enumerate(zip(axes.flatten(), all_joints, all_labels)):
            ax.plot(timesteps, joint_data,
                linewidth=0.2,
                marker='o',
                markersize=1,
                label=f'Joint {i+1}')
            
            ax.plot(timesteps, label_data,
                color='red',
                linewidth=0.2,
                marker='o',
                markersize=1,
                label=f'Label {i+1}')

            ax.set_title(f'Joint {i+1}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=8)
            ax.set_ylabel('Angle (rad)', fontsize=8)
            ax.legend(fontsize=8)

        # axes.flatten()[16].set_visible(False)
        plt.tight_layout()
        plt.savefig('/DATA/disk1/1024_grasp_packages/code/pi0_code/val_results/episode_2025-10-24_024207_528.png')
        # plt.show()



def _get_isaac_env():
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = "If the label is face-up, flip the package and put it in the basket"

    return task_description


if __name__ == "__main__":
    tyro.cli(eval_isaac)
