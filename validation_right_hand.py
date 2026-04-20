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
episode_dir = "/DATA/disk2/1024_grasp_packages/data/SF_origin_data/data_11_03/1103_03_快递_LD_140/bg_tablecloth1arm_leftobject_1/episode_2025-11-03_095210_708"
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
    right_video_path = episode_dir + "/cam_right/cam_right.mp4"
    csv_path = episode_dir + "/robot_data.csv"
    
    # 加载动作和状态数据 - 只使用右手数据
    df = pd.read_csv(
        csv_path,
        header=0,  # 使用第一行作为列名
        index_col=False,  # 禁止将首列作为索引
        usecols=lambda col: not col.startswith('Unnamed')  # 过滤自动生成的索引列
    )

    # 只使用右手的动作和状态 (r_a_0 到 r_a_7, r_s_0 到 r_s_7)
    actions = df[[f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
    states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)
    
    # 加载视频帧 - 只加载头部和右手图像
    head_cap = cv2.VideoCapture(str(head_video_path))
    head_frames = []
    while head_cap.isOpened():
        ret, frame = head_cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head_frames.append(rgb_frame)
    head_cap.release()

    right_cap = cv2.VideoCapture(str(right_video_path))
    right_frames = []
    while right_cap.isOpened():
        ret, frame = right_cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        right_frames.append(rgb_frame)
    right_cap.release()

    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        print(f"\nTask: {task_description}")

        # Reset environment
        action_plan = collections.deque()

        # Setup
        t = 0
        err_list = []
        # 只记录8个右手关节
        joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8 = [], [], [], [], [], [], [], []
        label1, label2, label3, label4, label5, label6, label7, label8 = [], [], [], [], [], [], [], []

        print(f"Starting episode {task_episodes+1}...")
        for i in range(min(states.shape[0], len(head_frames), len(right_frames))):
            try:
                img = head_frames[i]
                right_wrist_img = right_frames[i]

                # 预处理图像
                img = einops.rearrange(img, "h w c -> c h w")
                right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
                
                # 创建左手图像的零占位符
                left_wrist_img = np.zeros_like(right_wrist_img)

                if not action_plan:
                    # 准备观测数据 - 包含头部图像、右手图像、左手图像零占位符和右手状态
                    element = {
                        "observation/image": img,
                        "observation/wrist_image_right": right_wrist_img,
                        "observation/wrist_image_left": left_wrist_img,  # 使用零占位符
                        "observation/joint_position": states[i],  # 8维右手状态
                        "prompt": str(task_description),
                    }
                    # 查询模型获取动作
                    action_chunk = client.infer(element)["actions"]
                    
                    # 检查动作维度 - 现在应该是8维右手动作
                    # print(f"Action chunk shape: {action_chunk.shape}")
                    
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[:args.replan_steps])
                    
                # 获取动作 (应该是8维右手动作)
                action = action_plan.popleft().tolist()
                
                # 检查动作维度
                if len(action) != 8:
                    print(f"Warning: Expected 8-dimensional action, got {len(action)}-dimensional")
                    # 如果服务端返回的不是8维，我们只取前8维
                    action = action[:8]
                
                # 计算动作误差 - 比较8维右手动作
                action_error = np.mean(np.abs(action - actions[i]))
                err_list.append(action_error)
                
                # 记录预测的关节角度
                joint1.append(action[0])
                joint2.append(action[1])
                joint3.append(action[2])
                joint4.append(action[3])
                joint5.append(action[4])
                joint6.append(action[5])
                joint7.append(action[6])
                joint8.append(action[7])

                # 记录真实的关节角度
                label1.append(actions[i][0])
                label2.append(actions[i][1])
                label3.append(actions[i][2])
                label4.append(actions[i][3])
                label5.append(actions[i][4])
                label6.append(actions[i][5])
                label7.append(actions[i][6])
                label8.append(actions[i][7])

            except Exception as e:
                print(f"Caught exception: {e}")
                break
        
        if err_list:  # 确保err_list不为空
            print(f"Average action error: {np.mean(err_list)}")
        else:
            print("No actions were processed")
            
        task_episodes += 1
        total_episodes += 1

        # 准备绘图数据
        all_joints = [joint1, joint2, joint3, joint4, joint5, joint6, joint7, joint8]
        all_labels = [label1, label2, label3, label4, label5, label6, label7, label8]
        
        # 确保有数据可以绘图
        if joint1:
            timesteps = np.arange(len(joint1)) 

            # 创建图表 - 2x4布局显示8个右手关节
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle('Right Hand Joint Angles', fontsize=16, y=1.02)

            for i, (ax, joint_data, label_data) in enumerate(zip(axes.flatten(), all_joints, all_labels)):
                ax.plot(timesteps, joint_data,
                    linewidth=0.2,
                    marker='o',
                    markersize=1,
                    label=f'Predicted Joint {i+1}')
                
                ax.plot(timesteps, label_data,
                    color='red',
                    linewidth=0.2,
                    marker='o',
                    markersize=1,
                    label=f'Ground Truth Joint {i+1}')

                ax.set_title(f'Right Joint {i+1}', fontsize=10)
                ax.set_xlabel('Timestep', fontsize=8)
                ax.set_ylabel('Angle (rad)', fontsize=8)
                ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig('/DATA/disk2/1024_grasp_packages/code/pi0_code/val_SF_results/episode_2025-11-03_095210_708_v4.png')
            plt.show()
        else:
            print("No data to plot")



def _get_isaac_env():
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = "Pick and Place package to right part."

    return task_description


if __name__ == "__main__":
    tyro.cli(eval_isaac)