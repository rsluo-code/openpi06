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


prompt_map={
    "短袖": "fold the T-shirt neatly",
    "绿茶": "pick up the bottled tea and put it on the plate",
    "香蕉": "pick up the banana and put it on the plate",
    "茶饮": "pick up the tea and put it on the plate",
    "矿泉水": "pick up the water and put it on the plate",
    "猕猴桃": "pick up the kiwi and put it on the plate",
    "可乐": "pick up the cola and put it on the plate",
    "布绒玩具": "pick up the plush toy and put it on the plate",
    "螺丝刀": "pick up the screwdriver and put it on the plate",
    "盲盒": "pick up the square paper box and put it on the plate",
    "手电筒": "pick up the flashlight and put it on the plate",
    "魔方": "pick up the Rubik's cube and put it on the plate",
    "芒果": "pick up the mango and put it on the plate",
    "苹果": "pick up the apple and put it on the plate",
    "饼干": "pick up the rectangular box-packed biscuits and put it on the plate",
    "桃子": "pick up the peach and put it on the plate",
    "山竹": "pick up the mangosteen and put it on the plate",
    "纸巾": "pick up a pack of tissues and put it on the plate",
    "面包": "pick up the packaged snack cakes and put it on the plate",
    "柠檬": "pick up the lemon and put it on the plate",
    "杯子": "pick up the paper cup and put it on the plate",
    "毛巾": "pick up the towel and put it on the plate",
    "卷尺": "pick up the measure and put it on the plate",
    "碗": "pick up the bowl and place it on the plate",
}
from openpi.models_pytorch.some_func import get_index_and_max_len


# LEIBIE = "可乐"
# LEIBIE = "苹果"
LEIBIE = "茶饮"
# LEIBIE = "香蕉"
_, max_len,_ = get_index_and_max_len(prompt_map[LEIBIE])
NAME_SAVE = f"pi06__{LEIBIE}"
PATH_SAVE_BASE = f"/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/z_experimental_result/"


"""Rest everything else."""
@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8001
    replan_steps: int = 30

    num_trials_per_task: int = 1  # Number of rollouts per task
    episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd3/1009_04_茶饮_LD_冰红茶_197/bg_tablecloth1:arm_right:object_1/episode_2025-10-09_10:46:47_583"
    prompt: str = prompt_map[LEIBIE]


ORIGIN_IMG_H = 720
ORIGIN_IMG_W = 1280
TARGET_IMG_MAX = int(224)
if ORIGIN_IMG_H >= ORIGIN_IMG_W:
    TARGET_IMG_H = TARGET_IMG_MAX
    scale = TARGET_IMG_H / ORIGIN_IMG_H
    TARGET_IMG_W = int(ORIGIN_IMG_W * scale)
else:
    TARGET_IMG_W = TARGET_IMG_MAX
    scale = TARGET_IMG_W / ORIGIN_IMG_W
    TARGET_IMG_H = int(ORIGIN_IMG_H * scale)
import av
def _load_video_frames(video_path: str):
    # 1. 解码
    size=(TARGET_IMG_W, TARGET_IMG_H)
    with av.open(video_path) as container:
        frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]

    # 2. 批量 resize
    resized_frames = [cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
                      for frame in frames]
    # 3. stack
    return np.stack(resized_frames).astype(np.uint8)

def eval_isaac(args: Args) -> None:

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Initialize LIBERO environment and task description
    task_description = args.prompt

    # Start episodes
    task_episodes, task_successes = 0, 0
    action_success = 0
    done_flag = False
    episode_dir = args.episode_dir
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
    head_frames = _load_video_frames(head_video_path)
    right_frames = _load_video_frames(right_video_path)
    left_frames = _load_video_frames(left_video_path)

    L = min(states.shape[0], len(head_frames), len(right_frames))


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
        action_N = 30
        for i in range(states.shape[0]-action_N):
            try:
                img = head_frames[i]
                right_wrist_img = right_frames[i]
                left_wrist_img = left_frames[i]

                img = einops.rearrange(img, "h w c -> c h w")
                right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
                left_wrist_img = einops.rearrange(left_wrist_img, "h w c -> c h w")

                img_N = head_frames[i+action_N]
                right_wrist_img_N = right_frames[i+action_N]
                left_wrist_img_N = left_frames[i+action_N]
                img_N = einops.rearrange(img_N, "h w c -> c h w")
                right_wrist_img_N = einops.rearrange(right_wrist_img_N, "h w c -> c h w")
                left_wrist_img_N = einops.rearrange(left_wrist_img_N, "h w c -> c h w")
                actions_i = actions[i]
                actions_i_161 = actions_i.reshape(16, 1)
                if not action_plan:

                    element = {
                        "observation/image": img,
                        "observation/wrist_image_right": right_wrist_img,
                        "observation/wrist_image_left":left_wrist_img,
                        "observation/joint_position": states[i],
                        "observation/image_N": img_N,
                        "observation/wrist_image_right_N": right_wrist_img_N,
                        "observation/wrist_image_left_N": left_wrist_img_N,
                        "observation/joint_position_N": states[i+action_N],

                        "prompt": str(task_description),
                        "step_index": i,
                        "episode_length": L,
                        # "actions":actions_i_161,
                        # "actions":actions[i],
                        "language_instruction_index":1,
                        "language_instruction_max_len":L+1,
                        "language_instruction_at_30precent":-0.0,
                        "success_or_failure":1,
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
        plt.savefig(f"{PATH_SAVE_BASE}/{NAME_SAVE}.png")
        # plt.show()






if __name__ == "__main__":
    tyro.cli(eval_isaac)
