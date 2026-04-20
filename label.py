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
from scripts.fk_only.fk_only import DualArmFK

"""Rest everything else."""

ISAAC_DUMMY_ACTION = [0.0] * 7 + [1.0]

episode_dir = "/b3-mix03/sppro/permanent/yuanzhang10/资源部传输数据/叠衣服/SL-叠衣服-全检结束-铲夹起/SL-短袖1001_全检结束/1001_短袖_04_短袖_SL_1件/episode_20251001144430"

def eval_isaac() -> None:
    # Start episodes
    csv_path = episode_dir + "/robot.csv"
    pose_path = episode_dir + "/pose.csv"
    fk = DualArmFK('/work2/cv5/jfni3/openpi06/scripts/fk_only/ConfigTracikTeleop.yaml',"fk1")
    
    # 加载动作和状态数据
    df = pd.read_csv(
        csv_path,
        header=0,  # 使用第一行作为列名
        index_col=False,  # 禁止将首列作为索引
        usecols=lambda col: not col.startswith('Unnamed')  # 过滤自动生成的索引列
    )

    df_pose = pd.read_csv(
        pose_path,
        header=0,  # 使用第一行作为列名
        index_col=False,  # 禁止将首列作为索引
        usecols=lambda col: not col.startswith('Unnamed')  # 过滤自动生成的索引列
    )

    actions_joint = df[[f"l_a_{i}" for i in range(7)] + [f"r_a_{i}" for i in range(7)]].values.astype(np.float32)
    for episode_idx in tqdm.tqdm(range(1)):

        # Setup
        t = 0
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
        for i in range(actions_joint.shape[0]):
            try:                
                # import pdb
                # pdb.set_trace()
                # pose1 = fk.set_joint_states_get_pose([-0.50484681, 1.83702767, -1.12403905, -1.04149401, 0.98194021, 0.60637087], base_tip_link_index=0)
                # pose2 = fk.set_joint_states_get_pose([0.11438031, 0.21021764, -0.45431155, 0, 0.78238082, 0.02393317], base_tip_link_index=0)
                # print(pose2)
                actions_l = fk.set_joint_states_get_pose(actions_joint[i][:6],base_tip_link_index=0)   # -> [x,y,z,qw,qx,qy,qz]
                actions_r = fk.set_joint_states_get_pose(actions_joint[i][7:13],base_tip_link_index=0)
                label1.append(actions_l[0])
                label2.append(actions_l[1])
                label3.append(actions_l[2])
                label4.append(actions_l[3])
                label5.append(actions_l[4])
                label6.append(actions_l[5])
                label7.append(actions_l[6])
                label8.append(actions_joint[i][6])
                label9.append(actions_r[0])
                label10.append(actions_r[1])
                label11.append(actions_r[2])
                label12.append(actions_r[3])
                label13.append(actions_r[4])
                label14.append(actions_r[5])
                label15.append(actions_r[6])
                label16.append(actions_joint[i][13])
            except Exception as e:
                print(f"Caught exception: {e}")
                break
        all_labels = [label1, label2, label3, label4, label5, label6, label7, label8, label9, label10, label11, label12, label13, label14, label15, label16]
        timesteps = np.arange(len(label1)) 

        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        fig.suptitle('Joint Angles', fontsize=16, y=1.02)

        for i, (ax, label_data) in enumerate(zip(axes.flatten(), all_labels)):
            ax.plot(timesteps, label_data,
                color='red',
                linewidth=0.2,
                marker='o',
                markersize=1,
                label=f'Label {i+1}')
            ax.set_xlabel('Timestep', fontsize=8)
            ax.set_ylabel('Angle (rad)', fontsize=8)
            ax.legend(fontsize=8)

        # axes.flatten()[14].set_visible(False)
        plt.tight_layout()
        plt.savefig('/work2/cv5/jfni3/openpi06/label.png')
        # plt.show()

if __name__ == "__main__":
    tyro.cli(eval_isaac)
