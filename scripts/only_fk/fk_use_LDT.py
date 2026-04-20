#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从CSV文件读取关节角并发布：
  1) TF：坐标系 "end_link_pose" 相对于 "base_link"
  2) sensor_msgs/JointState 消息到话题 "/joint_states"

流程：
逐行读取CSV中的 l_a_0..l_a_5 → 作为 cur_joint 输入到
DualArmFK.set_joint_states_get_pose(cur_joint, base_tip_link_index=0)，
返回 [x, y, z, qw, qx, qy, qz]，然后发布TF和JointState。

用法示例：
  ros2 run your_pkg fk_use.py --ros-args -p config:=/path/to/ConfigTracikTeleop.yaml -p csv:=/path/to/data.csv

CSV要求：
- 表头必须包含：timestamp,l_a_0,l_a_1,l_a_2,l_a_3,l_a_4,l_a_5
- timestamp 可以是秒、毫秒或纳秒，会自动转换
- 如果加 loop 参数，会循环播放CSV
"""

import argparse
import csv
import math
import os
import sys
import time
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

# 导入FK类
from fk_only import DualArmFK

# 关节名称列表
JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
]

# 坐标系名称
BASE_FRAMES = ["left_shoulder_base_link","right_shoulder_base_link"]
END_FRAMES = ["left_end_link_pose","right_end_link_pose"]


def _normalize_timestamp(ts_raw: float) -> float:
    """将时间戳归一化为秒。"""
    if ts_raw > 1e12:
        return ts_raw / 1e9
    if ts_raw > 1e9:
        return ts_raw / 1e3
    return ts_raw


def _as_quaternion_xyzw_from_fk_pose(pose: List[float]):
    """FK返回 [x,y,z,qw,qx,qy,qz]，需要转换为 (x,y,z)、(qx,qy,qz,qw)。"""
    if len(pose) != 7:
        raise ValueError("FK返回的pose必须有7个元素")
    x, y, z, qw, qx, qy, qz = pose
    return (x, y, z), (qx, qy, qz, qw)


class FKCSVPlayer(Node):
    def __init__(self, config_path: str, csv_path: str, fk_name: str = "LDT",  loop: bool = False):
        super().__init__('fk_csv_player')
        print(f"FKCSVPlayer初始化，config: {config_path}, csv: {csv_path}, fk_name: {fk_name}, loop: {loop}")
        input("输入enter继续")
        base_tip_indexs = [0, 1]
        # 检查文件
        if not os.path.isfile(csv_path):
            self.get_logger().error(f"找不到CSV文件: {csv_path}")
            sys.exit(1)
        if not os.path.isfile(config_path):
            self.get_logger().error(f"找不到配置文件: {config_path}")
            sys.exit(1)

        # 初始化发布者
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.br = TransformBroadcaster(self)

        # FK
        self.fk = DualArmFK(config_path, fk_name)
        self.base_tip_indexs = base_tip_indexs
        self.loop = loop
        self.csv_path = csv_path

    def run_once(self):
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)

            required_cols = ["timestamp", "l_a_0", "l_a_1", "l_a_2", "l_a_3", "l_a_4", "l_a_5"]
            for col in required_cols:
                if col not in reader.fieldnames:
                    raise RuntimeError(f"CSV缺少必要列 '{col}'，实际列: {reader.fieldnames}")

            rows = list(reader)
            if not rows:
                self.get_logger().warn("CSV没有数据行")
                return

            ts_list = []
            for r in rows:
                try:
                    ts = float(r["timestamp"]) if r["timestamp"] != "" else float("nan")
                except Exception:
                    ts = float("nan")
                ts_list.append(ts)

            have_all_ts = all(not math.isnan(x) for x in ts_list)
            if have_all_ts:
                ts_list = [_normalize_timestamp(x) for x in ts_list]
                csv_t0 = ts_list[0]
                start_wall = time.time()
            else:
                csv_t0 = None
                start_wall = None
                self.get_logger().warn("时间戳不完整，使用固定频率发布")

            for i, row in enumerate(rows):
                if not rclpy.ok():
                    break

                # 关节角
                try:
                    left_cur_joint = [
                        float(row["l_a_0"]),
                        float(row["l_a_1"]),
                        float(row["l_a_2"]),
                        float(row["l_a_3"]),
                        float(row["l_a_4"]),
                        float(row["l_a_5"]),
                        float(row["l_a_6"]),
                    ]
                    right_cur_joint = [
                        float(row["r_a_0"]),
                        float(row["r_a_1"]),
                        float(row["r_a_2"]),
                        float(row["r_a_3"]),
                        float(row["r_a_4"]),
                        float(row["r_a_5"]),
                        float(row["r_a_6"]),
                    ]
                    cur_joint = left_cur_joint+right_cur_joint
                except Exception as e:
                    self.get_logger().warn(f"第{i}行关节角无效，跳过。错误: {e}")
                    continue

                # FK计算
                try:
                    left_pose = self.fk.set_joint_states_get_pose(cur_joint, base_tip_link_index=self.base_tip_indexs[0])
                    right_pose = self.fk.set_joint_states_get_pose(cur_joint, base_tip_link_index=self.base_tip_indexs[1])
                    # pose = self.fk.set_joint_states_get_pose_from_base_link(cur_joint, "gripper_base")
                except Exception as e:
                    self.get_logger().warn(f"第{i}行FK计算失败，跳过。错误: {e}")
                    continue

                # 发布TF
                trans, quat_xyzw = _as_quaternion_xyzw_from_fk_pose(left_pose)
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = BASE_FRAMES[0]
                t.child_frame_id = END_FRAMES[0]
                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = trans[2]
                t.transform.rotation.x = quat_xyzw[0]
                t.transform.rotation.y = quat_xyzw[1]
                t.transform.rotation.z = quat_xyzw[2]
                t.transform.rotation.w = quat_xyzw[3]
                self.br.sendTransform(t)

                trans, quat_xyzw = _as_quaternion_xyzw_from_fk_pose(right_pose)
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = BASE_FRAMES[1]
                t.child_frame_id = END_FRAMES[1]
                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = trans[2]
                t.transform.rotation.x = quat_xyzw[0]
                t.transform.rotation.y = quat_xyzw[1]
                t.transform.rotation.z = quat_xyzw[2]
                t.transform.rotation.w = quat_xyzw[3]
                self.br.sendTransform(t)

                
                # 发布JointState
                js = JointState()
                if have_all_ts:
                    js.header.stamp = rclpy.time.Time(seconds=ts_list[i]).to_msg()
                else:
                    js.header.stamp = self.get_clock().now().to_msg()
                js.name = JOINT_NAMES
                js.position = left_cur_joint+right_cur_joint
                print(f"第{i}行JointState发布: {JOINT_NAMES} {js.position}")
                self.js_pub.publish(js)

                # 控制速率
                time.sleep(0.005*3)


def main():
    parser = argparse.ArgumentParser(description="从CSV读取关节角并发布TF与JointState (ROS2)")
    parser.add_argument("--config", required=True, help="ConfigTracikTeleop.yaml 文件路径")
    parser.add_argument("--csv", required=True, help="CSV文件路径")
    parser.add_argument("--fk-name", default="LDT", help="DualArmFK 名称，默认LDT")
    parser.add_argument("--loop", action="store_true", help="是否循环播放CSV")

    args = parser.parse_args()

    rclpy.init()
    node = FKCSVPlayer(args.config, args.csv, args.fk_name,  args.loop)

    try:
        while rclpy.ok():
            node.run_once()
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


"""
python3 /home/rsluo/project/about_tele/iflytek_armtele_project/src/fk_only/fk_use_LDT.py  --config /home/rsluo/project/about_tele/iflytek_armtele_project/src/fk_only/ConfigFK.yaml --csv /home/rsluo/下载/0919-1031/robot_data.csv

"""
