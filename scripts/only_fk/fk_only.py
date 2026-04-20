# fk_only.py
import yaml
import numpy as np
import pybullet as p
import threading
import time
# 工具函数
from util_calibrate import load_urdf, get_link_relative_pose_from_joints,cleanup_robot,get_link_pose_from_joints

def _rotmat_to_quat(R):
    """3x3 -> (qx,qy,qz,qw)，数值稳定的旋转矩阵转四元数实现"""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        S = (trace + 1.0)**0.5 * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = (1.0 + m00 - m11 - m22)**0.5 * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = (1.0 + m11 - m00 - m22)**0.5 * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = (1.0 + m22 - m00 - m11)**0.5 * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    # 归一化
    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    return (qx/n, qy/n, qz/n, qw/n)

class DualArmFK:
    """
    只做 FK（正解）：
      - set_left_joint_states(joints) / get_left_pose()
      - set_right_joint_states(joints) / get_right_pose()
    joints 顺序与 YAML 的 slave_joint_names.{left_arm,right_arm} 一致（单位：弧度）
    pose 返回 [x,y,z,qw,qx,qy,qz]，坐标系 base_link → {left,right}_tip_link
    """
    def __init__(self, yaml_path: str,fk_name: str):
        t0 = time.perf_counter()

        self.lock = threading.Lock()
        self.load_cfg(yaml_path,fk_name)

        init_ms = (time.perf_counter() - t0) * 1e3
        print(f"[DualArmFK] init took {init_ms:.2f} ms")

    def load_cfg(self,yaml_path: str,fk_name:str):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # 你的 YAML 可能外面还包了一层 "yaml:"，做个兼容
        if fk_name not in cfg :
            raise RuntimeError(f"{fk_name} not in {yaml_path}")
        if 'base_links' not in cfg[fk_name] :
            raise RuntimeError(f"base_links not in {fk_name}")
        if 'tip_links' not in cfg[fk_name] :
            raise RuntimeError(f"base_link not in {fk_name}")
        if 'joint_names' not in cfg[fk_name] :
            raise RuntimeError(f"base_link not in {fk_name}")
        if 'urdf_path' not in cfg[fk_name] :
            raise RuntimeError(f"base_link not in {fk_name}")
        self.base_links   = cfg[fk_name]['base_links']
        self.tip_links    = cfg[fk_name]['tip_links']
        self.joint_names  = cfg[fk_name]['joint_names']
        self.urdf_path  = cfg[fk_name]['urdf_path']
        print(f"[DualArmFK] load_cfg {fk_name} base_links: {self.base_links} tip_links: {self.tip_links} joint_names: {self.joint_names} urdf_path: {self.urdf_path}")
        if len(self.base_links) != len(self.tip_links):
            raise RuntimeError(f"len(self.base_links) != len(self.tip_links)")

        # ——加载 URDF 与末端（拿到 robot_id & 关节映射）——
        self.robot_id, self.joint_name_to_id,self.physics_client = load_urdf(self.urdf_path)

        self.tip_link_ids = {}
        check_found = True
        for i in range(len(self.tip_links)):
            cur_id = self._find_link_id(self.tip_links[i])
            if  cur_id < 0:
                check_found = False
                print((f"未找到 tip_link id: {self.tip_links[i]}"))
            self.tip_link_ids[self.tip_links[i]]  =  cur_id

        if check_found == False:
            raise RuntimeError(f"未找到 tip_link id: {self.tip_links[i]}")

        check_found = True
        self.base_link_ids = {}
        for i in range(len(self.base_links)):
            cur_id = self._find_link_id(self.base_links[i])
            if  cur_id < 0:
                check_found = False
                print((f"未找到 bask_link id: {self.base_links[i]}"))
            self.base_link_ids[self.base_links[i]]  =  cur_id

        if check_found == False:
            raise RuntimeError(f"未找到 bask_link id: {self.base_links[i]}")

    def __del__(self):
        try:
            cleanup_robot(getattr(self, "robot_id", -1), getattr(self, "physics_client", -1))
            # 防止重复清理
            self.robot_id = -1
            self.physics_client = -1
        except Exception:
            pass
    
    def set_joint_states_get_pose(self, joints,base_tip_link_index):
        if base_tip_link_index >= len(self.base_links):
            raise RuntimeError(f" base_tip_link_index >= len(self.base_links): {base_tip_link_index} >= {len(self.base_links)}")
        with self.lock:
            cur_np_j = np.array(joints, dtype=float)
            T = get_link_relative_pose_from_joints(
                self.robot_id, self.joint_name_to_id, self.joint_names, cur_np_j,
                self.tip_link_ids[self.tip_links[base_tip_link_index]],
                self.base_link_ids[self.base_links[base_tip_link_index]]
            )
            
            cur_pose = self._T_to_pose(T)
            return cur_pose[:]
        
    def set_joint_states_get_pose_from_base_link(self, joints,tip_link_name):

        if tip_link_name not in self.tip_link_ids:
            raise RuntimeError(f" tip_link_name not in self.tip_link_ids: {tip_link_name}")
        with self.lock:
            cur_np_j = np.array(joints, dtype=float)
            T = get_link_pose_from_joints(self.robot_id,self.joint_name_to_id, self.joint_names, cur_np_j, self.tip_link_ids[tip_link_name])
            cur_pose = self._T_to_pose(T)
            return cur_pose[:]
        
    # ——内部工具——
    def _find_link_id(self, link_name: str) -> int:
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            lname = info[12].decode("utf-8")
            if lname == link_name:
                return j
        return -1

    @staticmethod
    def _T_to_pose(T):
            """4x4 → [x,y,z,qw,qx,qy,qz]，并将四元数规范到 w>=0"""
            x, y, z = T[:3, 3].tolist()
            qx, qy, qz, qw = _rotmat_to_quat(T[:3, :3])

            # # 规范化到同一半球：若 w 为负，整体取反（姿态不变）
            # if qw < 0.0:
            #     qx, qy, qz, qw = -qx, -qy, -qz, -qw

            # 返回顺序改为 [x,y,z,qw,qx,qy,qz]
            return [float(x), float(y), float(z), float(qw), float(qx), float(qy), float(qz)]





if __name__ == '__main__':
    
    fk_sl = DualArmFK('/b3-mix03/sppro/permanent/wjxu22/codes/rlds_factory/only_fk/ConfigFK.yaml',"SL")

    # 给 对应 个关节角（弧度） → 取末端位姿
    cur_joint = [0,	0.013117888,	-0.00366324,	0.021002576,	0.01037918,	0]
    t0 = time.perf_counter()
    print("pose:", fk_sl.set_joint_states_get_pose(cur_joint,base_tip_link_index=0))   # -> [x,y,z,qw,qx,qy,qz]
    init_ms = (time.perf_counter() - t0) * 1e3
    print(f"[DualArmFK] init took {init_ms:.2f} ms")

    fk_ldt = DualArmFK('/b3-mix03/sppro/permanent/wjxu22/codes/rlds_factory/only_fk/ConfigFK.yaml',"LDT")

    # 给 对应 个关节角（弧度） → 取末端位姿
    cur_joint = [-0.264173,	0.317929,	-0.61282,	-1.8231,	-0.191986,	0.0568279,	1.15806,
                 0.322537,	-0.233455,	0.969146,	1.93983,	0.238063,	0.291819,	-1.0014,
                 ]
    t0 = time.perf_counter()
    print("left_pose:", fk_ldt.set_joint_states_get_pose(cur_joint,base_tip_link_index=0))   # -> [x,y,z,qw,qx,qy,qz]
    print("right_pose:", fk_ldt.set_joint_states_get_pose(cur_joint,base_tip_link_index=1))   # -> [x,y,z,qw,qx,qy,qz]
    init_ms = (time.perf_counter() - t0) * 1e3
    print(f"[DualArmFK] init took {init_ms:.2f} ms")


# cur_joint = [0,	0.013117888,	-0.00366324,	0.021002576,	0.01037918,	0]
# pose csv 0.0572	1.90E-05	0.212827	0.007101822	0.682829002	0.007755505	0.730502546
# fk pose [0.05721336230635643, 2.1623403881676495e-05, 0.21274925768375397, 0.7304797971646753, 0.007093451183027943, 0.6828535985026895, 0.00774027754648152]