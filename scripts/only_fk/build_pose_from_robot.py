#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
# import sys 

# fk_path =  "/b3-mix03/sppro/permanent/wjxu22/codes/rlds_factory/only_fk"
# urdf_path = "/b3-mix03/sppro/permanent/wjxu22/codes/rlds_factory/piper_description"
# sys.path.append(urdf_path)
# sys.path.append(fk_path)

from fk_only import DualArmFK  # 确保可导入

def find_episode_dirs(roots: list[Path]):
    """在多个根路径下递归查找所有 episode* 目录（去重）"""
    seen = set()
    for root in roots:
        root = root.resolve()
        for p in root.rglob("episode*"):
            if p.is_dir():
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield rp

def ensure_backup_pose(ep_dir: Path):
    """
    若 pose.csv.bak 已存在且 pose.csv 也存在 -> 直接删除 pose.csv
    若 pose.csv 存在但 pose.csv.bak 不存在 -> 备份为 pose.csv.bak
    其余情况不处理
    """
    pose = ep_dir / "pose.csv"
    bak = ep_dir / "pose.csv.bak"

    if pose.exists():
        if bak.exists():
            # 已经有备份，不要覆盖，删除原 pose.csv
            pose.unlink()
            print(f"[INFO] 备份已存在，删除原 pose.csv: {pose}")
        else:
            pose.rename(bak)
            print(f"[INFO] 备份 pose.csv -> pose.csv.bak: {bak}")

def read_robot_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=0, index_col=False, engine="python")

def extract_arm_joint_arrays(df: pd.DataFrame):
    l_cols = [f"l_a_{i}" for i in range(6)]
    r_cols = [f"r_a_{i}" for i in range(6)]
    need = ["timestamp"] + l_cols + r_cols
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"robot.csv 缺少列: {miss}")
    timestamps = df["timestamp"].to_numpy()
    left_j = df[l_cols].to_numpy(dtype=np.float32)
    right_j = df[r_cols].to_numpy(dtype=np.float32)
    return timestamps, left_j, right_j

def compute_poses(
    cur_fk: DualArmFK,
    left_j: np.ndarray,
    right_j: np.ndarray,
    left_tip_index: int,
    right_tip_index: int,
):
    N = left_j.shape[0]
    left_pose = np.zeros((N, 7), dtype=np.float32)
    right_pose = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        l_pose = cur_fk.set_joint_states_get_pose(left_j[i].tolist(),
                                                  base_tip_link_index=left_tip_index)
        r_pose = cur_fk.set_joint_states_get_pose(right_j[i].tolist(),
                                                  base_tip_link_index=right_tip_index)
        left_pose[i] = np.asarray(l_pose, dtype=np.float32)
        right_pose[i] = np.asarray(r_pose, dtype=np.float32)
    return left_pose, right_pose

def write_pose_csv(out_path: Path, timestamps: np.ndarray,
                   left_pose: np.ndarray, right_pose: np.ndarray):
    cols = [
        "timestamp",
        "l_p_x","l_p_y","l_p_z","l_o_w","l_o_x","l_o_y","l_o_z",
        "r_p_x","r_p_y","r_p_z","r_o_w","r_o_x","r_o_y","r_o_z",
    ]
    df = pd.DataFrame(
        np.column_stack([timestamps, left_pose, right_pose]),
        columns=cols
    )
    df.to_csv(out_path, index=False)

def process_one_episode(
    ep_dir: Path,
    fk_cfg: Path,
    chain: str,
    left_tip_index: int,
    right_tip_index: int,
):
    robot_csv = ep_dir / "robot.csv"
    if not robot_csv.exists():
        print(f"[SKIP] 未找到 robot.csv: {ep_dir}")
        return

    # 1) 处理 pose.csv / pose.csv.bak
    ensure_backup_pose(ep_dir)

    # 2) 读取 robot.csv
    df = read_robot_csv(robot_csv)
    timestamps, left_j, right_j = extract_arm_joint_arrays(df)

    # 3) FK
    fk = DualArmFK(str(fk_cfg), chain)

    left_pose, right_pose = compute_poses(
        fk, left_j, right_j, left_tip_index, right_tip_index
    )

    # 4) 写新的 pose.csv
    out_csv = ep_dir / "pose.csv"
    write_pose_csv(out_csv, timestamps, left_pose, right_pose)
    print(f"[OK] 写出: {out_csv}  (N={len(timestamps)})")

def main():
    ap = argparse.ArgumentParser(
        description="在多个根路径下递归查找 episode*，用 robot.csv 关节角经 FK 生成新的 pose.csv"
    )
    ap.add_argument(
        "--roots", type=Path, nargs="+", required=True,
        help="一个或多个递归起点目录（空格分隔）"
    )
    ap.add_argument("--fk-config", type=Path, required=True, help="DualArmFK 的配置 YAML")
    ap.add_argument("--chain", type=str, default="SL",
                    help="urdf版本")
    ap.add_argument("--left-tip-index", type=int, default=0,
                    help="set_joint_states_get_pose 的 base_tip_link_index (左臂)")
    ap.add_argument("--right-tip-index", type=int, default=0,
                    help="set_joint_states_get_pose 的 base_tip_link_index (右臂)")
    args = ap.parse_args()

    roots = [r.resolve() for r in args.roots]
    episodes = sorted(find_episode_dirs(roots))
    if not episodes:
        print(f"[WARN] 未在这些根路径下找到 episode*：{', '.join(map(str, roots))}")
        return

    for ep in episodes:
        try:
            process_one_episode(
                ep, args.fk_config, args.chain,
                args.left_tip_index, args.right_tip_index
            )
        except Exception as e:
            print(f"[ERROR] 处理失败: {ep} -> {e}")

if __name__ == "__main__":
    main()
