import os
import cv2
import yaml
import numpy as np
import pandas as pd
np.set_printoptions(precision=5, suppress=True)
import pybullet as p
import pybullet_data
from urdf_parser_py.urdf import URDF
from glob import glob

def load_urdf_endlink(urdf_path,end_link):
    # 2. 加载URDF和CSV文件

    # 加载 URDF 文件
    try:
        # 1. 初始化 pybullet（无图形界面模式）
        # p.connect(p.DIRECT)  
        physics_client = p.connect(p.DIRECT) # 不显示物理模拟窗口，仅用于运动学计算
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 加载默认数据（可选）

        # 2. 加载 URDF 模型
        robot_id = p.loadURDF(urdf_path, useFixedBase=True)  # useFixedBase=True 固定机器人基座
        print(f"URDF 加载成功，机器人 ID: {robot_id}")

        # 3. 获取关节和链接信息（用于映射关节名称到 ID）
        joint_name_to_id = {}
        for j in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, j)
            joint_name = info[1].decode("utf-8")  # 关节名称（字节转字符串）
            joint_name_to_id[joint_name] = j

        # 4. 获取末端链接 ID（根据 end_link 名称）
        end_link_id = -1
        for j in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, j)
            link_name = info[12].decode("utf-8")  # 链接名称
            if link_name == end_link:
                end_link_id = j
                break
        if end_link_id == -1:
            raise ValueError(f"未找到末端链接: {end_link}")
    except Exception as e:
        raise RuntimeError(f"加载 URDF 文件({urdf_path})失败: {str(e)}")
    return robot_id,joint_name_to_id,end_link_id,physics_client



def load_urdf(urdf_path):
    # 2. 加载URDF和CSV文件

    # 加载 URDF 文件
    try:
        # 1. 初始化 pybullet（无图形界面模式）
        # p.connect(p.DIRECT)  
        physics_client = p.connect(p.DIRECT) # 不显示物理模拟窗口，仅用于运动学计算
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 加载默认数据（可选）

        # 2. 加载 URDF 模型
        robot_id = p.loadURDF(urdf_path, useFixedBase=True)  # useFixedBase=True 固定机器人基座
        print(f"URDF 加载成功，机器人 ID: {robot_id}")

        # 3. 获取关节和链接信息（用于映射关节名称到 ID）
        joint_name_to_id = {}
        for j in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, j)
            joint_name = info[1].decode("utf-8")  # 关节名称（字节转字符串）
            joint_name_to_id[joint_name] = j

    except Exception as e:
        raise RuntimeError(f"加载 URDF 文件({urdf_path})失败: {str(e)}")
    return robot_id,joint_name_to_id,physics_client

def cleanup_robot(robot_id, physics_client):
    print(f"清理机器人 {robot_id}")
    if robot_id is not None:
        p.removeBody(robot_id)  # 移除模型
    if physics_client >= 0:
        p.disconnect(physics_client)  # 断开连接，释放所有资源


def get_link_pose_from_joints(robot_id,joint_name_to_id,joint_names, joint_angles,end_link_id):
    """使用 pybullet 计算 base_link 到 end_link 的变换矩阵"""
    # 设置关节角度
    for name, angle in zip(joint_names, joint_angles):
        if name in joint_name_to_id:
            joint_id = joint_name_to_id[name]
            # 设置关节位置（忽略动力学，仅用于运动学计算）
            p.resetJointState(robot_id, joint_id, targetValue=angle)

    # 获取末端链接的位姿（相对于基座坐标系）
    link_state = p.getLinkState(robot_id, end_link_id, computeForwardKinematics=True)
    # link_state[4] 是末端位置（x,y,z），link_state[5] 是末端姿态（四元数 x,y,z,w）
    position = link_state[4]
    orientation = link_state[5]

    # 将四元数转换为旋转矩阵
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    R = np.array(rot_matrix).reshape(3, 3)

    # 构建 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T

def _pose_to_T(pos, orn):
    """(pos, quat) -> 4x4 T"""
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = np.array(pos)
    return T

def _T_inv(T):
    """快速求齐次矩阵的逆：R^T, -R^T t"""
    R = T[:3, :3]
    t = T[:3,  3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti

def world_T(robot_id, link_id):
    """
    指定 link 在世界坐标系下的 4x4 齐次矩阵。
    link_id = -1 表示 base。
    """
    if link_id == -1:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
    else:
        ls = p.getLinkState(robot_id, link_id, computeForwardKinematics=True)
        # 使用 worldLinkFrame（ls[4], ls[5]）
        pos, orn = ls[4], ls[5]
    return _pose_to_T(pos, orn)

def relative_T(robot_id, link_from_id, link_to_id):
    """
    返回 T_to_from（link_from 相对 link_to 的 4x4 矩阵）：
    T_to_from = inv(T_world_to) @ T_world_from
    """
    T_w_from = world_T(robot_id, link_from_id)
    T_w_to   = world_T(robot_id, link_to_id)
    return _T_inv(T_w_to) @ T_w_from

def get_link_relative_pose_from_joints(robot_id, joint_name_to_id, joint_names, joint_angles,link_from_id,link_to_id):
    """使用 pybullet 计算 base_link 到 end_link 的变换矩阵"""
    # 设置关节角度
    for name, angle in zip(joint_names, joint_angles):
        if name in joint_name_to_id:
            joint_id = joint_name_to_id[name]
            # 设置关节位置（忽略动力学，仅用于运动学计算）
            p.resetJointState(robot_id, joint_id, targetValue=angle)
    
    return relative_T(robot_id,link_from_id,link_to_id)


def compute_eye_in_hand_calibration(T_base_to_end_list, T_chess_to_camera_list):
    """眼在手上的手眼标定算法（使用OpenCV内置函数）"""
    if len(T_base_to_end_list) < 3:
        raise ValueError("手眼标定需要至少3组位姿数据")
    
    # 提取旋转矩阵和平移向量
    R_gripper2base = [T[:3, :3] for T in T_base_to_end_list]
    t_gripper2base = [T[:3, 3] for T in T_base_to_end_list]
    R_target2cam = [T[:3, :3] for T in T_chess_to_camera_list]
    t_target2cam = [T[:3, 3] for T in T_chess_to_camera_list]
    
    # 调用OpenCV手眼标定函数（AX=XB）
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS  # 可选：TSAI/LENZ/ZHANG等算法
    )
    print(f"R_cam2gripper:\n{R_cam2gripper}")
    print(f"t_cam2gripper:\n{t_cam2gripper}")
    # 构建末端到相机的变换矩阵
    # T_end_to_camera = np.eye(4)
    # T_end_to_camera[:3, :3] = R_cam2gripper
    # T_end_to_camera[:3, 3] = t_cam2gripper.flatten()
    # return T_end_to_camera

    T_end_to_camera = np.hstack((R_cam2gripper, t_cam2gripper))
    T_end_to_camera = np.vstack((T_end_to_camera, np.array([0,0,0,1])))
    T_camera_to_end = np.linalg.inv(T_end_to_camera)
    
    R_mat = T_end_to_camera[:3, :3]
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_matrix(R_mat)
    # 提取欧拉角（rpy，XYZ 顺序）
    rpy_rad = rotation.as_euler('xyz', degrees=False)
    rpy_deg = rotation.as_euler('xyz', degrees=True)
    print("RPY (rad):", rpy_rad)
    print("RPY (deg):", rpy_deg)
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_euler('xyz', rpy_rad)
    quat = rotation.as_quat()  # 返回格式：[x, y, z, w]
    # 打印结果
    print("Quaternion [x, y, z, w]:", quat)
    return T_end_to_camera


def compute_eye_out_hand_calibration(T_base_to_end_list, T_chess_to_camera_list):
    """眼在手外的手眼标定算法（使用OpenCV内置函数）"""
    if len(T_base_to_end_list) < 3:
        raise ValueError("手眼标定需要至少3组位姿数据")
    
    # 提取旋转矩阵和平移向量
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    for T_be, T_ct in zip(T_base_to_end_list, T_chess_to_camera_list):
        # base → end (正向变换) → 取逆 得到 end → base（相对运动）
        T_eb = np.linalg.inv(T_be)
        R_eb = T_eb[:3, :3]
        t_eb = T_eb[:3, 3]
        R_gripper2base.append(R_eb)
        t_gripper2base.append(t_eb)

        # chess → camera (正向变换) → 取逆 得到 camera → chess
        T_tc = np.linalg.inv(T_ct)
        R_tc = T_tc[:3, :3]
        t_tc = T_tc[:3, 3]
        R_tc = T_ct[:3, :3]
        t_tc = T_ct[:3, 3]
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)
    
    # 调用OpenCV手眼标定函数（AX=XB）
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS  # 可选：TSAI/LENZ/ZHANG等算法
    )
    print(f"R_cam2gripper:\n{R_cam2gripper}")
    print(f"t_cam2gripper:\n{t_cam2gripper}")

    T_base_to_camera = np.hstack((R_cam2gripper, t_cam2gripper))
    T_base_to_camera = np.vstack((T_base_to_camera, np.array([0,0,0,1])))
    T_camera_to_base = np.linalg.inv(T_base_to_camera)
    
    R_mat = T_base_to_camera[:3, :3]
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_matrix(R_mat)
    # 提取欧拉角（rpy，XYZ 顺序）
    rpy_rad = rotation.as_euler('xyz', degrees=False)
    rpy_deg = rotation.as_euler('xyz', degrees=True)
    print("RPY (rad):", rpy_rad)
    print("RPY (deg):", rpy_deg)
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_euler('xyz', rpy_rad)
    quat = rotation.as_quat()  # 返回格式：[x, y, z, w]
    # 打印结果
    print("Quaternion [x, y, z, w]:", quat)


    return T_base_to_camera


def generate_object_points(chessboard_size, square_size):
    """生成棋盘格世界坐标系点（假设z=0）"""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp



def generate_apriltag_object_points(tag_size=0.04):
    """生成AprilTag的四个角点在自身坐标系中的3D坐标"""
    half = tag_size / 2.0
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)




def calibrate_camera_intrinsic(image_dir,chessboard_size=(11, 8),square_size=0.030,output_path="intrinsic.yaml"):
    """
    使用棋盘格图像标定相机内参
    参数:
        image_dir: 图像文件夹路径
        chessboard_size: 棋盘格内角点数 (列数, 行数)
        square_size: 每个格子的实际边长（单位：米）
        output_path: 标定结果保存路径（YAML）
    """
    # 准备棋盘格的3D点坐标
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 所有图像的3D点
    img_points = []  # 所有图像的2D角点
    image_size = None

    # 遍历图像
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    # 总数量
    total = len(image_files)

    # 目标数量
    target = 80

    # 如果图像总数不够 80，就全选
    if total <= target:
        image_files = image_files
    else:
        step = total / target
        image_files = [image_files[int(i * step)] for i in range(target)]


    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # 检测角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not ret:
            print(f"✗ 跳过未检测成功的图像: {fname}")
            continue

        # 亚像素优化
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        obj_points.append(objp)
        img_points.append(corners2)
        print(f"✓ 检测成功: {fname}")

    # 检查样本数
    if len(obj_points) < 3:
        raise ValueError("有效图像数量不足，至少需要3张成功检测的图像。")

    # 执行标定
    print(f" 相机内参标定开始")
    ret, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print("\n✅ 相机内参标定完成")
    print("K:\n", K)
    print("distortion coefficients:", dist.ravel())
    print(f"重投影误差: {ret:.4f} px")

    # 保存结果
    data = {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "distortion_model": "plumb_bob",
        "reprojection_error": float(f"{ret:.3f}"), 
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"📁 内参结果保存至: {output_path}")

    return K, dist, ret


def PrintTFcmd(T_base_to_camera,tf_a_link,tf_b_link):
    T_result = T_base_to_camera

    from transformations import quaternion_from_matrix
    # 提取平移
    x, y, z = T_result[0:3, 3]

    # 提取四元数
    qw, qx, qy, qz = quaternion_from_matrix(T_result)

    # 格式化为字符串，保留6位小数
    cmd = (
        f"ros2 run tf2_ros static_transform_publisher "
        f"{x:.6f} {y:.6f} {z:.6f} "
        f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} "
        f"{tf_a_link} {tf_b_link}"
    )
    print(f"四元数 qx, qy, qz, qw = {qx:.3f} {qy:.3f} {qz:.3f} {qw:.3f}")
    print(f"✅ 复制下面这条命令在终端中运行即可：\n{cmd}")




# ==== 加载 YAML ====
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ==== 提取棋盘角点位姿 ====
def get_chessboard_pose(image, camera_matrix, dist_coeffs, board_size=(11, 8), square_size=0.030):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size)
    if not found:
        return None, None
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not ret:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    T_chess_to_cam = np.eye(4)
    T_chess_to_cam[:3, :3] = R
    T_chess_to_cam[:3, 3] = tvec.flatten()
    return T_chess_to_cam, objp

# ==== 相机坐标系下角点变换到 base 坐标系 ====
def transform_to_base(T_cam_to_base, T_chess_to_cam, obj_points):
    # 将 obj_points（在棋盘坐标系）变换到 base 坐标系
    chess_points_homo = np.hstack((obj_points, np.ones((len(obj_points), 1)))).T  # 4 x N
    points_in_cam = T_chess_to_cam @ chess_points_homo  # 4 x N
    points_in_base = T_cam_to_base @ points_in_cam      # 4 x N
    return points_in_base[:3, :].T  # Nx3

def resize_camera_matrix_from_yaml(yaml_data, current_shape):
    """
    根据 YAML 中的相机内参和当前图像 shape，自动缩放 camera_matrix。
    - yaml_data: dict, 读取的 intrinsic yaml 数据
    - current_shape: (height, width) 当前图像尺寸
    """
    orig_w = yaml_data['image_width']
    orig_h = yaml_data['image_height']
    curr_h, curr_w = current_shape

    scale_x = curr_w / orig_w
    scale_y = curr_h / orig_h

    cam_matrix = np.array(yaml_data['camera_matrix'])
    scaled = cam_matrix.copy()
    scaled[0, 0] *= scale_x  # fx
    scaled[0, 2] *= scale_x  # cx
    scaled[1, 1] *= scale_y  # fy
    scaled[1, 2] *= scale_y  # cy

    return scaled


# ==== 主流程 ====
def process_calibration_compare(
    data_dir,
    head_intrinsic_path,
    head_extrinsic_path,
    side_intrinsic_path,
    side_extrinsic_path,
    csv_path,
    urdf_path,
    side_camera_end_link,
    side,
    error_yaml_name
):
    # 1. 加载相机标定参数
    head_intrinsic = load_yaml(head_intrinsic_path)
    head_extrinsic = np.array(load_yaml(head_extrinsic_path)['extrinsic'])

    side_intrinsic_data = load_yaml(side_intrinsic_path)
    side_intrinsic = np.array(side_intrinsic_data['camera_matrix'])
    side_dist = np.array(side_intrinsic_data['distortion_coefficients'])

    side_extrinsic_data = load_yaml(side_extrinsic_path)
    side_extrinsic = np.array(side_extrinsic_data['extrinsic'])
    side_joint_names = side_extrinsic_data['joint_subset']

    # 2. 读取CSV
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['timestamp'] + [f'joint_{i}' for i in range(1, 8)]

    # 3. 加载 URDF 模型
    robot_id, joint_name_to_id, end_link_id = load_urdf_endlink(urdf_path, side_camera_end_link)

    errors = []
    valid_count = 0
    mean_errors_x = []
    mean_errors_y = []
    mean_errors_z = []
    mean_abs_errors_x = []
    mean_abs_errors_y = []
    mean_abs_errors_z = []
    for i in range(len(df)):
        index = i + 1
        head_img_path = os.path.join(data_dir, f"camhead_cam{side}_compare_cam_head", f"camhead_cam{side}_compare_cam_head_{index}.jpeg")
        side_img_path = os.path.join(data_dir, f"camhead_cam{side}_compare_cam_{side}", f"camhead_cam{side}_compare_cam_{side}_{index}.jpeg")

        if not os.path.exists(head_img_path) or not os.path.exists(side_img_path):
            print(f"[{index}] 缺失图像，跳过")
            continue

        img_head = cv2.imread(head_img_path)
        img_side = cv2.imread(side_img_path)

        # 获取当前图像 shape
        curr_shape = img_side.shape[:2]  # (height, width)

        # 缩放内参
        scaled_head_intrinsic = resize_camera_matrix_from_yaml(head_intrinsic, curr_shape)
        scaled_side_intrinsic = resize_camera_matrix_from_yaml(side_intrinsic_data, curr_shape)

        # 4. 提取角点位姿（T_chess_to_cam）
        T_chess_to_head, obj_points = get_chessboard_pose(
            img_head,
            scaled_head_intrinsic,
            np.array(head_intrinsic['distortion_coefficients'])
        )

        T_chess_to_side, _ = get_chessboard_pose(
            img_side,
            scaled_side_intrinsic,
            np.array(side_intrinsic_data['distortion_coefficients'])
        )


        if T_chess_to_head is None or T_chess_to_side is None:
            print(f"[{index}] 棋盘角点提取失败")
            continue

        # 5. 固定头部相机 → base
        points_head_base = transform_to_base(head_extrinsic, T_chess_to_head, obj_points)
        print(f"points_head_base:\n{points_head_base[15]}")
        # 6. 左手相机通过FK → base
        joint_angles = df.iloc[i, 1:8].values.astype(float)
        T_base_to_end = get_link_pose_from_joints(
            side_joint_names, joint_angles, robot_id, joint_name_to_id, end_link_id
        )
        T_side_cam_to_base = T_base_to_end @ side_extrinsic

        points_side_base = transform_to_base(T_side_cam_to_base, T_chess_to_side, obj_points)
        print(f"points_side_base:\n{points_side_base[15]}")

        # 7. 误差计算（分别计算X、Y、Z轴的平均误差）
        # 计算每个点在三个轴上的差值（头部相机结果 - 左手相机结果）
        diff = points_head_base - points_side_base  # 形状为 (N, 3)，N为点的数量

        # 计算每个轴的平均绝对误差（或平均误差，根据需求选择）
        # 平均误差（可能受正负抵消影响，适合对称分布的误差）
        mean_error_x = diff[:, 0].mean()  # X轴平均误差
        mean_error_y = diff[:, 1].mean()  # Y轴平均误差
        mean_error_z = diff[:, 2].mean()  # Z轴平均误差

        # 平均绝对误差（更常用，避免正负抵消，反映实际偏差大小）
        mean_abs_error_x = np.abs(diff[:, 0]).mean()
        mean_abs_error_y = np.abs(diff[:, 1]).mean()
        mean_abs_error_z = np.abs(diff[:, 2]).mean()

        # 打印结果
        print(f"-----------[{index}] 各轴平均误差 (X, Y, Z):")
        print(f"  平均误差: X={mean_error_x:.4f} m, Y={mean_error_y:.4f} m, Z={mean_error_z:.4f} m")
        print(f"  平均绝对误差: X={mean_abs_error_x:.4f} m, Y={mean_abs_error_y:.4f} m, Z={mean_abs_error_z:.4f} m")
        mean_errors_x.append(mean_error_x)
        mean_errors_y.append(mean_error_y)
        mean_errors_z.append(mean_error_z)
        mean_abs_errors_x.append(mean_abs_error_x)
        mean_abs_errors_y.append(mean_abs_error_y)
        mean_abs_errors_z.append(mean_abs_error_z)


        # 7. 误差计算
        error = np.linalg.norm(points_head_base - points_side_base, axis=1).mean()
        print(f"-----------[{index}] 平均误差: {error:.4f} m")
        errors.append(error)
        valid_count += 1

    # 8. 总结
    if valid_count > 0:
        print(f"\n✅ 总共有效图像对数: {valid_count}")
        print(f"📊 平均误差: {np.mean(errors):.4f} m")
        print(f"  平均误差: X={np.mean(mean_errors_x):.4f} m, Y={np.mean(mean_errors_y):.4f} m, Z={np.mean(mean_errors_z):.4f} m")
        print(f"  平均绝对误差: X={np.mean(mean_abs_errors_x):.4f} m, Y={np.mean(mean_abs_errors_y):.4f} m, Z={np.mean(mean_abs_errors_z):.4f} m")
        # ✅ 构造 YAML 保存内容（英文键）
        result_data = {
            'valid_image_pairs': int(valid_count),
            'mean_error_m': float(np.mean(errors)),
            'mean_error': {
                'x_m': float(np.mean(mean_errors_x)),
                'y_m': float(np.mean(mean_errors_y)),
                'z_m': float(np.mean(mean_errors_z)),
            },
            'mean_absolute_error': {
                'x_m': float(np.mean(mean_abs_errors_x)),
                'y_m': float(np.mean(mean_abs_errors_y)),
                'z_m': float(np.mean(mean_abs_errors_z)),
            }
        }

        # ✅ 保存到 YAML 文件
        yaml_path = os.path.join(data_dir, error_yaml_name)
        os.makedirs(data_dir, exist_ok=True)  # 如果目录不存在则创建
        with open(yaml_path, 'w') as f:
            yaml.dump(result_data, f, sort_keys=False)

        print(f"\n📄 YAML 文件已保存至: {yaml_path}")
    else:
        print("❌ 没有有效图像对进行误差计算")