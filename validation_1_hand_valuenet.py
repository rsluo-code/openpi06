import dataclasses
import glob
import pathlib
import numpy as np
import cv2
import einops
import tqdm
import tyro
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd

from openpi_client import websocket_client_policy as _websocket_client_policy


import numpy as np
import torch
import torch.nn.functional as F

def _to_uint8_hwc(img):
    """
    统一把输入帧转成 uint8 的 HWC (RGB)。
    允许输入：
      - uint8 HWC
      - float HWC (0~1 或 0~255)
      - CHW
      - 灰度 HW
    """
    if img is None:
        raise ValueError("img is None")

    arr = np.asarray(img)

    # 灰度 HW -> HWC(3)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    # 单通道 -> 3 通道
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # 归一化/转换到 uint8
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        # 猜测范围：如果最大 <= 1.5 当作 0~1
        if arr_f.max() <= 1.5:
            arr_f = arr_f * 255.0
        arr_f = np.clip(arr_f, 0.0, 255.0)
        arr = arr_f.astype(np.uint8)

    return arr


def _resize_to_hw_torch(img_u8_hwc, H, W):
    """
    用 torch 插值把 uint8 HWC RGB resize 到 (H, W)，返回 uint8 HWC。
    """
    x = torch.from_numpy(img_u8_hwc).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,h,w]
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    x = x.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return x


def _load_panel_image(path: str, H: int, W: int) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load panel image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return _resize_to_hw_torch(img_rgb, H, W)


def _render_plot_panel(
    panel_spec: dict,
    step_idx: int,
    total_steps: int,
    H: int,
    W: int,
) -> np.ndarray:
    fig = Figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    title = panel_spec["title"]
    ylabel = panel_spec["ylabel"]
    xlabel = panel_spec.get("xlabel", "Timestep")
    series = panel_spec["series"]
    xlim = panel_spec.get("xlim", (0, max(total_steps - 1, 1)))
    ylim = panel_spec.get("ylim", None)
    secondary_series = panel_spec.get("secondary_series", [])
    secondary_ylabel = panel_spec.get("secondary_ylabel", None)
    secondary_ylim = panel_spec.get("secondary_ylim", None)

    for item in series:
        x = np.asarray(item["x"])
        y = np.asarray(item["y"])
        label = item["label"]
        color = item.get("color", None)
        if x.size == 0 or y.size == 0:
            continue

        ax.plot(x, y, linewidth=1.0, alpha=0.22, color=color)
        mask = x <= step_idx
        if np.any(mask):
            ax.plot(x[mask], y[mask], linewidth=1.5, label=label, color=color)

    ax2 = None
    secondary_handles = []
    secondary_labels = []
    if secondary_series:
        ax2 = ax.twinx()
        for item in secondary_series:
            x = np.asarray(item["x"])
            y = np.asarray(item["y"])
            label = item["label"]
            color = item.get("color", None)
            if x.size == 0 or y.size == 0:
                continue

            ax2.plot(x, y, linewidth=1.0, alpha=0.22, color=color)
            mask = x <= step_idx
            if np.any(mask):
                line = ax2.plot(x[mask], y[mask], linewidth=1.5, label=label, color=color)[0]
                secondary_handles.append(line)
                secondary_labels.append(label)

    ax.axvline(step_idx, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ax2 is not None:
        if secondary_ylabel is not None:
            ax2.set_ylabel(secondary_ylabel)
        if secondary_ylim is not None:
            ax2.set_ylim(*secondary_ylim)
        ax2.axvline(step_idx, color="red", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if secondary_handles:
        handles += secondary_handles
        labels += secondary_labels
    ax.legend(handles, labels, fontsize=7, loc="best")
    ax.text(
        0.01,
        0.98,
        f"t={step_idx}/{max(total_steps - 1, 0)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="red",
        fontsize=8,
    )
    fig.tight_layout()
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return _resize_to_hw_torch(rgba[..., :3], H, W)


def render_dashboard_video(
    head_frames,
    left_frames,
    right_frames,
    plot_specs: list[dict],
    out_mp4_path: str = "out.mp4",
    fps: int = 30,
    panel_height: int = 360,
    panel_width: int = 640,
):
    """
    生成 3 行 2 列拼接视频：
      第 1 行: head | out_png
      第 2 行: left | out_png_At
      第 3 行: right| out_png_It

    左列是逐帧视频，右列是逐帧动态生成的三张 plot 图。
    所有 panel 都会 resize 到 (panel_height, panel_width)。
    """
    L = min(len(head_frames), len(left_frames), len(right_frames))
    if L == 0:
        raise ValueError("Empty input sequences.")

    H = int(panel_height)
    W = int(panel_width)
    if H <= 0 or W <= 0:
        raise ValueError(f"panel_height and panel_width must be positive, got H={H}, W={W}")

    import subprocess

    out_h, out_w = 3 * H, 2 * W
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        for t in range(L):
            head = _resize_to_hw_torch(_to_uint8_hwc(head_frames[t]), H, W)
            left = _resize_to_hw_torch(_to_uint8_hwc(left_frames[t]), H, W)
            right = _resize_to_hw_torch(_to_uint8_hwc(right_frames[t]), H, W)
            plot_main_t = _render_plot_panel(plot_specs[0], t, L, H, W)
            plot_at_t = _render_plot_panel(plot_specs[1], t, L, H, W)
            plot_it_t = _render_plot_panel(plot_specs[2], t, L, H, W)

            row1 = np.concatenate([head, plot_main_t], axis=1)
            row2 = np.concatenate([left, plot_at_t], axis=1)
            row3 = np.concatenate([right, plot_it_t], axis=1)
            frame = np.concatenate([row1, row2, row3], axis=0)
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    return out_mp4_path



USE_LEFT = False
USE_RIGHT = True
DEFAULT_MODEL_TIME = "20260423"
DEFAULT_MODEL_STEP = 80000
DEFAULT_MODEL_DIM = "8dim"
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
    "sf包裹": "Pick and Place package to the right part",
}
from openpi.models_pytorch.some_func import get_index_and_max_len


# LEIBIE = "可乐"
# LEIBIE = "苹果"
# LEIBIE = "茶饮"
# LEIBIE = "香蕉"
# LEIBIE = "猕猴桃"
# LEIBIE = "矿泉水"
# LEIBIE = "纸巾"
# LEIBIE = "碗"
# LEIBIE = "盲盒"
# LEIBIE = "布绒玩具"
LEIBIE = "sf包裹"


_, max_len,_ = get_index_and_max_len(prompt_map[LEIBIE])
NAME_SAVE = f"valuenet_20260423_8wstep_8dim_0.002At_valbase_700maxlen_{LEIBIE}"
PATH_SAVE_BASE = f"/home/rsluo/codes/openpi06/z_valn_output/"

from openpi.models_pytorch.some_func import get_index_and_max_len

@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8001

    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd4/1001_06_香蕉_LD_香蕉_204/bg_tablecloth1:arm_left:object_1/episode_2025-10-01_09:33:32_657"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd1/1006_02_猕猴桃_LD_猕猴桃_298/bg_tablecloth1:arm_Left:object_1/episode_2025-10-05_16:01:22_596"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd4/1006_15_矿泉水_LD_农夫山泉_298/bg_tablecloth1:arm_left:object_1/episode_2025-10-06_14:49:43_015"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd2/1010_07_纸巾_LD_纸巾_199/bg_tablecloth1:arm_left:object_1/episode_2025-10-10_09:59:25_938"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd2/1024_04_碗_LD_塑料碗_80/bg_tablecloth1:arm_left:object_1/episode_2025-10-24_17:27:37_897"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd2/1009_02_盲盒_LD_盲盒_300/bg_tablecloth1:arm_Left:object_1/episode_2025-10-09_15:26:31_681"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd2/1022_08_布绒玩具_LD_卡皮巴拉_300/bg_tablecloth1:arm_left:object_1/episode_2025-10-22_13:53:45_594"
    episode_dir: str = "/data0/origin_datas_cut/sf_packages/data_10_31/1031_04_抓放包裹_LD_236/bg_tablecloth1arm_rightobject_1/episode_2025-10-31_111541_583_part_0_part_1"
    episode_glob: str | None = None

    # Output and naming
    output_base_dir: str = PATH_SAVE_BASE
    name_save: str | None = None
    model_time: str = DEFAULT_MODEL_TIME
    model_step: int = DEFAULT_MODEL_STEP
    model_dim: str = DEFAULT_MODEL_DIM
    prompt_type: str = LEIBIE
    video_panel_width: int = 640
    video_panel_height: int = 360

    # Value bins
    num_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0

    # Decode method: expectation over bins (recommended); if False, use argmax bin center
    use_expectation: bool = True

    # Task prompt (kept for compatibility with server)
    prompt: str = prompt_map[LEIBIE]
    use_left: bool =  USE_LEFT
    use_right: bool = USE_RIGHT
    image_keys: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    plot_state_index: int = 7
    custom_tail_direct_rt0: bool = False
    custom_tail_length: int = 30


def _sanitize_path_name(path_str: str) -> str:
    path = pathlib.Path(path_str)
    parts = [part for part in path.parts if part not in ("/", "")]
    return "__".join(parts[-3:]) if len(parts) >= 3 else "__".join(parts)


def _build_output_paths(base_dir: str, base_name: str, episode_dir: str) -> dict[str, str]:
    episode_tag = _sanitize_path_name(episode_dir)
    stem = f"{base_name}__{episode_tag}"
    return {
        "out_png": f"{base_dir}/{stem}_val.png",
        "out_png_At": f"{base_dir}/{stem}_At.png",
        "out_png_It": f"{base_dir}/{stem}_It.png",
        "out_mp4": f"{base_dir}/{stem}.mp4",
    }


def _sanitize_name_part(value: str) -> str:
    sanitized = str(value).strip()
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("", "_"),
    ]:
        sanitized = sanitized.replace(old, new)
    return sanitized


def _arm_mode_tag(use_left: bool, use_right: bool) -> str:
    if use_left and use_right:
        return "both"
    if use_left:
        return "left"
    if use_right:
        return "right"
    return "none"


def _resolve_prompt_type(args: Args) -> str:
    if getattr(args, "prompt_type", None):
        return _sanitize_name_part(args.prompt_type)

    for prompt_name, prompt_text in prompt_map.items():
        if prompt_text == args.prompt:
            return _sanitize_name_part(prompt_name)
    return _sanitize_name_part(args.prompt)


def _build_name_save(args: Args, max_len: int) -> str:
    if getattr(args, "name_save", None):
        return args.name_save
    arm_mode = _arm_mode_tag(args.use_left, args.use_right)
    prompt_type = _resolve_prompt_type(args)
    model_time = _sanitize_name_part(args.model_time)
    model_dim = _sanitize_name_part(args.model_dim)
    return (
        f"valuenet_{model_time}"
        f"_step{args.model_step}"
        f"_{arm_mode}"
        f"_{model_dim}"
        f"_maxlen{max_len}"
        f"_{prompt_type}"
    )


def _resolve_episode_dirs(args: Args) -> list[str]:
    if args.episode_glob:
        matches = sorted(path for path in glob.glob(args.episode_glob) if pathlib.Path(path).is_dir())
        if not matches:
            raise FileNotFoundError(f"No episode directories matched episode_glob={args.episode_glob}")
        return matches
    return [args.episode_dir]


def _resolve_plot_state_label(use_left: bool, use_right: bool, plot_state_index: int) -> str:
    state_idx_1based = plot_state_index + 1
    if plot_state_index == 7:
        if use_right and not use_left:
            return "Right gripper state"
        if use_left and not use_right:
            return "Left gripper state"
    if use_right and not use_left:
        return f"Right state[{state_idx_1based}]"
    if use_left and not use_right:
        return f"Left state[{state_idx_1based}]"
    return f"State[{state_idx_1based}]"


def _build_request_element(
    *,
    image_keys: tuple[str, ...],
    img,
    right_wrist_img,
    left_wrist_img,
    episode_first_head_img,
    state,
    prompt: str,
    step_index: int,
    episode_length: int,
    max_len: int,
) -> dict:
    element = {
        "observation/joint_position": state,
        "prompt": prompt,
        "step_index": step_index,
        "episode_length": episode_length,
        "language_instruction_index": 1,
        "language_instruction_max_len": max_len,
        "language_instruction_at_30precent": -0.3,
        "success_or_failure": 1,
    }
    if "base_0_rgb" in image_keys:
        element["observation/image"] = img
    if "left_wrist_0_rgb" in image_keys:
        element["observation/wrist_image_left"] = left_wrist_img
    if "right_wrist_0_rgb" in image_keys:
        element["observation/wrist_image_right"] = right_wrist_img
    if "episode_first_head_img" in image_keys:
        element["observation/episode_first_head_img"] = episode_first_head_img
    return element


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
    # cap = cv2.VideoCapture(str(video_path))
    # frames = []
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # cap.release()
    # return frames
    # 1. 解码
    size=(TARGET_IMG_W, TARGET_IMG_H)
    with av.open(video_path) as container:
        frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]

    # 2. 批量 resize
    resized_frames = [cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
                      for frame in frames]
    # 3. stack
    return np.stack(resized_frames).astype(np.uint8)


def _extract_logits(resp: dict) -> np.ndarray:
    """Return logits as a numpy array with shape [201] (or [1,201])."""
    for k in ["logits", "value_logits", "values", "actions"]:
        if k in resp:
            x = resp[k]
            x = np.asarray(x)
            # allow [T,201] or [201]
            if x.ndim == 2 and x.shape[-1] == 201:
                # if server returns a chunk, take the first step
                return x[0]
            if x.ndim == 1 and x.shape[0] == 201:
                return x
    raise KeyError(f"Cannot find 201-bin logits in response keys: {list(resp.keys())}")


def eval_isaac(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    episode_dirs = _resolve_episode_dirs(args)
    print(f"Resolved {len(episode_dirs)} episode(s) for visualization.")

    for episode_idx, episode_dir in enumerate(episode_dirs, start=1):
        print(f"\n===== [{episode_idx}/{len(episode_dirs)}] episode_dir={episode_dir} =====")
        _eval_single_episode(client, args, episode_dir)


def _eval_single_episode(client: _websocket_client_policy.WebsocketClientPolicy, args: Args, episode_dir: str) -> None:
    head_video_path = episode_dir + "/cam_head/cam_head.mp4"
    if args.use_left and args.use_right:
        left_video_path = episode_dir + "/cam_left/cam_left.mp4"
        right_video_path = episode_dir + "/cam_right/cam_right.mp4"
    elif args.use_left and args.use_right == False:
        left_video_path = episode_dir + "/cam_left/cam_left.mp4"
        right_video_path = episode_dir + "/cam_left/cam_left.mp4"
    elif args.use_left == False and args.use_right:
        left_video_path = episode_dir + "/cam_right/cam_right.mp4"
        right_video_path = episode_dir + "/cam_right/cam_right.mp4"
    else:
        raise RuntimeError("At least one arm must be enabled.")

    csv_path = episode_dir + "/robot_data.csv"

    idx, max_len,at_30precent = get_index_and_max_len(args.prompt)
    print("prompt idx=",idx)
    print("prompt max_len=",max_len)
    print("prompt at_30precent=",at_30precent)
    name_save = _build_name_save(args, max_len)
    output_paths = _build_output_paths(args.output_base_dir, name_save, episode_dir)
    print("name_save=", name_save)
    print("output_base_dir=", args.output_base_dir)

    # load states (right hand only)
    df = pd.read_csv(
        csv_path,
        header=0,
        index_col=False,
        usecols=lambda col: not col.startswith("Unnamed"),
    )
    # r_states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)
    # l_states = df[[f"l_s_{i}" for i in range(8)]].values.astype(np.float32)
    # l_actions = df[[f"l_a_{i}" for i in range(8)]].values.astype(np.float32)
    # r_actions = df[[f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
    # states = np.concatenate([l_states, r_states], axis=1)
    # actions = np.concatenate([l_actions, r_actions], axis=1)
    use_left = args.use_left
    use_right = args.use_right
    if(use_left == use_right):
        raise RuntimeError("only use one arm")
    if use_left:
        actions = df[[f"l_a_{i}" for i in range(8)]].values.astype(np.float32)
        states = df[[f"l_s_{i}" for i in range(8)]].values.astype(np.float32)
    if use_right:
        actions = df[ [f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
        states = df[ [f"r_s_{i}" for i in range(8)]].values.astype(np.float32)
    plot_state_index = int(args.plot_state_index)
    if plot_state_index < 0 or plot_state_index >= states.shape[1]:
        raise ValueError(
            f"plot_state_index={plot_state_index} out of range for states.shape[1]={states.shape[1]}"
        )


    # load frames
    head_frames = _load_video_frames(head_video_path)
    right_frames = _load_video_frames(right_video_path)
    left_frames = _load_video_frames(left_video_path)

    # episode length (use the aligned length)
    L = min(states.shape[0], len(head_frames), len(right_frames))
    if L <= 0:
        raise RuntimeError("No aligned frames/states to process (L<=0).")


    # bin centers in [-1, 0]
    bin_centers = torch.linspace(args.v_min, args.v_max, args.num_bins, dtype=torch.float32)

    Rt0_list = []
    Rt1_list = []
    Rt2_list = []
    gripper_state_list = []

    print(f"Episode length (aligned) = {L}")
    request_image_keys = tuple(args.image_keys)
    tail_length = max(0, int(args.custom_tail_length))
    episode_first_head_img = None
    if "episode_first_head_img" in request_image_keys:
        episode_first_head_img = einops.rearrange(head_frames[0], "h w c -> c h w")
    for i in tqdm.tqdm(range(L)):
        img = head_frames[i]
        right_wrist_img = right_frames[i]
        left_wrist_img = left_frames[i]
        img = einops.rearrange(img, "h w c -> c h w")
        right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
        left_wrist_img = einops.rearrange(left_wrist_img, "h w c -> c h w")

        # target Rt0: current remaining trajectory length / task max length.
        # The first frame is -L/max_len, so it is not necessarily -1.
        ratio = (L - i) / max(1, max_len)
        ratio =  max(0.0, min(1.0, ratio))
        Rt0 = -1*ratio
        gripper_state = float(states[i, plot_state_index])

        is_tail_window = tail_length > 0 and i >= max(0, L - tail_length)
        if args.custom_tail_direct_rt0 and is_tail_window:
            Rt1 = Rt0
            Rt2 = Rt0
            print(
                f"[tail override] step={i} within last {tail_length} frames, "
                "skip infer and set Rt1=Rt0, Rt2=Rt0"
            )
        else:
            element = _build_request_element(
                image_keys=request_image_keys,
                img=img,
                right_wrist_img=right_wrist_img,
                left_wrist_img=left_wrist_img,
                episode_first_head_img=episode_first_head_img,
                state=states[i],
                prompt=str(args.prompt),
                step_index=i,
                episode_length=L,
                max_len=max_len,
            )

            resp = client.infer(element)

            logits_np = _extract_logits(resp)  # [201]
            logits = torch.as_tensor(logits_np, dtype=torch.float32)  # [201]
            probs = torch.softmax(logits, dim=-1)  # [201]

            Rt1 = float((probs * bin_centers).sum().item())
            Rt2 = float(bin_centers[int(torch.argmax(probs).item())].item())

        Rt0_list.append(Rt0)
        Rt1_list.append(Rt1)
        Rt2_list.append(Rt2)
        gripper_state_list.append(gripper_state)
        print("Rt0: ",Rt0," Rt1: ",Rt1," Rt2: ",Rt2)
        # import pdb; pdb.set_trace()


    # plot
    timesteps = np.arange(L)
    gripper_label = _resolve_plot_state_label(use_left, use_right, plot_state_index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timesteps, Rt0_list, linewidth=1.0, label="Rt0 (target from step_index)")
    ax.plot(timesteps, Rt1_list, linewidth=1.0, label="Rt1 (model prediction mean)")
    ax.plot(timesteps, Rt2_list, linewidth=1.0, label="Rt2 (model prediction mode)")
    ax.set_title("ValueNet: target Rt0 vs predicted Rt1 Rt2")
    ax.set_xlabel("Timestep (step_index)")
    ax.set_ylabel("R(t) in [-1, 0]")
    ax2 = ax.twinx()
    ax2.plot(timesteps, gripper_state_list, linewidth=1.0, color="C3", label=gripper_label)
    ax2.set_ylabel(gripper_label)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(output_paths["out_png"], dpi=200)
    plt.close()
    print(f"Saved plot to: {output_paths['out_png']}")


    # plot At
    chunk_size = 30
    if L <= chunk_size:
        raise RuntimeError(f"Episode length L={L} must be greater than chunk_size={chunk_size} to compute At.")
    timesteps_cut = np.arange(L - chunk_size)
    task_max_len = max(1, max_len)
    RtN_by_task_max_len = -chunk_size / task_max_len
    At1 = [rt0 - rt1 for rt0, rt1 in zip(Rt0_list, Rt1_list)]
    At2 = [
        Rt1_list[i + chunk_size] - Rt1_list[i] + RtN_by_task_max_len
        for i in range(len(Rt1_list) - chunk_size)
    ]
    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, At1, linewidth=1.0, label="At1(length L) = Rt0(t) - Rt1(t)")
    plt.plot(timesteps_cut, At2, linewidth=1.0, label="At2(length L-N) = Rt1(t+n) - Rt1(t) - n/max_len")
    plt.title("ValueNet: At1(length L) vs At2(length L-N)")
    plt.xlabel("Timestep (step_index)")
    plt.ylabel("A(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_paths["out_png_At"], dpi=200)
    plt.close()
    print(f"Saved plot to: {output_paths['out_png_At']}")
    It1 = [a > at_30precent for a in At1]
    It2 = [a > at_30precent for a in At2]
    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, It1, linewidth=1.0, label="It1 = At1>At1_30precent")
    plt.plot(timesteps_cut, It2, linewidth=1.0, label="It2 = At2>At1_30precent ")
    plt.title("ValueNet: It1  vs  It2")
    plt.xlabel("Timestep (step_index)")
    plt.ylabel("R(t) in [-1, 0]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_paths["out_png_It"], dpi=200)
    plt.close()
    print(f"Saved plot to: {output_paths['out_png_It']}")


    plot_specs = [
        {
            "title": "ValueNet: target Rt0 vs predicted Rt1 Rt2",
            "xlabel": "Timestep (step_index)",
            "ylabel": "R(t) in [-1, 0]",
            "xlim": (0, max(L - 1, 1)),
            "ylim": (min(min(Rt0_list), min(Rt1_list), min(Rt2_list)) - 0.05, max(max(Rt0_list), max(Rt1_list), max(Rt2_list)) + 0.05),
            "series": [
                {"x": timesteps, "y": Rt0_list, "label": "Rt0 (target from step_index)", "color": "C0"},
                {"x": timesteps, "y": Rt1_list, "label": "Rt1 (model prediction mean)", "color": "C1"},
                {"x": timesteps, "y": Rt2_list, "label": "Rt2 (model prediction mode)", "color": "C2"},
            ],
            "secondary_ylabel": gripper_label,
            "secondary_ylim": (
                min(gripper_state_list) - 0.05 * max(1e-6, max(gripper_state_list) - min(gripper_state_list) or 1.0),
                max(gripper_state_list) + 0.05 * max(1e-6, max(gripper_state_list) - min(gripper_state_list) or 1.0),
            ),
            "secondary_series": [
                {"x": timesteps, "y": gripper_state_list, "label": gripper_label, "color": "C3"},
            ],
        },
        {
            "title": "ValueNet: At1(length L) vs At2(length L-N)",
            "xlabel": "Timestep (step_index)",
            "ylabel": "A(t)",
            "xlim": (0, max(L - 1, 1)),
            "ylim": (min(min(At1), min(At2)) - 0.05, max(max(At1), max(At2)) + 0.05),
            "series": [
                {"x": timesteps, "y": At1, "label": "At1(length L) = Rt0(t) - Rt1(t)", "color": "C0"},
                {"x": timesteps_cut, "y": At2, "label": "At2(length L-N) = Rt1(t+n) - Rt1(t) - n/max_len", "color": "C1"},
            ],
        },
        {
            "title": "ValueNet: It1 vs It2",
            "xlabel": "Timestep (step_index)",
            "ylabel": "Indicator",
            "xlim": (0, max(L - 1, 1)),
            "ylim": (-0.1, 1.1),
            "series": [
                {"x": timesteps, "y": np.asarray(It1, dtype=np.float32), "label": "It1 = At1>At1_30precent", "color": "C0"},
                {"x": timesteps_cut, "y": np.asarray(It2, dtype=np.float32), "label": "It2 = At2>At1_30precent", "color": "C1"},
            ],
        },
    ]

    # quick stats
    Rt0_arr = np.asarray(Rt0_list)
    Rt1_arr = np.asarray(Rt1_list)
    mae = np.mean(np.abs(Rt1_arr - Rt0_arr))
    
    print(f"Rt0 range: [{Rt0_arr.min():.4f}, {Rt0_arr.max():.4f}]  Rt1 range: [{Rt1_arr.min():.4f}, {Rt1_arr.max():.4f}]")
    print(f"MAE(|Rt1-Rt0|): {mae:.6f}")
    out_path = render_dashboard_video(
        head_frames=head_frames,
        left_frames=left_frames,
        right_frames=right_frames,
        plot_specs=plot_specs,
        out_mp4_path=output_paths["out_mp4"],
        fps=30,
        panel_width=args.video_panel_width,
        panel_height=args.video_panel_height,
    )
    print("mp4 saved to:", out_path)

if __name__ == "__main__":
    tyro.cli(eval_isaac)
