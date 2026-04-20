import dataclasses
import numpy as np
import cv2
import einops
import tqdm
import tyro
import torch
import matplotlib.pyplot as plt
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


def render_quadrant_video(Rt0_list, Rt1_list,  Rt2_list, head_frames, right_frames, left_frames, out_mp4_path="out.mp4", fps=30,  plot_ylim=(-1.05, 0.05), plot_title="Rt curves",
):
    """
    生成 2x2 拼接视频：
      TL: left
      TR: right
      BL: head
      BR: 曲线图(随时间增长)

    参数：
      Rt0_list, Rt1_list, Rt2_list: 长度 L 的数值列表/np数组
      head_frames/right_frames/left_frames: 长度 L 的图像序列（每帧 HWC/CHW/灰度均可）
      out_mp4_path: 输出 mp4
      fps: 帧率
      plot_ylim: 右下角图 y 轴范围
    """
    # --------- 基本检查 ---------
    L = min(len(Rt0_list), len(Rt1_list), len(Rt2_list),
            len(head_frames), len(right_frames), len(left_frames))
    if L == 0:
        raise ValueError("Empty input sequences.")

    # 转成 numpy 方便切片
    Rt0 = np.asarray(Rt0_list, dtype=np.float32)[:L]
    Rt1 = np.asarray(Rt1_list, dtype=np.float32)[:L]
    Rt2 = np.asarray(Rt2_list, dtype=np.float32)[:L]

    # 以 head 的尺寸作为目标尺寸
    head0 = _to_uint8_hwc(head_frames[0])
    H, W = head0.shape[0], head0.shape[1]

    # --------- 准备 matplotlib 画布（只创建一次，逐帧更新线）---------
    import matplotlib
    matplotlib.use("Agg")  # 后台渲染，避免弹窗
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = plt.Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title(plot_title)
    ax.set_xlabel("t")
    ax.set_ylabel("R")
    ax.set_ylim(plot_ylim[0], plot_ylim[1])
    ax.grid(True, alpha=0.3)

    # 预设 x 轴范围
    ax.set_xlim(0, max(1, L - 1))

    # 三条线对象，后续 set_data 更新
    line0, = ax.plot([], [], linewidth=2.0, label="Rt0 (target)")
    line1, = ax.plot([], [], linewidth=2.0, label="Rt1 (pred mean)")
    line2, = ax.plot([], [], linewidth=2.0, label="Rt2 (pred mode)")
    ax.legend(loc="lower right")

    # --------- 启动 ffmpeg 写视频（raw rgb pipe）---------
    import subprocess

    out_h, out_w = 2 * H, 2 * W  # 2x2 拼接
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
            # ---------- 读取并对齐三路图像到 head 的 HxW ----------
            head = _to_uint8_hwc(head_frames[t])
            left = _to_uint8_hwc(left_frames[t])
            right = _to_uint8_hwc(right_frames[t])

            if head.shape[0] != H or head.shape[1] != W:
                head = _resize_to_hw_torch(head, H, W)
            if left.shape[0] != H or left.shape[1] != W:
                left = _resize_to_hw_torch(left, H, W)
            if right.shape[0] != H or right.shape[1] != W:
                right = _resize_to_hw_torch(right, H, W)

            # ---------- 更新右下角曲线（只画到当前 t）----------
            xs = np.arange(t + 1, dtype=np.float32)
            line0.set_data(xs, Rt0[:t + 1])
            line1.set_data(xs, Rt1[:t + 1])
            line2.set_data(xs, Rt2[:t + 1])

            # 可选：让 x 轴跟随（如果你想滚动窗口）
            # ax.set_xlim(max(0, t-200), max(1, t))

            canvas.draw()


            # 兼容不同 matplotlib 版本：优先用 buffer_rgba（最稳定）
            w, h = canvas.get_width_height()

            try:
                # buffer_rgba() 返回 RGBA 的 buffer（h, w, 4）
                buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
                plot_rgba = buf.reshape((h, w, 4))
                plot_rgb = plot_rgba[:, :, :3]  # 丢掉 alpha
            except Exception:
                # 兜底：用 tostring_argb 再转成 rgb
                buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
                argb = buf.reshape((h, w, 4))           # A,R,G,B
                plot_rgb = argb[:, :, 1:4]              # R,G,B

            # 把 plot resize 到 HxW（对齐其它象限）
            plot_rgb = _resize_to_hw_torch(plot_rgb, H, W)

            # ---------- 拼 2x2 画面 ----------
            # TL=left, TR=right
            top = np.concatenate([left, right], axis=1)     # [H, 2W, 3]
            # BL=head, BR=plot
            bottom = np.concatenate([head, plot_rgb], axis=1)  # [H, 2W, 3]
            frame = np.concatenate([top, bottom], axis=0)   # [2H, 2W, 3]

            # 写入 ffmpeg
            proc.stdin.write(frame.tobytes())

    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    return out_mp4_path


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
NAME_SAVE = f"valuenet_episode_plot9_20260116model_2b_31w5step_exchangemodel_noexchangeimg_{LEIBIE}"
PATH_SAVE_BASE = f"/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/z_experimental_result/"
@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8001

    # Episode data
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd1/1003_02_可乐_LD_可口可乐_178/bg_tablecloth1:arm_left:object_1/episode_2025-10-03_15:00:15_930"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd4/1014_07_苹果_LD_红苹果_300/bg_tablecloth1:arm_left:object_1/episode_2025-10-14_09:36:39_110"
    episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd3/1009_04_茶饮_LD_冰红茶_197/bg_tablecloth1:arm_right:object_1/episode_2025-10-09_10:46:47_583"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd1/0930_02_香蕉_LD_香蕉_212/bg_tablecloth1:arm_left:object_1/episode_2025-09-30_12:57:17_047"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LD_抓物品_all_0930_1114/zd1/1001_04_可乐_LD_可口可乐_199/bg_tablecloth1:arm_left:object_1/episode_2025-10-01_11:19:26_659"

    # episode_dir: str =  "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LDT_抓物品_测试集/抓空数据/0122/苹果_抓空/no_oak_date/episode_2026-01-22_113642_898"
    # episode_dir: str = "/wx-mix01/sppro/permanent/yuanzhang10/资源部传输数据/抓物品/V4新规则抓取/LDT_抓物品_测试集/抓空数据/0122/可乐_抓空/no_oak_date/episode_2026-01-22_113923_964"
    
    # Plot output
    out_png: str = f"{PATH_SAVE_BASE}/{NAME_SAVE}.png"

    # Value bins
    num_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0

    # Decode method: expectation over bins (recommended); if False, use argmax bin center
    use_expectation: bool = True

    # Task prompt (kept for compatibility with server)
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

    episode_dir = args.episode_dir
    head_video_path = episode_dir + "/cam_head/cam_head.mp4"
    left_video_path = episode_dir + "/cam_left/cam_left.mp4"
    right_video_path = episode_dir + "/cam_right/cam_right.mp4"
    csv_path = episode_dir + "/robot_data.csv"

    # load states (right hand only)
    df = pd.read_csv(
        csv_path,
        header=0,
        index_col=False,
        usecols=lambda col: not col.startswith("Unnamed"),
    )
    r_states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)
    l_states = df[[f"l_s_{i}" for i in range(8)]].values.astype(np.float32)
    l_actions = df[[f"l_a_{i}" for i in range(8)]].values.astype(np.float32)
    r_actions = df[[f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
    states = np.concatenate([l_states, r_states], axis=1)
    actions = np.concatenate([l_actions, r_actions], axis=1)

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

    print(f"Episode length (aligned) = {L}")
    for i in tqdm.tqdm(range(L)):
        img = head_frames[i]
        right_wrist_img = right_frames[i]
        left_wrist_img = left_frames[i]
        img = einops.rearrange(img, "h w c -> c h w")
        right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
        left_wrist_img = einops.rearrange(left_wrist_img, "h w c -> c h w")


        element = {
            "observation/image": img,
            "observation/wrist_image_right": right_wrist_img,
            "observation/wrist_image_left": left_wrist_img,
            "observation/joint_position": states[i],


            # "observation/joint_position_N": states[i],
            "prompt": str(args.prompt),
            "step_index": i,
            "episode_length": L,
            # "actions":actions[i],
            "language_instruction_index":1,
            "language_instruction_max_len":L+1,
            "language_instruction_at_30precent":-0.3,
            "success_or_failure":1,
        }

        resp = client.infer(element)
        # import pdb; pdb.set_trace()

        logits_np = _extract_logits(resp)  # [201]
        logits = torch.as_tensor(logits_np, dtype=torch.float32)  # [201]
        probs = torch.softmax(logits, dim=-1)  # [201]
        # print(f"{probs.sum(dim=-1).detach().cpu()}")

        Rt1 = float((probs * bin_centers).sum().item())
        Rt2 = float(bin_centers[int(torch.argmax(probs).item())].item())
        max_len=L
        # target Rt0
        ratio = (L-i)/max_len
        ratio =  max(0.0, min(1.0, ratio))
        Rt0 = -1*ratio
        Rt0_list.append(Rt0)
        Rt1_list.append(Rt1)
        Rt2_list.append(Rt2)
        print("Rt0: ",Rt0," Rt1: ",Rt1," Rt2: ",Rt2)
        # import pdb; pdb.set_trace()


    # plot
    timesteps = np.arange(L)
    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, Rt0_list, linewidth=1.0, label="Rt0 (target from step_index)")
    plt.plot(timesteps, Rt1_list, linewidth=1.0, label="Rt1 (model prediction mean)")
    plt.plot(timesteps, Rt2_list, linewidth=1.0, label="Rt2 (model prediction mode)")
    plt.title("ValueNet: target Rt0 vs predicted Rt1 Rt2")
    plt.xlabel("Timestep (step_index)")
    plt.ylabel("R(t) in [-1, 0]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.show()

    # quick stats
    Rt0_arr = np.asarray(Rt0_list)
    Rt1_arr = np.asarray(Rt1_list)
    mae = np.mean(np.abs(Rt1_arr - Rt0_arr))
    print(f"Saved plot to: {args.out_png}")
    print(f"Rt0 range: [{Rt0_arr.min():.4f}, {Rt0_arr.max():.4f}]  Rt1 range: [{Rt1_arr.min():.4f}, {Rt1_arr.max():.4f}]")
    print(f"MAE(|Rt1-Rt0|): {mae:.6f}")
    out_path = render_quadrant_video(
        Rt0_list=Rt0_list,
        Rt1_list=Rt1_list,
        Rt2_list=Rt2_list,
        head_frames=head_frames,
        right_frames=right_frames,
        left_frames=left_frames,
        out_mp4_path=f"{PATH_SAVE_BASE}/{NAME_SAVE}.mp4",
        fps=30,
    )
    print("mp4 saved to:", out_path)

if __name__ == "__main__":
    tyro.cli(eval_isaac)
