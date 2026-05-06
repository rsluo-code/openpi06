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


"""
ValueNet validation script.

- Reads one recorded episode (head cam, right wrist cam, robot_data.csv right-hand joint states).
- For each frame i:
    Rt0 = a + i/(L-1) * (b-a)   with a=-1, b=0, L=episode_length
    Rt1 is predicted by the model as 201-bin logits -> probs -> continuous value in [-1, 0]
- Plots Rt0 (target) and Rt1 (prediction) over timesteps.

Notes about server response:
- This script tries these keys (in order) to fetch the 201-bin logits:
    "logits", "value_logits", "values", "actions"
  (Some deployments may reuse the "actions" key; we treat it as logits if its last dim is 201.)
"""


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8001

    # Episode data
    episode_dir: str = "/data0/origin_datas_cut/sf_packages/data_10_31/1031_04_抓放包裹_LD_236/bg_tablecloth1arm_rightobject_1/episode_2025-10-31_111541_583_part_0_part_1"

    # Plot output
    out_png: str = "/home/rsluo/codes/openpi06/valuenet_episode_plot2.png"

    # Value bins
    num_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0

    # Decode method: expectation over bins (recommended); if False, use argmax bin center
    use_expectation: bool = True

    # Task prompt (kept for compatibility with server)
    prompt: str = "Pick and Place package to the right part"


def _load_video_frames(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


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
    right_video_path = episode_dir + "/cam_right/cam_right.mp4"
    csv_path = episode_dir + "/robot_data.csv"

    # load states (right hand only)
    df = pd.read_csv(
        csv_path,
        header=0,
        index_col=False,
        usecols=lambda col: not col.startswith("Unnamed"),
    )
    states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)

    # load frames
    head_frames = _load_video_frames(head_video_path)
    right_frames = _load_video_frames(right_video_path)

    # episode length (use the aligned length)
    L = min(states.shape[0], len(head_frames), len(right_frames))
    if L <= 0:
        raise RuntimeError("No aligned frames/states to process (L<=0).")

    denom = max(L - 1, 1)

    # bin centers in [-1, 0]
    bin_centers = torch.linspace(args.v_min, args.v_max, args.num_bins, dtype=torch.float32)

    Rt0_list = []
    Rt1_list = []

    print(f"Episode length (aligned) = {L}")
    episode_first_head_img = einops.rearrange(head_frames[0], "h w c -> c h w")
    for i in tqdm.tqdm(range(L)):
        img = head_frames[i]
        right_wrist_img = right_frames[i]

        # preprocess image
        img = einops.rearrange(img, "h w c -> c h w")
        right_wrist_img = einops.rearrange(right_wrist_img, "h w c -> c h w")
        left_wrist_img = np.zeros_like(right_wrist_img)

        element = {
            "observation/image": img,
            "observation/episode_first_head_img": episode_first_head_img,
            "observation/wrist_image_right": right_wrist_img,
            "observation/wrist_image_left": left_wrist_img,
            "observation/joint_position": states[i],
            "prompt": str(args.prompt),
            "step_index": i,
            "episode_length": L,
        }
        print(element)
        # model inference
        resp = client.infer(element)
        print(resp)

        logits_np = _extract_logits(resp)  # [201]

        logits = torch.as_tensor(logits_np, dtype=torch.float32)  # [201]
        probs = torch.softmax(logits, dim=-1)  # [201]

        if args.use_expectation:
            Rt1 = float((probs * bin_centers).sum().item())
        else:
            Rt1 = float(bin_centers[int(torch.argmax(probs).item())].item())

        # target Rt0
        a, b = -1.0, 0.0
        Rt0 = a + (i / denom) * (b - a)

        Rt0_list.append(Rt0)
        Rt1_list.append(Rt1)
        print("Rt0: ",Rt0," Rt1: ",Rt1)

    # plot
    timesteps = np.arange(L)
    plt.figure(figsize=(12, 4))
    plt.plot(timesteps, Rt0_list, linewidth=1.0, label="Rt0 (target from step_index)")
    plt.plot(timesteps, Rt1_list, linewidth=1.0, label="Rt1 (model prediction)")
    plt.title("ValueNet: target Rt0 vs predicted Rt1")
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


if __name__ == "__main__":
    tyro.cli(eval_isaac)
