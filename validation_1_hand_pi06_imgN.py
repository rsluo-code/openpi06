import collections
import dataclasses
import pathlib

import av
import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import tyro

from openpi.models_pytorch.some_func import get_index_and_max_len
from openpi_client import websocket_client_policy as _websocket_client_policy


prompt_map = {
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


ORIGIN_IMG_H = 720
ORIGIN_IMG_W = 1280
TARGET_IMG_MAX = 224
if ORIGIN_IMG_H >= ORIGIN_IMG_W:
    TARGET_IMG_H = TARGET_IMG_MAX
    _scale = TARGET_IMG_H / ORIGIN_IMG_H
    TARGET_IMG_W = int(ORIGIN_IMG_W * _scale)
else:
    TARGET_IMG_W = TARGET_IMG_MAX
    _scale = TARGET_IMG_W / ORIGIN_IMG_W
    TARGET_IMG_H = int(ORIGIN_IMG_H * _scale)


def _default_prompt() -> str:
    return prompt_map["sf包裹"]


def _default_max_len() -> int:
    _, max_len, _ = get_index_and_max_len(_default_prompt())
    return max_len


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8001
    replan_steps: int = 30
    use_left: bool = False
    use_right: bool = True
    prompt_type: str = "sf包裹"
    prompt: str = _default_prompt()
    num_trials_per_task: int = 1
    episode_dir: str = (
        "/data0/origin_datas_cut/sf_packages/data_10_31/1031_04_抓放包裹_LD_236/"
        "bg_tablecloth1arm_rightobject_1/episode_2025-10-31_111541_583_part_0_part_1"
    )
    output_base_dir: str = "/home/rsluo/codes/openpi06/z_pi06_output"
    name_save: str = ""
    model_time: str = "20260506"
    model_step: str = "100000"
    model_dim: str = "8dim"
    language_instruction_index: int = 1
    language_instruction_at_30precent: float = -0.0
    success_or_failure: int = 1


def _load_video_frames(video_path: str) -> np.ndarray:
    size = (TARGET_IMG_W, TARGET_IMG_H)
    with av.open(video_path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    resized_frames = [cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) for frame in frames]
    return np.stack(resized_frames).astype(np.uint8)


def _resolve_output_name(args: Args, episode_dir: str) -> str:
    if args.name_save:
        return args.name_save
    arm_tag = "both" if args.use_left and args.use_right else ("left" if args.use_left else "right")
    episode_name = pathlib.Path(episode_dir).name
    return f"pi06_{args.model_time}_{args.model_step}_{args.model_dim}_{args.prompt_type}_{arm_tag}_{episode_name}"


def _resolve_prompt(args: Args) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_type not in prompt_map:
        raise KeyError(f"Unknown prompt_type={args.prompt_type}. Available prompt_types: {list(prompt_map.keys())}")
    return prompt_map[args.prompt_type]


def _resolve_max_len(prompt: str) -> int:
    _, max_len, _ = get_index_and_max_len(prompt)
    return max_len


def _load_episode_arrays(args: Args, episode_dir: str):
    episode_path = pathlib.Path(episode_dir)
    head_video_path = episode_path / "cam_head" / "cam_head.mp4"
    if args.use_left and args.use_right:
        left_video_path = episode_path / "cam_left" / "cam_left.mp4"
        right_video_path = episode_path / "cam_right" / "cam_right.mp4"
    elif args.use_left and not args.use_right:
        left_video_path = episode_path / "cam_left" / "cam_left.mp4"
        right_video_path = episode_path / "cam_left" / "cam_left.mp4"
    elif (not args.use_left) and args.use_right:
        left_video_path = episode_path / "cam_right" / "cam_right.mp4"
        right_video_path = episode_path / "cam_right" / "cam_right.mp4"
    else:
        raise RuntimeError("At least one of use_left/use_right must be true")

    csv_path = episode_path / "robot_data.csv"
    df = pd.read_csv(
        csv_path,
        header=0,
        index_col=False,
        usecols=lambda col: not col.startswith("Unnamed"),
    )
    if args.use_left == args.use_right:
        raise RuntimeError("only use one arm")
    if args.use_left:
        actions = df[[f"l_a_{i}" for i in range(8)]].values.astype(np.float32)
        states = df[[f"l_s_{i}" for i in range(8)]].values.astype(np.float32)
    else:
        actions = df[[f"r_a_{i}" for i in range(8)]].values.astype(np.float32)
        states = df[[f"r_s_{i}" for i in range(8)]].values.astype(np.float32)

    head_frames = _load_video_frames(str(head_video_path))
    right_frames = _load_video_frames(str(right_video_path))
    left_frames = _load_video_frames(str(left_video_path))
    return actions, states, head_frames, right_frames, left_frames


def _plot_actions(output_png: pathlib.Path, predicted: list[list[float]], labels: list[list[float]]) -> None:
    predicted_arrays = [np.asarray(values, dtype=np.float32) for values in predicted]
    label_arrays = [np.asarray(values, dtype=np.float32) for values in labels]
    timesteps = np.arange(len(predicted_arrays[0]))

    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Joint Angles", fontsize=8, y=1.02)

    for idx, (ax, pred_values, label_values) in enumerate(zip(axes.flatten(), predicted_arrays, label_arrays), start=1):
        ax.plot(timesteps, pred_values, linewidth=0.2, marker="o", markersize=1, label=f"Joint {idx}")
        ax.plot(timesteps, label_values, color="red", linewidth=0.2, marker="o", markersize=1, label=f"Label {idx}")
        ax.set_title(f"Joint {idx}", fontsize=10)
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel("Angle (rad)", fontsize=8)
        ax.legend(fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)


def _eval_single_episode(
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    episode_dir: str,
) -> pathlib.Path:
    task_description = _resolve_prompt(args)
    max_len = _resolve_max_len(task_description)
    output_name = _resolve_output_name(args, episode_dir)
    output_png = pathlib.Path(args.output_base_dir) / f"{output_name}.png"

    actions, states, head_frames, right_frames, left_frames = _load_episode_arrays(args, episode_dir)
    action_n = 30
    episode_length = min(states.shape[0], len(head_frames), len(right_frames))
    loop_length = min(states.shape[0] - action_n, episode_length - action_n)
    if loop_length <= 0:
        raise ValueError(f"Episode too short for action_n={action_n}: {episode_dir}")

    predicted = [[] for _ in range(8)]
    labels = [[] for _ in range(8)]
    err_list: list[float] = []

    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        print(f"\nTask: {task_description}")
        print(f"Starting episode {episode_idx + 1}...")
        action_plan = collections.deque()

        for i in range(loop_length):
            img = einops.rearrange(head_frames[i], "h w c -> c h w")
            right_wrist_img = einops.rearrange(right_frames[i], "h w c -> c h w")
            left_wrist_img = einops.rearrange(left_frames[i], "h w c -> c h w")

            img_n = einops.rearrange(head_frames[i + action_n], "h w c -> c h w")
            right_wrist_img_n = einops.rearrange(right_frames[i + action_n], "h w c -> c h w")
            left_wrist_img_n = einops.rearrange(left_frames[i + action_n], "h w c -> c h w")

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image_right": right_wrist_img,
                    "observation/wrist_image_left": left_wrist_img,
                    "observation/joint_position": states[i],
                    "observation/image_N": img_n,
                    "observation/wrist_image_right_N": right_wrist_img_n,
                    "observation/wrist_image_left_N": left_wrist_img_n,
                    "observation/joint_position_N": states[i + action_n],
                    "prompt": str(task_description),
                    "step_index": i,
                    "episode_length": episode_length,
                    "language_instruction_index": args.language_instruction_index,
                    "language_instruction_max_len": max_len,
                    "language_instruction_at_30precent": args.language_instruction_at_30precent,
                    "success_or_failure": args.success_or_failure,
                }
                action_chunk = client.infer(element)["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft().tolist()
            action_error = float(np.mean(np.abs(np.asarray(action[:8], dtype=np.float32) - actions[i][:8])))
            err_list.append(action_error)

            for joint_idx in range(8):
                predicted[joint_idx].append(action[joint_idx])
                labels[joint_idx].append(float(actions[i][joint_idx]))

    print(np.mean(err_list))
    _plot_actions(output_png, predicted, labels)
    print(f"Saved plot to {output_png}")
    return output_png


def eval_isaac(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    _eval_single_episode(client, args, args.episode_dir)


if __name__ == "__main__":
    eval_isaac(tyro.cli(Args))
