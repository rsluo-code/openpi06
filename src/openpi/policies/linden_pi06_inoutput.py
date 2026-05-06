import dataclasses

import einops
import numpy as np
import random
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LindenInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    use_left:bool = False
    use_right:bool = False
    train_or_infer:str = "infer"
    image_keys: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    def __call__(self, data: dict) -> dict:
        if self.use_left == False and self.use_right == False:
            raise RuntimeError(f"LindenInputs use_left and use_right must one be True")
        # import pdb; pdb.set_trace()
        

        # {
        #     actions
        #     observation
        #     prompt
        #     step_index
        #     episode_length
        # }


        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference

        state = np.asarray(data["observation/joint_position"])

        base_image = _parse_image(data["observation/image"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])
        episode_first_head_img = None
        if "episode_first_head_img" in self.image_keys:
            episode_first_head_img = _parse_image(data["observation/episode_first_head_img"])

        state_N = np.asarray(data["observation/joint_position_N"])
        base_image_N = _parse_image(data["observation/image_N"])
        wrist_image_left_N = _parse_image(data["observation/wrist_image_left_N"])
        wrist_image_right_N = _parse_image(data["observation/wrist_image_right_N"])
        episode_first_head_img_N = None
        if "episode_first_head_img" in self.image_keys:
            episode_first_head_img_N = _parse_image(data["observation/episode_first_head_img_N"])

        # print(f"shape base_image = {base_image.shape}")
        # print(f"shape wrist_image_left = {wrist_image_left.shape}")
        # print(f"shape wrist_image_right = {wrist_image_right.shape}")
        # print(f"shape base_image_N = {base_image_N.shape}")
        # print(f"shape wrist_image_left_N = {wrist_image_left_N.shape}")
        # print(f"shape wrist_image_right_N = {wrist_image_right_N.shape}")


        # 全局统一判断：50%概率触发所有处理（要么都处理，要么都不处理）
        should_process=False
        if(self.train_or_infer == "train"):
            should_process = random.random() < 0.5
        # print(f"self.train_or_infer = {self.train_or_infer}")
        should_process=False

        # 定义通用处理函数（支持state/actions的动态维度，由全局变量控制是否处理）
        def process_array(arr):
            """对数组进行：指定列取负 + 前后半部分交换（全局统一触发）"""
            if should_process:  # 使用全局布尔变量，不再单独判断概率
                d = arr.shape[-1]  # 取最后一维（适配state(16,)/(14,)、actions(30,16)）
                # print(f"!!!!!!{d}")
                half_d = d // 2
                arr_processed = arr.copy()
                
                # 0-(half_d-1)、(half_d+1)-(d-1)列取负（适配任意维度，用...兼容多维数组）
                arr_processed[..., :half_d-1] = -arr_processed[..., :half_d-1]
                arr_processed[..., half_d:d-1] = -arr_processed[..., half_d:d-1]
                
                # 前后半部分交换（沿最后一维拼接）
                arr_processed = np.concatenate(
                    [arr_processed[..., half_d:], arr_processed[..., :half_d]],
                    axis=-1
                )
                return arr_processed
            else:
                return arr  # 不处理，返回原数组

        def flip_image(image):
            """对图像进行水平翻转（全局统一触发，与数组处理同步）"""
            if should_process:  # 使用全局布尔变量，与数组处理保持一致
                return np.fliplr(image)  # 适配(360,640,3)图像格式，水平翻转
            else:
                return image  # 不翻转，返回原图像

        # 执行所有处理（全局概率统一控制，要么都处理，要么都不处理）
        state = process_array(state)
        # actions = np.asarray(data["actions"])
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = process_array(actions)
        base_image = flip_image(base_image)
        wrist_image_left_exchange = flip_image(wrist_image_left)
        wrist_image_right_exchange = flip_image(wrist_image_right)
        if episode_first_head_img is not None:
            episode_first_head_img = flip_image(episode_first_head_img)


        state_N = process_array(state_N)
        base_image_N = flip_image(base_image_N)
        wrist_image_left_exchange_N = flip_image(wrist_image_left_N)
        wrist_image_right_exchange_N = flip_image(wrist_image_right_N)
        if episode_first_head_img_N is not None:
            episode_first_head_img_N = flip_image(episode_first_head_img_N)

        if should_process:
            wrist_image_left = wrist_image_right_exchange
            wrist_image_right = wrist_image_left_exchange
            wrist_image_left_N = wrist_image_right_exchange_N
            wrist_image_right_N = wrist_image_left_exchange_N
        else: 
            wrist_image_left = wrist_image_left_exchange
            wrist_image_right = wrist_image_right_exchange 
            wrist_image_left_N = wrist_image_left_exchange_N
            wrist_image_right_N = wrist_image_right_exchange_N
            
        # import pdb; pdb.set_trace() #debug默认停的一个断点

        if self.use_left==False and self.use_right == True:
            wrist_image_left = np.zeros_like(base_image)
        elif self.use_right==False:
            wrist_image_right = np.zeros_like(base_image) 

        image_masks = (np.True_, np.True_, np.True_)

        if self.use_left==False and self.use_right == True:
            wrist_image_left = np.zeros_like(base_image)
            image_masks = (np.True_, np.False_, np.True_)
        elif self.use_left==True and self.use_right == False:
            wrist_image_right = np.zeros_like(base_image) 
            image_masks = (np.True_, np.True_, np.False_)
        

        image_by_key = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image_left,
            "right_wrist_0_rgb": wrist_image_right,
        }
        image_by_key_N = {
            "base_0_rgb": base_image_N,
            "left_wrist_0_rgb": wrist_image_left_N,
            "right_wrist_0_rgb": wrist_image_right_N,
        }
        if episode_first_head_img is not None:
            image_by_key["episode_first_head_img"] = episode_first_head_img
        if episode_first_head_img_N is not None:
            image_by_key_N["episode_first_head_img"] = episode_first_head_img_N
        missing = set(self.image_keys) - set(image_by_key)
        if missing:
            raise ValueError(f"Unsupported image_keys: {sorted(missing)}")
        missing_N = set(self.image_keys) - set(image_by_key_N)
        if missing_N:
            raise ValueError(f"Unsupported image_keys for observation_N: {sorted(missing_N)}")

        mask_by_key = {
            "base_0_rgb": image_masks[0],
            "left_wrist_0_rgb": image_masks[1],
            "right_wrist_0_rgb": image_masks[2],
        }
        if episode_first_head_img is not None:
            mask_by_key["episode_first_head_img"] = np.True_
        
        inputs = {
            "state": state,
            "state_N": state_N,
            "image": {key: image_by_key[key] for key in self.image_keys},
            "image_N": {key: image_by_key_N[key] for key in self.image_keys},
            "image_mask": {key: mask_by_key[key] for key in self.image_keys},


            "step_index": data["step_index"],
            "episode_length": data["episode_length"],
            "language_instruction_index": data["language_instruction_index"],
            "language_instruction_max_len": data["language_instruction_max_len"],
            "language_instruction_at_30precent": data["language_instruction_at_30precent"],
            "success_or_failure": data["success_or_failure"],
            
        }
        if "actions" in data:
            inputs["actions"]=actions

        # print(f"rsluo_test","step_index=",inputs["step_index"],"state=",inputs["state"],"state_N=",inputs["state_N"])


        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LindenOutputs(transforms.DataTransformFn):
    use_left:bool = False
    use_right:bool = False
    left_size_joint_num: int = 8
    right_size_joint_num: int = 8
    def __call__(self, data: dict) -> dict:
        if self.use_left == False and self.use_right == False:
            raise RuntimeError(f"LindenOutputs use_left and use_right must one be True")
        if self.use_left==True and self.use_right==False:
            return {"actions": np.asarray(data["actions"][:, 0:self.left_size_joint_num])}

        if self.use_left==False and self.use_right==True:
            return {"actions": np.asarray(data["actions"][:, 0:self.right_size_joint_num])}

        return {"actions": np.asarray(data["actions"][:, 0:(self.left_size_joint_num+self.right_size_joint_num)])}
