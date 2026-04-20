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

        # state_N = np.asarray(data["observation/joint_position_N"])
        # base_image_N = _parse_image(data["observation/image_N"])
        # wrist_image_left_N = _parse_image(data["observation/wrist_image_left_N"])
        # wrist_image_right_N = _parse_image(data["observation/wrist_image_right_N"])

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
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = process_array(actions)
        base_image = flip_image(base_image)
        wrist_image_left_exchange = flip_image(wrist_image_left)
        wrist_image_right_exchange = flip_image(wrist_image_right)


        # state_N = process_array(state_N)
        # base_image_N = flip_image(base_image_N)
        # wrist_image_left_exchange_N = flip_image(wrist_image_left_N)
        # wrist_image_right_exchange_N = flip_image(wrist_image_right_N)

        if should_process:
            wrist_image_left = wrist_image_right_exchange
            wrist_image_right = wrist_image_left_exchange
            # wrist_image_left_N = wrist_image_right_exchange_N
            # wrist_image_right_N = wrist_image_left_exchange_N
        else: 
            wrist_image_left = wrist_image_left_exchange
            wrist_image_right = wrist_image_right_exchange 
            # wrist_image_left_N = wrist_image_left_exchange_N
            # wrist_image_right_N = wrist_image_right_exchange_N
            
        # import pdb; pdb.set_trace() #debug默认停的一个断点
        image_masks = (np.True_, np.True_, np.True_)

        if self.use_left==False and self.use_right == True:
            wrist_image_left = np.zeros_like(base_image)
            image_masks = (np.True_, np.False_, np.True_)
        elif self.use_right==False:
            wrist_image_right = np.zeros_like(base_image) 
            image_masks = (np.True_, np.True_, np.False_)
        

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        # images_N = (base_image_N, wrist_image_left_N, wrist_image_right_N)
        # image_masks = (np.True_, np.True_, np.True_)


        inputs = {
            "state": state,
            # "state_N": state_N,
            "image": dict(zip(names, images, strict=True)),
            # "image_N": dict(zip(names, images_N, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "step_index": data["step_index"],
            "episode_length": data["episode_length"],

        }

        if "actions" in data:
            inputs["actions"]=actions
        if "language_instruction_index" in data:
            inputs["language_instruction_index"]=data["language_instruction_index"]
        if "language_instruction_max_len" in data:
            inputs["language_instruction_max_len"]=data["language_instruction_max_len"]
        if "language_instruction_at_30precent" in data:
            inputs["language_instruction_at_30precent"]=data["language_instruction_at_30precent"]
        if "success_or_failure" in data:
            inputs["success_or_failure"]=data["success_or_failure"]
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

        return {"actions": np.asarray(data["actions"])}

