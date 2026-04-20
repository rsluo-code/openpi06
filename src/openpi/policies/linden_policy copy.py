import dataclasses

import einops
import numpy as np

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

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/joint_position"])
        # {
        #     actions
        #     observation
        #     prompt
        #     step_index
        #     episode_length
        # }


        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # import pdb; pdb.set_trace() #debug默认停的一个断点

        base_image = _parse_image(data["observation/image"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])
        images = (base_image, wrist_image_left, wrist_image_right)
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        image_masks = (np.True_, np.True_, np.True_)

        # match self.model_type:
        #     case _model.ModelType.PI0 | _model.ModelType.PI05:
        #         names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        #         images = (base_image, np.zeros_like(base_image), wrist_image_right)
        #         # images = (base_image, wrist_image_left, wrist_image_right)
        #         image_masks = (np.True_, np.True_, np.True_)
        #         # image_masks = (np.True_, np.True_, np.True_)
        #     case _model.ModelType.PI06 :
        #         names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        #         images = (base_image, np.zeros_like(base_image), wrist_image_right)
        #         # images = (base_image, wrist_image_left, wrist_image_right)
        #         image_masks = (np.True_, np.True_, np.True_)
        #     case _model.ModelType.PI06_ValueNet :
        #         names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        #         images = (base_image, np.zeros_like(base_image), wrist_image_right)
        #         # images = (base_image, wrist_image_left, wrist_image_right)
        #         image_masks = (np.True_, np.True_, np.True_)
        #     case _model.ModelType.PI0_FAST:
        #         names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
        #         # We don't mask out padding images for FAST models.
        #         images = (base_image, np.zeros_like(base_image), wrist_image)
        #         image_masks = (np.True_, np.True_, np.True_)
        #     case _:
        #         raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "step_index": data["step_index"],
            "episode_length": data["episode_length"],
            "language_instruction_index": data["language_instruction_index"],
            "language_instruction_index_max_len": data["language_instruction_index_max_len"],
            "success_or_failure": data["success_or_failure"],
            
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LindenOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
        return {"actions": np.asarray(data["actions"][:, :8])}
