"""See _CONFIGS for the list of available configs."""
import sys
print(f"Loading {__name__}")
print(f"Modules loaded so far: {list(sys.modules.keys())[-5:]}") 


import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.linden_policy as linden_policy
import openpi.policies.linden_valuenet_inoutput as linden_valuenet_inoutput
import openpi.policies.linden_pi06_inoutput as linden_pi06_inoutput
import openpi.policies.linden_norm_state_inoutput as linden_norm_state_inoutput

import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.linden_rlds_valunet_dataset as linden_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    local_files_only: bool = False
    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    use_left: bool = False
    use_right: bool = False
    left_size_joint_num: int = 8
    right_size_joint_num: int = 8
    is_norm_state: bool = False
    is_pi06_data: bool = False
    is_valuenet_data: bool = False



class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        print(f"model_config.model_type={model_config.model_type}")
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI06_ValueNet:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI06:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    # outputs=[
                    #     _transforms.ExtractFASTActions(
                    #         tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                    #         action_horizon=model_config.action_horizon,
                    #         action_dim=model_config.action_dim,
                    #     )
                    # ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            # use_quantile_norm=False,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = None
    # filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )

@dataclasses.dataclass(frozen=True)
class RLDSLindenDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    use_left: bool = False
    use_right: bool = False
    left_size_joint_num: int = 8
    right_size_joint_num: int = 8
    train_or_infer:str = "infer"
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/wrist_image_left": "observation/left_wrist_image",
                        "observation/wrist_image_right": "observation/right_wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "actions": "actions",
                        "prompt": "prompt",
                        "step_index": "step_index",
                        "episode_length": "episode_length",
                        "language_instruction_index": "language_instruction_index",
                        "language_instruction_max_len": "language_instruction_max_len",
                        "success_or_failure": "success_or_failure",
                        
                        "observation/image_N": "observation/image_N",
                        "observation/wrist_image_left_N": "observation/left_wrist_image_N",
                        "observation/wrist_image_right_N": "observation/right_wrist_image_N",
                        "observation/joint_position_N": "observation/joint_position_N",
                        "language_instruction_at_30precent": "language_instruction_at_30precent",
                    }
                )
            ]
        )

        # data_transforms = _transforms.Group(
        #     inputs=[linden_policy.LindenInputs(model_type=model_config.model_type)],
        #     outputs=[linden_policy.LindenOutputs()],
        # )
        data_transforms = _transforms.Group(
            inputs=[linden_policy.LindenInputs(model_type=model_config.model_type,use_left=self.use_left,use_right=self.use_right,train_or_infer = self.train_or_infer)],
            outputs=[linden_policy.LindenOutputs(use_left=self.use_left,use_right=self.use_right,left_size_joint_num=self.left_size_joint_num,right_size_joint_num=self.right_size_joint_num)],
        )
        # Data loader returns absolute joint position actions -- convert to delta actions for training.
        if self.use_left == False and self.use_right == False:
            raise RuntimeError(f"RLDSLindenValueNetDataConfig use_left and use_right must one be True")
        elif self.use_left==True and self.use_right==False:
            delta_action_mask = _transforms.make_bool_mask(self.left_size_joint_num-1, -1)

        elif self.use_left==False and self.use_right==True:
            delta_action_mask = _transforms.make_bool_mask(self.right_size_joint_num-1, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(self.left_size_joint_num-1, -1,self.right_size_joint_num-1, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            use_left= self.use_left,
            use_right = self.use_right,
            left_size_joint_num = self.left_size_joint_num,
            right_size_joint_num = self.right_size_joint_num,
            # is_valuenet_data=True,

        )


@dataclasses.dataclass(frozen=True)
class RLDSLindenValueNetDataConfig(DataConfigFactory):

    rlds_data_dir: str | None = None
    use_left: bool = False
    use_right: bool = False
    left_size_joint_num: int = 8
    right_size_joint_num: int = 8
    train_or_infer:str = "infer"
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/wrist_image_left": "observation/left_wrist_image",
                        "observation/wrist_image_right": "observation/right_wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "actions": "actions",
                        "prompt": "prompt",
                        "step_index": "step_index",
                        "episode_length": "episode_length",
                        "language_instruction_index": "language_instruction_index",
                        "language_instruction_max_len": "language_instruction_max_len",
                        "language_instruction_at_30precent": "language_instruction_at_30precent",
                        
                        "success_or_failure": "success_or_failure",
                        
                        # "observation/image_N": "observation/image_N",
                        # "observation/wrist_image_left_N": "observation/left_wrist_image_N",
                        # "observation/wrist_image_right_N": "observation/right_wrist_image_N",
                        # "observation/joint_position_N": "observation/joint_position_N",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[linden_valuenet_inoutput.LindenInputs(model_type=model_config.model_type,use_left=self.use_left,use_right=self.use_right,train_or_infer = self.train_or_infer)],
            outputs=[linden_valuenet_inoutput.LindenOutputs(use_left=self.use_left,use_right=self.use_right,left_size_joint_num=self.left_size_joint_num,right_size_joint_num=self.right_size_joint_num)],
        )

        # Data loader returns absolute joint position actions -- convert to delta actions for training.
        if self.use_left == False and self.use_right == False:
            raise RuntimeError(f"RLDSLindenValueNetDataConfig use_left and use_right must one be True")
        elif self.use_left==True and self.use_right==False:
            delta_action_mask = _transforms.make_bool_mask(self.left_size_joint_num-1, -1)

        elif self.use_left==False and self.use_right==True:
            delta_action_mask = _transforms.make_bool_mask(self.right_size_joint_num-1, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(self.left_size_joint_num-1, -1,self.right_size_joint_num-1, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            use_left= self.use_left,
            use_right = self.use_right,
            left_size_joint_num = self.left_size_joint_num,
            right_size_joint_num = self.right_size_joint_num,
            is_valuenet_data=True
        )


@dataclasses.dataclass(frozen=True)
class RLDSLindenPI06DataConfig(DataConfigFactory):

    rlds_data_dir: str | None = None
    use_left: bool = False
    use_right: bool = False
    left_size_joint_num: int = 8
    right_size_joint_num: int = 8
    train_or_infer:str = "infer"
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/wrist_image_left": "observation/left_wrist_image",
                        "observation/wrist_image_right": "observation/right_wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "actions": "actions",
                        "prompt": "prompt",
                        "step_index": "step_index",
                        "episode_length": "episode_length",
                        "language_instruction_index": "language_instruction_index",
                        "language_instruction_max_len": "language_instruction_max_len",
                        "language_instruction_at_30precent": "language_instruction_at_30precent",
                        
                        "success_or_failure": "success_or_failure",
                        
                        "observation/image_N": "observation/image_N",
                        "observation/wrist_image_left_N": "observation/left_wrist_image_N",
                        "observation/wrist_image_right_N": "observation/right_wrist_image_N",
                        "observation/joint_position_N": "observation/joint_position_N",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[linden_pi06_inoutput.LindenInputs(model_type=model_config.model_type,use_left=self.use_left,use_right=self.use_right,train_or_infer = self.train_or_infer)],
            outputs=[linden_pi06_inoutput.LindenOutputs(use_left=self.use_left,use_right=self.use_right,left_size_joint_num=self.left_size_joint_num,right_size_joint_num=self.right_size_joint_num)],
        )

        # Data loader returns absolute joint position actions -- convert to delta actions for training.
        if self.use_left == False and self.use_right == False:
            raise RuntimeError(f"RLDSLindenPI06DataConfig use_left and use_right must one be True")
        elif self.use_left==True and self.use_right==False:
            delta_action_mask = _transforms.make_bool_mask(self.left_size_joint_num-1, -1)

        elif self.use_left==False and self.use_right==True:
            delta_action_mask = _transforms.make_bool_mask(self.right_size_joint_num-1, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(self.left_size_joint_num-1, -1,self.right_size_joint_num-1, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            use_left= self.use_left,
            use_right = self.use_right,
            left_size_joint_num = self.left_size_joint_num,
            right_size_joint_num = self.right_size_joint_num,
            is_pi06_data=True
        )



@dataclasses.dataclass(frozen=True)
class RLDSLindenNormStateDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {

                        "actions": "actions",
                        "observation/joint_position": "observation/joint_position",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[linden_norm_state_inoutput.LindenInputs(model_type=model_config.model_type)],
            outputs=[linden_norm_state_inoutput.LindenOutputs()],
        )

        # Data loader returns absolute joint position actions -- convert to delta actions for training.
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
        # delta_action_mask = _transforms.make_bool_mask(7, -1)
        # delta_action_mask = _transforms.make_bool_mask(3, -5, 3, -5)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[],
        )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

         
        result = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            is_norm_state=True,
            
        )

        return result

@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    # checkpoint_base_dir: str = "/yrfs2/cv5/jfni3/checkpoints"
    # checkpoint_base_dir: str = "/b3-mix03/sppro/permanent/jfni3/pi05_checkpoints"
    checkpoint_base_dir: str = "/data0/rsluo/pi06_torch"
    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    
    pi06: bool = False
    pi06_valuenet: bool = False

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),


    # norm_state_16dim
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="norm_state_16dim",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_300m_valuenet", action_expert_variant="gemma_300m"),
        data=RLDSLindenNormStateDataConfig(
            repo_id="20260413_sf_packages_rightarm",
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            # rlds_data_dir="/DATA/disk3/train_data/1024_翻转包裹V2_rlsd_224/data_10_23",
        ),
        pytorch_weight_path="/data0/rsluo/pi05_base_torch_bf16",
        pi06=False,
        pi06_valuenet=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=160_001,
        batch_size=128,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),


    # value_pretrain_16dim_calAt
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="value_pretrain_16dim_calAt",
        # model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_300m_valuenet", action_expert_variant="gemma_300m"),
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"),
        data=RLDSLindenPI06DataConfig(
            repo_id="sf_packages_rightarm_20260413_normstats",
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            use_left= False,
            use_right = True,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "infer"
        ),
        pytorch_weight_path="/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260413/80000",
        pi06=False,
        pi06_valuenet=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=3_341,
        batch_size=3072,
        log_interval=2,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),

    # value_pretrain_16dim_calval
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="value_pretrain_16dim_calval",
        # model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_300m_valuenet", action_expert_variant="gemma_300m"),
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"),
        # data=RLDSLindenPI06DataConfig(
        data=RLDSLindenValueNetDataConfig(
            # repo_id="LDT_1228_16joint",
            repo_id="sf_packages_rightarm_20260413_normstats",
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            use_left= False,
            use_right = True,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "infer"
        ),
        pytorch_weight_path="/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260413/80000",
        pi06=False,
        pi06_valuenet=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=250_001,
        # num_train_steps=24_001,
        batch_size=80,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),

    # value_pretrain_16dim
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="value_pretrain_16dim",
        # model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_300m_valuenet", action_expert_variant="gemma_300m"),
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"),
        data=RLDSLindenValueNetDataConfig(
            repo_id="sf_packages_rightarm_20260413_normstats",
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            use_left= False,
            use_right = True,

            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "train"
        ),
        pytorch_weight_path="/data0/rsluo/pi05_base_torch_bf16",
        pi06=False,
        pi06_valuenet=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=520_001,
        batch_size=80,
        log_interval=20,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        resume=True
    ),

    # value_pretrain_right_arm_8dim_sf_packages
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="value_pretrain_right_arm_8dim_sf_packages",
        # model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_300m_valuenet", action_expert_variant="gemma_300m"),
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"),
        data=RLDSLindenValueNetDataConfig(
            repo_id="SF_delivery_10_29-12_12_01_27_v1",
            rlds_data_dir="/DATA/disk2/1024_grasp_packages/data/SF_convert_rlds_data",
            use_left= False,
            use_right = True,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "train"
        ),
        pytorch_weight_path="/root/rsluo/model/pi05_torch_原始权重",
        pi06=False,
        pi06_valuenet=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=520_001,
        batch_size=64,
        log_interval=20,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),


    # PI06_pretrain_validation
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="PI06_pretrain_validation",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",if_use_valuenet=False),
        # data=RLDSLindenValueNetDataConfig(
        data=RLDSLindenPI06DataConfig(
            repo_id="sf_packages_rightarm_20260413_normstats",
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            use_left= True,
            use_right = False,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            # train_or_infer = "train",
            train_or_infer = "infer",

        ),
        pytorch_weight_path="/data0/rsluo/pi05_base_torch_bf16",
        pi06=True,
        pi06_valuenet=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_001,
        batch_size=256,
        log_interval=20,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        resume=True
    ),


    # PI06_pretrain
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="PI06_pretrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",if_use_valuenet=False),
        # data=RLDSLindenValueNetDataConfig(
        data=RLDSLindenPI06DataConfig(
            repo_id="sf_packages_rightarm_20260413_normstats",
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            use_left= False,
            use_right = True,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "train",
            # train_or_infer = "infer",
        ),
        pytorch_weight_path="/data0/rsluo/pi05_base_torch_bf16",
        pi06=True,
        pi06_valuenet=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_001,
        batch_size=256,
        log_interval=20,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),

    # PI05_bc
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="PI05_bc",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"),
        data=RLDSLindenValueNetDataConfig(
        # data=RLDSLindenPI06DataConfig(
            repo_id="LDT_20260226_16joint_10_31",
            # repo_id="LDT_1026_joint",
            # repo_id="LDT_20260104_16joint_exchangesize_stateN",
            rlds_data_dir="/DATA/disk3/train_data/1024_翻转包裹V2_rlsd_224/partall",
            use_left= True,
            use_right = True,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "train",
            # train_or_infer = "infer",
        ),
        # pytorch_weight_path="/data0/rsluo/pi05_base_torch_bf16",
        pi06=False,
        pi06_valuenet=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_001,
        batch_size=12,
        log_interval=20,
        save_interval=2500,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),




    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="linden_torch",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"),
        data=RLDSLindenValueNetDataConfig(
            repo_id="20260413_sf_packages_rightarm",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/data0/rlds_datas_cut/sf_packages",
            use_left= True,
            use_right = True,
            left_size_joint_num = 8,
            right_size_joint_num = 8,
            train_or_infer = "train"
        ),
        pytorch_weight_path="/data0/rsluo/pi05_base_torch_bf16",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=60_000,
        batch_size=128,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        wandb_enabled=False,        
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # resume=True
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
