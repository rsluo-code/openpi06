import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import numpy as np

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from openpi.models_pytorch.reward_head import ProgressHead
from openpi.models_pytorch.model_utils import TCNDownsampler


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        num_timesteps = 3 
        suffix_embs_expand = []
        att_2d_masks_4d_expand = []
        position_ids_expand = []
        adarms_cond_expand = []
        u_t_expand = []
        batch_size, num_steps, num_motors = actions.shape
        if noise is None:
            noise = self.sample_noise((num_timesteps * batch_size, num_steps, num_motors), actions.device)
        
        if time is None:
            time = self.sample_time(num_timesteps * batch_size, actions.device)

        noise = noise.reshape(num_timesteps, batch_size, num_steps, num_motors)
        time = time.reshape(num_timesteps, batch_size)
        for i in range(num_timesteps):
            
            time_expanded = time[i, :, None, None]
            x_t = time_expanded * noise[i] + (1 - time_expanded) * actions
            u_t = noise[i] - actions
            
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time[i])
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            # Prepare attention masks
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            suffix_embs_expand.append(suffix_embs)
            att_2d_masks_4d_expand.append(att_2d_masks_4d)
            position_ids_expand.append(position_ids)
            adarms_cond_expand.append(adarms_cond)
            u_t_expand.append(u_t)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs_expand, att_2d_masks_4d_expand, position_ids_expand, adarms_cond_expand, num_timesteps):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d_expand,
                position_ids=position_ids_expand,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs_expand],
                use_cache=False,
                adarms_cond=[None, adarms_cond_expand],
                num_timesteps=num_timesteps,
            )
            return suffix_out

        suffix_out_expand = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs_expand, att_2d_masks_4d_expand, position_ids_expand, adarms_cond_expand, num_timesteps
        )

        losses = 0
        for i in range(num_timesteps):
            suffix_out = suffix_out_expand[i][:, -self.config.action_horizon :]
            suffix_out = suffix_out.to(dtype=torch.float32)

            # Apply gradient checkpointing to final action projection if enabled
            def action_out_proj_func(suffix_out):
                return self.action_out_proj(suffix_out)

            v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
            loss = F.mse_loss(u_t_expand[i], v_t, reduction="none")
            losses += loss

        # return losses/num_timesteps

        base_loss = losses/num_timesteps
        device = actions.device
        x = torch.linspace(-1, 1, num_steps, device=device)
        
        # mu = 0.0  
        # sigma = 0.0         
        # gaussian_weights = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)        
        # gaussian_weights = (gaussian_weights - gaussian_weights.min()) / (gaussian_weights.max() - gaussian_weights.min())
        # gaussian_weights = gaussian_weights * 1.0 + 1.0
        
        gaussian_weights = x * 0.0
        
        # (chunk_size, ) -> (1, chunk_size, 1) -> (batch_size, chunk_size, action_dim)
        weights = gaussian_weights[None, :, None].expand(batch_size, num_steps, num_motors)     

        special_weights = torch.ones_like(weights) 
    
        for batch_idx in range(batch_size):
            gaussian_centers = []            
            for step_idx in range(num_steps):
                if actions[batch_idx, step_idx, 7] <= 0.00019904:  #(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0   #or actions[batch_idx, step_idx, 13] <= 0.03:
                    gaussian_centers.append(step_idx)
            
            gaussian_distributions = []
            for center_step in gaussian_centers:
                x_positions = torch.arange(num_steps, device=device, dtype=torch.float32)
                sigma_gripper = 2.0
                
                gaussian = torch.exp(-0.5 * ((x_positions - center_step) / sigma_gripper) ** 2)
                
                gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
                gaussian = gaussian * 1.0 + 1.0 
                
                gaussian_distributions.append(gaussian)
            
            if gaussian_distributions:
                stacked_gaussians = torch.stack(gaussian_distributions)  # (num_centers, chunk_size)
                max_gaussian = stacked_gaussians.max(dim=0)[0]  # (chunk_size,)
                
                special_weights[batch_idx, :, :] = torch.maximum(
                    special_weights[batch_idx, :, :],
                    max_gaussian[:, None].expand(num_steps, num_motors)
                )

        
        final_weights = weights + special_weights
        weighted_loss = base_loss * final_weights
        return weighted_loss


        # images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        
        # batch_size, num_steps, num_motors = actions.shape
        # num_timesteps = 2 
        
        # if noise is None:
        #     noise = self.sample_noise((batch_size * num_timesteps, num_steps, num_motors), actions.device)
        
        # if time is None:
        #     time = self.sample_time(batch_size * num_timesteps, actions.device)  # (batch_size * num_timesteps,)
        
        # actions_expanded = actions.repeat_interleave(num_timesteps, dim=0)  # (batch_size * num_timesteps, num_steps, num_motors)
        # state_expanded = state.repeat_interleave(num_timesteps, dim=0)  # (batch_size * num_timesteps, state_dim)
        
        # time_expanded = time[:, None, None]  # (batch_size * num_timesteps, 1, 1)
        # x_t = time_expanded * noise + (1 - time_expanded) * actions_expanded
        # u_t = noise - actions_expanded  # (batch_size * num_timesteps, num_steps, num_motors)
        
        # prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        # prefix_embs_expanded = prefix_embs.repeat_interleave(num_timesteps, dim=0)
        # prefix_pad_masks_expanded = prefix_pad_masks.repeat_interleave(num_timesteps, dim=0)
        # prefix_att_masks_expanded = prefix_att_masks.repeat_interleave(num_timesteps, dim=0)
        
        # suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
        #     state_expanded, x_t, time
        # )
        
        # if (
        #     self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        #     == torch.bfloat16
        # ):
        #     suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        #     prefix_embs_expanded = prefix_embs_expanded.to(dtype=torch.bfloat16)
        
        # pad_masks = torch.cat([prefix_pad_masks_expanded, suffix_pad_masks], dim=1)
        # att_masks = torch.cat([prefix_att_masks_expanded, suffix_att_masks], dim=1)
        
        # att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        
        # def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        #     (_, suffix_out), _ = self.paligemma_with_expert.forward(
        #         attention_mask=att_2d_masks_4d,
        #         position_ids=position_ids,
        #         past_key_values=None,
        #         inputs_embeds=[prefix_embs, suffix_embs],
        #         use_cache=False,
        #         adarms_cond=[None, adarms_cond],
        #     )
        #     return suffix_out
        
        # suffix_out = self._apply_checkpoint(
        #     forward_func, prefix_embs_expanded, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        # )
        
        # suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)
        
        # def action_out_proj_func(suffix_out):
        #     return self.action_out_proj(suffix_out)
        
        # v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        
        # loss = F.mse_loss(u_t, v_t, reduction="none")
        
        # loss = loss.view(batch_size, num_timesteps, num_steps, num_motors)
        
        # return loss.mean(dim=1)



        # images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        # prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # if noise is None:
        #     noise = self.sample_noise(actions.shape, actions.device)

        # if time is None:
        #     time = self.sample_time(actions.shape[0], actions.device)

        # time_expanded = time[:, None, None]
        # x_t = time_expanded * noise + (1 - time_expanded) * actions
        # u_t = noise - actions
        
        # suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        # if (
        #     self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        #     == torch.bfloat16
        # ):
        #     suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        #     prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        # att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # # Prepare attention masks
        # att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # # Apply gradient checkpointing if enabled
        # def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        #     (_, suffix_out), _ = self.paligemma_with_expert.forward(
        #         attention_mask=att_2d_masks_4d,
        #         position_ids=position_ids,
        #         past_key_values=None,
        #         inputs_embeds=[prefix_embs, suffix_embs],
        #         use_cache=False,
        #         adarms_cond=[None, adarms_cond],
        #     )
        #     return suffix_out

        # suffix_out = self._apply_checkpoint(
        #     forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        # )

        # suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)

        # # Apply gradient checkpointing to final action projection if enabled
        # def action_out_proj_func(suffix_out):
        #     return self.action_out_proj(suffix_out)

        # v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond = [None, None],
            num_timesteps = 1,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            num_timesteps = 1,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI0StateValueModelPytorch(PI0Pytorch):
    IMAGE_KEYS = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        # "base_t0_rgb",
    )
    
    def __init__(self, config):
        super().__init__(config)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # self.post_layer = TCNDownsampler(paligemma_config.width, 
        #                                  paligemma_config.width,
        #                                  seq_len=968,
        #                                  kernel_size=3,
        #                                  dropout=0.0)

        self.progress_head = ProgressHead(paligemma_config.width,
                                          max_progress=1, 
                                          min_progress=-1000, 
                                          num_bins=100,
                                          hidden_sizes=(1024, 512, 512, 256))
        
    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train, image_keys=self.IMAGE_KEYS)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
        
    def forward0(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        task_progress = observation.task_progress
        assert task_progress is not None

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        num_timesteps = 3 
        suffix_embs_expand = []
        att_2d_masks_4d_expand = []
        position_ids_expand = []
        adarms_cond_expand = []
        u_t_expand = []
        batch_size, num_steps, num_motors = actions.shape
        if noise is None:
            noise = self.sample_noise((num_timesteps * batch_size, num_steps, num_motors), actions.device)
        
        if time is None:
            time = self.sample_time(num_timesteps * batch_size, actions.device)

        noise = noise.reshape(num_timesteps, batch_size, num_steps, num_motors)
        time = time.reshape(num_timesteps, batch_size)
        for i in range(num_timesteps):
            
            time_expanded = time[i, :, None, None]
            x_t = time_expanded * noise[i] + (1 - time_expanded) * actions
            u_t = noise[i] - actions
            
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time[i])
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            # Prepare attention masks
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            suffix_embs_expand.append(suffix_embs)
            att_2d_masks_4d_expand.append(att_2d_masks_4d)
            position_ids_expand.append(position_ids)
            adarms_cond_expand.append(adarms_cond)
            u_t_expand.append(u_t)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs_expand, att_2d_masks_4d_expand, position_ids_expand, adarms_cond_expand, num_timesteps):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d_expand,
                position_ids=position_ids_expand,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs_expand],
                use_cache=False,
                adarms_cond=[None, adarms_cond_expand],
                num_timesteps=num_timesteps,
            )
            return suffix_out, prefix_out

        suffix_out_expand, prefix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs_expand, att_2d_masks_4d_expand, position_ids_expand, adarms_cond_expand, num_timesteps
        )

        losses = 0
        for i in range(num_timesteps):
            suffix_out = suffix_out_expand[i][:, -self.config.action_horizon :]
            suffix_out = suffix_out.to(dtype=torch.float32)

            # Apply gradient checkpointing to final action projection if enabled
            def action_out_proj_func(suffix_out):
                return self.action_out_proj(suffix_out)

            v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
            loss = F.mse_loss(u_t_expand[i], v_t, reduction="none")
            losses += loss

        # return losses/num_timesteps

        base_loss = losses/num_timesteps
        device = actions.device
        x = torch.linspace(-1, 1, num_steps, device=device)
        
        # mu = 0.0  
        # sigma = 0.0         
        # gaussian_weights = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)        
        # gaussian_weights = (gaussian_weights - gaussian_weights.min()) / (gaussian_weights.max() - gaussian_weights.min())
        # gaussian_weights = gaussian_weights * 1.0 + 1.0
        
        gaussian_weights = x * 0.0
        
        # (chunk_size, ) -> (1, chunk_size, 1) -> (batch_size, chunk_size, action_dim)
        weights = gaussian_weights[None, :, None].expand(batch_size, num_steps, num_motors)     

        special_weights = torch.ones_like(weights) 
    
        for batch_idx in range(batch_size):
            gaussian_centers = []            
            for step_idx in range(num_steps):
                if actions[batch_idx, step_idx, 7] <= 0.00019904:  #(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0   #or actions[batch_idx, step_idx, 13] <= 0.03:
                    gaussian_centers.append(step_idx)
            
            gaussian_distributions = []
            for center_step in gaussian_centers:
                x_positions = torch.arange(num_steps, device=device, dtype=torch.float32)
                sigma_gripper = 2.0
                
                gaussian = torch.exp(-0.5 * ((x_positions - center_step) / sigma_gripper) ** 2)
                
                gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
                gaussian = gaussian * 1.0 + 1.0 
                
                gaussian_distributions.append(gaussian)
            
            if gaussian_distributions:
                stacked_gaussians = torch.stack(gaussian_distributions)  # (num_centers, chunk_size)
                max_gaussian = stacked_gaussians.max(dim=0)[0]  # (chunk_size,)
                
                special_weights[batch_idx, :, :] = torch.maximum(
                    special_weights[batch_idx, :, :],
                    max_gaussian[:, None].expand(num_steps, num_motors)
                )

        
        final_weights = weights + special_weights
        bc_loss = base_loss * final_weights

        # 进度预测模型训练
        prefix_out = prefix_out.to(torch.float32)
        prefix_out = torch.mean(prefix_out, dim=1)
        logits = self.progress_head(prefix_out)
        progress_labels = self.progress_head.progress_to_indices(task_progress)
        progress_log_probs = self.progress_head.get_log_prob(logits, progress_labels.squeeze())
        progress_loss = -torch.mean(progress_log_probs)

        loss = torch.mean(bc_loss.detach()) * 0. + progress_loss

        train_info = {
            "bc_loss": torch.mean(bc_loss).item(),
            "progress_loss": progress_loss.item(),
            "progress_logits": torch.mean(logits).item(),
            "progress_entropy": torch.mean(self.progress_head.get_entropy(logits)).item(),
        }
        return loss, train_info
    
    def forward1(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        task_progress = observation.task_progress
        assert task_progress is not None

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_out, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            adarms_cond=None,
            num_timesteps=None,
        )

        # 进度预测模型训练
        prefix_out = prefix_out.to(torch.float32)
        prefix_out = torch.sum(prefix_out * prefix_pad_masks[..., None], dim=1) / (torch.sum(prefix_pad_masks[..., None], dim=1) + 1e-8)
        # prefix_out = torch.mean(prefix_out, dim=1)
        logits = self.progress_head(prefix_out)
        progress_labels = self.progress_head.progress_to_indices(task_progress)
        progress_log_probs = self.progress_head.get_log_prob(logits, progress_labels.squeeze())
        progress_loss = -torch.mean(progress_log_probs)

        loss = progress_loss

        acc = (torch.argmax(logits, dim=1) == progress_labels.squeeze(1)).to(torch.float32)
        soft_acc = torch.abs(self.progress_head.logits_to_progress(logits) - task_progress.squeeze(1))

        train_info = {
            "progress_loss": progress_loss.item(),
            "progress_logits": torch.mean(logits).item(),
            "progress_entropy": torch.mean(self.progress_head.get_entropy(logits)).item(),
            "progress_acc": acc.mean().item(),
            "progress_soft_acc": soft_acc.mean().item(),
            "progress_soft_acc_max": soft_acc.max().item(),
            "progress_soft_acc_min": soft_acc.min().item(),
        }
        return loss, train_info
    
    def forward(self, observation, actions, ratio=None, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        # task_progress = observation.task_progress
        task_progress = ratio.unsqueeze(1)  
        assert task_progress is not None

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward_pi0(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # 进度预测模型训练
        prefix_out = prefix_out.to(torch.float32)
        prefix_out = torch.sum(prefix_out * prefix_pad_masks[..., None], dim=1) / (torch.sum(prefix_pad_masks[..., None], dim=1) + 1e-8)
        # prefix_out = torch.mean(prefix_out, dim=1)
        # prefix_out = prefix_out * prefix_pad_masks[..., None]
        # prefix_out = self.post_layer(prefix_out).squeeze(1)
        logits = self.progress_head(prefix_out)
        progress_labels = self.progress_head.progress_to_indices(task_progress)
        progress_log_probs = self.progress_head.get_log_prob(logits, progress_labels.squeeze())
        progress_loss = -torch.mean(progress_log_probs)

        loss = progress_loss

        acc = (torch.argmax(logits, dim=1) == progress_labels.squeeze(1)).to(torch.float32)
        soft_acc = torch.abs(self.progress_head.logits_to_progress(logits) - task_progress.squeeze(1))

        train_info = {
            "progress_loss": progress_loss.item(),
            "progress_logits": torch.mean(logits).item(),
            "progress_entropy": torch.mean(self.progress_head.get_entropy(logits)).item(),
            "progress_acc": acc.mean().item(),
            "progress_soft_acc": soft_acc.mean().item(),
            "progress_soft_acc_max": soft_acc.max().item(),
            "progress_soft_acc_min": soft_acc.min().item(),
        }
        return loss, train_info


    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, **kwargs) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = x_t = self.sample_noise(actions_shape, device)
        time = self.sample_time(actions_shape[0], device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Apply gradient checkpointing if enabled
        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward_pi0(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        prefix_out = prefix_out.to(torch.float32)
        prefix_out = torch.sum(prefix_out * prefix_pad_masks[..., None], dim=1) / (torch.sum(prefix_pad_masks[..., None], dim=1) + 1e-8)
        # prefix_out = torch.mean(prefix_out, dim=1)
        # prefix_out = prefix_out * prefix_pad_masks[..., None]
        # prefix_out = self.post_layer(prefix_out).squeeze(1)
        logits = self.progress_head(prefix_out)
        return self.progress_head.logits_to_progress(logits)

    @torch.no_grad()
    def value_get(self, observation, actions, noise=None, time=None) -> Tensor:

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward_pi0(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # 进度预测模型训练
        prefix_out = prefix_out.to(torch.float32)
        prefix_out = torch.sum(prefix_out * prefix_pad_masks[..., None], dim=1) / (torch.sum(prefix_pad_masks[..., None], dim=1) + 1e-8)
        # prefix_out = torch.mean(prefix_out, dim=1)
        # prefix_out = prefix_out * prefix_pad_masks[..., None]
        # prefix_out = self.post_layer(prefix_out).squeeze(1)
        logits = self.progress_head(prefix_out)
        logits = self.progress_head.logits_to_progress(logits)
        return logits
   
class PI0PolicyPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        valuenet_config = _config.get_config("reward_model-pick_place") 
        valuenet_checkpoint_dir = "/wx-mix01/sppro/permanent/zqnie7/checkpoints/reward_model-pick_place/value_reward_model-pick_place/42500"
        self.policy = _policy_config.create_trained_policy(valuenet_config, valuenet_checkpoint_dir)#, pytorch_device="cuda:1"
        

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond
    
    # 连续动作用高斯分布的log_prob
    def forward_gs(self, observation, actions, returns=None, noise=None, time=None) -> tuple[torch.Tensor, dict]:
        """
        重构为AWR策略网络的Loss计算：
        - 核心：优势加权的对数似然损失（AWR Policy Loss）
        - 保留原有前向传播、special_weights等自定义逻辑
        """
        # ===================== 1. AWR核心参数初始化（修正原numpy操作为torch，避免设备不匹配） =====================
        beta = 0.03  # 修正：论文中beta是0.05（原0.005过小，权重敏感度会异常）
        weight_max = 20.0  # 权重裁剪阈值（对齐AWR论文）
        
        # 计算状态价值V(s,a)（来自value网络）
        V_s = self.policy._model.value_get(observation, actions)

        
        # 计算优势函数 A(s,a) = R(s,a) - V(s,a)（确保维度匹配）
        assert returns is not None, "AWR需要returns计算优势函数，请传入returns参数"
        advantages = returns - V_s  # [batch_size, num_steps, num_motors]
        
        # 计算优势加权权重：exp(A / beta)，全程用torch操作（避免numpy和tensor设备不匹配）
        weights_awr = torch.exp(advantages * beta)  # 修正：原代码是*beta，论文中是1/beta
        weights_awr = torch.clamp(weights_awr, 0.0, weight_max)  # 权重裁剪，防止梯度爆炸

        # ===================== 2. 保留原有前向传播逻辑（paligemma模型、noise/time处理） =====================
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        batch_size, num_steps, num_motors = actions.shape

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward_pi0(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # ===================== 3. 重构Loss计算：替换为AWR Policy Loss =====================
        # AWR核心：策略梯度损失 = -E[优势权重 * 对数似然]（连续动作用高斯分布的log_prob）
        awr_losses = 0.0
        device = actions.device
        
        # 合并权重：AWR优势权重 + 自定义special_weights
        final_weights = weights_awr.unsqueeze(1).unsqueeze(2)

        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # 动作投影（保留原有梯度检查点）
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)
        y_t = self._apply_checkpoint(action_out_proj_func, suffix_out)  # 策略网络输出的动作均值
        
        # 连续动作：假设策略输出高斯分布（均值y_t，固定小方差增加稳定性）
        action_std = 0.1  # 可配置，连续动作的标准差（AWR论文中常用0.1~0.5）
        normal_dist = torch.distributions.Normal(y_t, action_std)
        
        # 计算真实动作的对数似然 log_prob(a|s)
        log_prob = normal_dist.log_prob(actions)  # [batch_size, num_steps, num_motors]
        log_prob = torch.clamp(log_prob, min=-10.0, max=10.0)  # 防止log_prob溢出
        
        # AWR Loss：-（优势权重 * 对数似然）的均值（reduction="none"保留维度）
        weighted_loss = - (final_weights * log_prob)

        # 最终Loss：求所有维度的均值（用于反向传播）
        total_loss = weighted_loss.mean()

        # ===================== 4. 完善train_info：添加AWR关键指标 =====================
        train_info = {
            "weighted_loss": total_loss.item(),
            "awr_advantage_mean": advantages.mean().item(),  # 优势函数均值
            "awr_weights_mean": weights_awr.mean().item(),   # AWR权重均值
            "log_prob_mean": log_prob.mean().item()          # 对数似然均值
        }

        return total_loss, train_info
        
    # 连续动作用fm似然
    def forward(self, observation, actions, returns=None, noise=None, time=None) -> tuple[torch.Tensor, dict]:
        """
        重构为AWR策略网络的Loss计算：
        - 核心：优势加权的对数似然损失（AWR Policy Loss）
        - 保留原有前向传播、special_weights等自定义逻辑
        """
        # ===================== 1. AWR核心参数初始化（修正原numpy操作为torch，避免设备不匹配） =====================
        beta = 0.014 # 修正：论文中beta是0.05（原0.005过小，权重敏感度会异常）
        weight_max = 20.0  # 权重裁剪阈值（对齐AWR论文）

        # 计算状态价值V(s,a)（来自value网络）
        V_s = self.policy._model.value_get(observation, actions)

        
        # 计算优势函数 A(s,a) = R(s,a) - V(s,a)（确保维度匹配）
        assert returns is not None, "AWR需要returns计算优势函数，请传入returns参数"
        advantages = returns - V_s  # [batch_size, num_steps, num_motors]
        # advantages = torch.zeros_like(returns, device=actions.device)  # [batch_size, num_steps, num_motors]
        
        # 计算优势加权权重：exp(A / beta)，全程用torch操作（避免numpy和tensor设备不匹配）
        weights_awr = torch.exp(advantages * beta)  # 修正：原代码是*beta，论文中是1/beta
        weights_awr = torch.clamp(weights_awr, 0.0, weight_max)  # 权重裁剪，防止梯度爆炸

        # ===================== 2. 保留原有前向传播逻辑（paligemma模型、noise/time处理） =====================
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        batch_size, num_steps, num_motors = actions.shape

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward_pi0(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        # ===================== 3. 核心修改：Flow Matching似然计算（替换原高斯分布log_prob） =====================
        # 定义Flow Matching对数似然计算函数（含蒙特卡洛采样）
        def compute_fm_log_likelihood(y_t, actions, noise, time, device):
            """
            Flow Matching对数似然计算（基于你提供的公式下界近似）
            :param y_t: 策略模型输出的动作修正项 f_θ，[batch_size, num_steps, num_motors]
            :param actions: 真实动作序列 a_{1:H}，[batch_size, num_steps, num_motors]
            :param observation: 条件信息 I_t/o_t 等
            :param N: 蒙特卡洛采样次数（越大越准，速度越慢）
            :return: fm_log_prob: Flow Matching对数似然，[batch_size, num_steps, num_motors]
            """
            total_log_prob = torch.zeros_like(actions, device=device)
            
            # 步骤1：采样噪声η（标量，均匀分布/指数分布，这里选均匀分布[0,1]）
            eta = time
            w_eta = torch.exp(-eta / 2)  # 权重项 w(η) = e^(-η/2)
            
            # 步骤2：采样噪声ω（与动作同维度的高斯噪声，ω ~ N(0, I)）
            omega = noise
            
            # 步骤3：计算Flow Matching误差项的L2范数平方
            # 公式中的 f_θ 对应代码中策略输出的 y_t，条件信息已融入y_t的计算
            error = omega - actions - y_t  # ω - a_{1:H} - f_θ
            error_norm_sq = torch.sum(error **2, dim=[1,2], keepdim=True)  # L2范数平方（保留batch维度）
            
            # 步骤4：加权并累加（公式中的 -w(η)*||·||² 部分）
            step_log_prob = -w_eta * error_norm_sq
    
            # 步骤5：蒙特卡洛平均 + 公式中的1/2系数（近似对数似然下界）
            fm_log_prob = step_log_prob  * 0.5  # 忽略常数c（不影响优化）
            # 裁剪防止溢出（和原逻辑一致）
            # fm_log_prob = torch.clamp(fm_log_prob, min=-10.0, max=10.0)
            return fm_log_prob

        # ===================== 4. 重构Loss计算：AWR + Flow Matching似然 =====================
        # AWR核心：策略梯度损失 = -E[优势权重 * Flow Matching对数似然]
        awr_losses = 0.0
        device = actions.device
        
        # 合并权重：AWR优势权重
        final_weights = weights_awr.unsqueeze(1).unsqueeze(2)
        
        suffix_out = suffix_out.to(dtype=torch.float32)

        # 动作投影（保留原有梯度检查点）
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)
        y_t = self._apply_checkpoint(action_out_proj_func, suffix_out)  # 对应Flow Matching的f_θ
        
        # 核心替换：用Flow Matching计算对数似然（替代原高斯分布log_prob）
        log_prob = compute_fm_log_likelihood(
            y_t=y_t,
            actions=actions,
            noise=noise,
            time=time,
            device=device,
        )

        # AWR Loss：-（优势权重 * Flow Matching对数似然）
        weighted_loss = - (final_weights * log_prob)

        # 最终Loss：求所有维度的均值（用于反向传播）
        total_loss = weighted_loss.mean()

        # ===================== 5. 完善train_info：添加Flow Matching关键指标 =====================
        train_info = {
            "weighted_loss": total_loss.item(),
            "awr_advantage_mean": advantages.mean().item(),  # 优势函数均值
            "awr_weights_mean": weights_awr.mean().item(),   # AWR权重均值
            "fm_log_prob_mean": log_prob.mean().item(),      # Flow Matching对数似然均值
        }

        return total_loss, train_info


    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond = [None, None],
            num_timesteps = 1,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            num_timesteps = 1,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

class PI0QValueModelPytorch(PI0Pytorch):
    IMAGE_KEYS = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        # "base_t0_rgb",
    )
    
    def __init__(self, config):
        super().__init__(config)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.progress_head = ProgressHead(paligemma_config.width,
                                          max_progress=0, 
                                          min_progress=-1000, 
                                          num_bins=100,
                                          hidden_sizes=(1024, 512, 512, 256))
        
    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train, image_keys=self.IMAGE_KEYS)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
    
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        returns = observation.returns
        assert returns is not None

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        x_t = actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward_pi0(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=None,#[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out.to(torch.float32)[:, -1, :]
        logits = self.progress_head(suffix_out)
        progress_labels = self.progress_head.progress_to_indices(returns)
        progress_log_probs = self.progress_head.get_log_prob(logits, progress_labels.squeeze())
        progress_loss = -torch.mean(progress_log_probs)

        loss = progress_loss

        acc = (torch.argmax(logits, dim=1) == progress_labels.squeeze(1)).to(torch.float32)
        soft_acc = torch.abs(self.progress_head.logits_to_progress(logits) - returns.squeeze(1))

        train_info = {
            "progress_loss": progress_loss.item(),
            "progress_logits": torch.mean(logits).item(),
            "progress_entropy": torch.mean(self.progress_head.get_entropy(logits)).item(),
            "progress_acc": acc.mean().item(),
            "progress_soft_acc": soft_acc.mean().item(),
            "progress_soft_acc_max": soft_acc.max().item(),
            "progress_soft_acc_min": soft_acc.min().item(),
        }
        return loss, train_info
