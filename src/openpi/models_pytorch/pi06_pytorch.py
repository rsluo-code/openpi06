import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import time as sys_time
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

import re
import sentencepiece
import openpi.shared.download as download

PATH_paligemma_tokenizer_model = "/home/rsluo/codes/openpi06/src/tokenizer_model/paligemma_tokenizer.model"
# PATH_paligemma_tokenizer_model = "gs://big_vision/paligemma_tokenizer.model"
path = download.maybe_download(PATH_paligemma_tokenizer_model, gs={"token": "anon"})
with path.open("rb") as f:
    _tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
_IND_RE = re.compile(r",\s*Indicator:\s*\[INDICATOR\]\s*")   # 删除 Indicator: [INDICATOR]
_STATE_RE = re.compile(r",\s*State:\s*")                     # 找到 ", State:"



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


def load_checkpioint_for_value_net(model, ckpt_dir: str):
    import os
    ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
    print(f"[ValueNet] Loading Pi05 prefix-only weights from: {ckpt_path}")

    from safetensors.torch import load_file
    # 1. 读 ckpt
    src_sd = load_file(ckpt_path, device="cpu")

    # 2. 取出当前模型的 state_dict（注意 DDP 的情况）
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    target = model.module if is_ddp else model
    tgt_sd = target.state_dict()

    # 3. 只保留 paligemma_with_expert.paligemma.* 并且在当前模型中存在且 shape 一致的参数
    filtered_sd = {}
    for k, v in src_sd.items():

        # 当前 ValueNet 里必须也有同名参数，并且 shape 一致
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            print(f"pi06 valuenet({k}):  ckpt.shape {v.shape} !=  tgt_sd.shape {tgt_sd[k].shape}")

    print(f"[ValueNet] Will load {len(filtered_sd)} parameters")

    # 4. 只用 filtered_sd 来覆盖（strict=False 可以忽略没有加载到的 value_head 等）
    missing, unexpected = target.load_state_dict(filtered_sd, strict=False)

    print(f"[ValueNet] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[ValueNet] missing key: {missing}")
    if unexpected:
        print(f"[ValueNet] unexpected keys from ckpt (已忽略): {unexpected}")

class PI06Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)


##########

        self.max_bsize = 128
        # 提前初始化内存池张量
        # self.indicator_pool = torch.zeros(
        self.indicator_pool = torch.ones(
            (self.max_bsize, 1),  # 固定形状，避免动态变化
            dtype=torch.long,    
            device="cpu"   
        )
##########
            

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
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None
        
        #####
        if config.if_use_valuenet == True:
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config
            valuenet_config = _config.get_config("value_pretrain_16dim") 
            # valuenet_checkpoint_dir = "/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260128_2b_8dim_noexchangimg_only5item/45000"
            valuenet_checkpoint_dir = "/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260413/80000"
            self.value_net = _policy_config.create_trained_policy(valuenet_config, valuenet_checkpoint_dir)
            print("config.if_use_valuenet == True")
        else:
            self.value_net =None
            print("config.if_use_valuenet == False")

        # import openpi
        # from openpi.models_pytorch.valuenet_pytorch import ValueNetPytorch
        # valuenet_model_cfg = openpi.models.pi0_config.Pi0Config(
        #     action_horizon=30, 
        #     paligemma_variant="gemma_2b", 
        #     action_expert_variant="gemma_300m",
        #     pi05=True,
        # )
        # self.value_net = ValueNetPytorch(valuenet_model_cfg,num_bins=201)
        # load_checkpioint_for_value_net(self.value_net,"/wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/pi06_torch/value_pretrain_16dim/valuenet_pretrain_20260128_2b_8dim_noexchangimg_only5item/45000")
        # # 冻结 value_net 的所有参数
        # for param in self.value_net.parameters():
        #     param.requires_grad = False
        # from openpi.models_pytorch.some_func import print_parameter_stats
        # print_parameter_stats(self.value_net,"pi06_value_net")
        # self.value_net.eval()

        #####

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
        self, images, img_masks, lang_tokens, lang_masks,indicator=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if indicator is None:
            bsize = lang_tokens.shape[0]
            indicator = self.indicator_pool[:bsize].to(lang_tokens.device)
        # import pdb; pdb.set_trace()

        # foo = getattr(self, "suffix_tensor_pi06", None)   # 不存在就返回 None
        # if foo is None:
        #     device, dtype = lang_tokens.device, lang_tokens.dtype
        #     self.suffix_tensor_pi06 = torch.tensor([108, 4022, 235292, 235248], device=device, dtype=dtype) 
        #     self.pat0_tensor_pi06 = torch.tensor([105985, 235292, 6222, 235289, 108, 4022, 235292, 235248], device=device, dtype=dtype) 
        #     self.pat1_tensor_pi06 = torch.tensor([105985, 235292, 8322, 235289, 108, 4022, 235292, 235248], device=device, dtype=dtype)
        # lang_tokens, lang_masks = self.overwrite_tail_by_suffix(
        #     lang_tokens, lang_masks, indicator,
        #     suffix=self.suffix_tensor_pi06,
        #     pat0=self.pat0_tensor_pi06, pat1=self.pat1_tensor_pi06
        # )
        lang_tokens, lang_masks = self.rewrite_lang_tokens_with_advantage(lang_tokens, lang_masks, indicator, L=200)
        # import pdb; pdb.set_trace()
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


    import torch
    from typing import Optional, Sequence, Tuple

    def overwrite_tail_by_suffix(
        self,
        lang_tokens: torch.Tensor,          # [B, L]
        lang_masks: torch.Tensor,           # [B, L] bool
        Indicator: torch.Tensor,            # [B] 0/1
        suffix: Sequence[int] | torch.Tensor,
        pat0: Optional[Sequence[int] | torch.Tensor] = None,
        pat1: Optional[Sequence[int] | torch.Tensor] = None,
        insert0: Optional[Sequence[int] | torch.Tensor] = None,
        insert1: Optional[Sequence[int] | torch.Tensor] = None,
        check_suffix: bool = True,
        check_padding_zeros: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        认为每行有效 token 的末尾是 suffix（长度 S），并把：
            [最后 S 个有效 token] + [后面 K 个 padding(通常是0)]  共 M=S+K 个位置
        覆盖成 pat0 或 pat1（由 Indicator 决定）。

        你可以：
        - 直接提供 pat0/pat1（长度 M）
        - 或提供 insert0/insert1（长度 K），函数会自动 pat = insert + suffix

        mask 会把有效长度增加 K（即把 K 个 padding 变成有效 token）。
        """
        assert lang_tokens.shape == lang_masks.shape
        assert lang_tokens.dim() == 2
        B, L = lang_tokens.shape
        device, dtype = lang_tokens.device, lang_tokens.dtype

        # --- suffix tensor ---
        suffix_t = suffix if isinstance(suffix, torch.Tensor) else torch.tensor(suffix, device=device, dtype=dtype)
        suffix_t = suffix_t.to(device=device, dtype=dtype).flatten()
        S = int(suffix_t.numel())
        if S <= 0:
            raise ValueError("suffix 不能为空")

        # --- build pat0/pat1 ---
        if pat0 is None or pat1 is None:
            if insert0 is None or insert1 is None:
                raise ValueError("请提供 (pat0, pat1) 或者 (insert0, insert1) 其中一组参数")
            ins0 = insert0 if isinstance(insert0, torch.Tensor) else torch.tensor(insert0, device=device, dtype=dtype)
            ins1 = insert1 if isinstance(insert1, torch.Tensor) else torch.tensor(insert1, device=device, dtype=dtype)
            ins0 = ins0.to(device=device, dtype=dtype).flatten()
            ins1 = ins1.to(device=device, dtype=dtype).flatten()
            if ins0.numel() != ins1.numel():
                raise ValueError("insert0 和 insert1 长度必须一致")
            K = int(ins0.numel())
            pat0_t = torch.cat([ins0, suffix_t], dim=0)
            pat1_t = torch.cat([ins1, suffix_t], dim=0)
        else:
            pat0_t = pat0 if isinstance(pat0, torch.Tensor) else torch.tensor(pat0, device=device, dtype=dtype)
            pat1_t = pat1 if isinstance(pat1, torch.Tensor) else torch.tensor(pat1, device=device, dtype=dtype)
            pat0_t = pat0_t.to(device=device, dtype=dtype).flatten()
            pat1_t = pat1_t.to(device=device, dtype=dtype).flatten()
            if pat0_t.numel() != pat1_t.numel():
                raise ValueError("pat0 和 pat1 长度必须一致")
            M = int(pat0_t.numel())
            if M < S:
                raise ValueError(f"pat 长度(M={M}) 不能小于 suffix 长度(S={S})")
            K = M - S

        M = int(pat0_t.numel())  # 总替换长度 = S + K

        # 可选：检查 pat0/pat1 的末尾确实是 suffix（更符合你“最后有效是suffix”的语义）
        if check_suffix and M >= S:
            if not torch.equal(pat0_t[-S:], suffix_t) or not torch.equal(pat1_t[-S:], suffix_t):
                raise ValueError("pat0/pat1 的末尾不是 suffix；如果你不需要这个约束可把 check_suffix=False")

        # --- lengths ---
        lengths = lang_masks.sum(dim=1).long()  # [B] 有效长度
        if (lengths < S).any():
            bad = (lengths < S).nonzero(as_tuple=False).squeeze(1).tolist()
            raise ValueError(f"这些行有效长度 < suffix长度(S={S})：{bad}")

        if ((lengths + K) > L).any():
            bad = ((lengths + K) > L).nonzero(as_tuple=False).squeeze(1).tolist()
            raise ValueError(f"这些行末尾没有足够 K={K} 个位置可覆盖（length+K>{L}）：{bad}")

        # --- positions to overwrite ---
        start = lengths - S  # suffix 起点
        posM = start[:, None] + torch.arange(M, device=device)[None, :]  # [B, M]

        # --- check current suffix matches ---
        if check_suffix:
            b = torch.arange(B, device=device)
            old_suffix = lang_tokens[b[:, None], posM[:, :S]]
            ok = (old_suffix == suffix_t[None, :]).all(dim=1)
            if (~ok).any():
                bad = (~ok).nonzero(as_tuple=False).squeeze(1).tolist()
                raise ValueError(f"这些行末尾 suffix 不匹配：{bad}")

        # 可选：检查 suffix 后面 K 个确实是 padding 0（更严格）
        if check_padding_zeros and K > 0:
            b = torch.arange(B, device=device)
            pad_part = lang_tokens[b[:, None], posM[:, S:]]  # [B, K]
            ok0 = (pad_part == 0).all(dim=1)
            if (~ok0).any():
                bad = (~ok0).nonzero(as_tuple=False).squeeze(1).tolist()
                raise ValueError(f"这些行 suffix 后面的 K={K} 个位置不是全0 padding：{bad}")

        # --- select pattern by Indicator ---
        ind = Indicator.to(device=device).to(torch.bool)  # [B]
        pattern = torch.where(ind[:, None], pat1_t[None, :], pat0_t[None, :])  # [B, M]

        # --- write ---
        out_tokens = lang_tokens.clone()
        b = torch.arange(B, device=device)
        out_tokens[b[:, None], posM] = pattern

        # --- update mask: 增加 K 个有效 token ---
        new_lengths = lengths + K
        out_masks = torch.arange(L, device=device)[None, :] < new_lengths[:, None]

        return out_tokens, out_masks

    def rewrite_lang_tokens_with_advantage(
        self,
        lang_tokens: torch.Tensor,      # [B, L]
        lang_masks: torch.Tensor,       # [B, L] bool
        Indicator: torch.Tensor,        # [B] 0/1
        L: int = 200,
        add_bos: bool = True,
        strict: bool = True,            # encode后超长是否报错
    ):
        assert lang_tokens.shape == lang_masks.shape
        B, L0 = lang_tokens.shape
        assert L0 == L, f"expect L={L}, got {L0}"

        out_tokens = torch.zeros_like(lang_tokens)
        out_masks  = torch.zeros_like(lang_masks)

        bos_id = _tokenizer.bos_id() if hasattr(_tokenizer, "bos_id") else -1
        eos_id = _tokenizer.eos_id() if hasattr(_tokenizer, "eos_id") else -1

        for b in range(B):
            # # 取有效 token ids
            # n = int(lang_masks[b].sum().item())
            # ids = lang_tokens[b, :n].tolist()
            
            # 1. 移除 .item()，保留张量类型的求和结果
            sum_mask = lang_masks[b].sum()  # 形状：[]（0维张量），值为有效token数
            # 2. 转为整型张量（确保切片索引为整数类型）
            n = sum_mask.to(dtype=torch.int64)
            
            # 3. 关键修改：用张量作为切片结束位置（PyTorch支持张量切片）
            # 切片 lang_tokens[b, :n] 会自动适配张量n，无需转为Python int
            ids_tensor = lang_tokens[b, :n]
            # 4. 直接对张量调用 .tolist()，避免中间标量转换
            ids = ids_tensor.tolist()


            # 为了让字符串替换更稳定，decode 前把 bos/eos 去掉
            if bos_id != -1 and len(ids) > 0 and ids[0] == bos_id:
                ids = ids[1:]
            if eos_id != -1 and len(ids) > 0 and ids[-1] == eos_id:
                ids = ids[:-1]

            text = _tokenizer.decode(ids)   # 得到原 prompt 字符串（大概率含 Task/State/Indicator）
            # print(f"text1={text}")
            # 1) 删除 ", Indicator: [INDICATOR]"
            text2 = _IND_RE.sub("", text, count=1)

            # 2) 在 ", State:" 前插入 ", Advantage: positive/negative"
            adv = "positive" if int(Indicator[b].item()) == 1 else "negative"

            m = _STATE_RE.search(text2)
            if m is None:
                # 找不到 State 段，按需处理：严格模式报错，否则不改这条
                if strict:
                    raise ValueError(f"Row {b}: cannot find ', State:' in decoded text:\n{text}")
                else:
                    text3 = text2
            else:
                insert_pos = m.start()
                text3 = text2[:insert_pos] + f", Advantage: {adv}" + text2[insert_pos:]

            # print(f"text3={text3}")
            # 3) encode 回去（保持末尾仍是 ";\nAction: " 这种原始换行）
            new_ids = _tokenizer.encode(text3, add_bos=add_bos)

            # 4) pad 到 200
            if len(new_ids) > L:
                if strict:
                    raise ValueError(f"Row {b}: encoded length {len(new_ids)} > L={L}")
                new_ids = new_ids[:L]

            out_tokens[b, :len(new_ids)] = torch.tensor(new_ids, device=lang_tokens.device, dtype=lang_tokens.dtype)
            out_masks[b, :len(new_ids)] = True

        return out_tokens, out_masks


    def forward(self, observation,observation_tN, actions, step_index, episode_length,language_instruction_index,language_instruction_max_len,success_or_failure ,language_instruction_at_30precent, noise=None, time=None) -> Tensor:
        
        with torch.no_grad():
            if self.value_net is not None:
                # cal_i_start_time = sys_time.time()
                indicator = self.value_net._model.forward_cal_indicator( observation,observation_tN, step_index, episode_length, language_instruction_max_len,language_instruction_at_30precent)
                # cal_i_end_time = sys_time.time()
                # print(f"cal_i_use_time:{(cal_i_end_time-cal_i_start_time):.3f}s")
            else:
                bsize = observation.state.shape[0]
                indicator = self.indicator_pool[:bsize].to(observation.state.device)

        # import pdb; pdb.set_trace()

        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        

        

        


        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks,indicator=indicator)

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
                if actions[batch_idx, step_idx, 7] <= 0.00019904:  #(x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0  #夹爪0.5（LD）0.05(SL)
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

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        print("in pi06 sample actions")
        # w = self.success_token_linear.weight
        # b = self.success_token_linear.bias
        # print("weight shape:", w.shape)
        # print("weight dtype:", w.dtype)
        # print("weight device:", w.device)
        # print("weight mean:", w.float().mean().item())
        # print("weight std:",  w.float().std().item())
        # print("bias mean:", b.float().mean().item())

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
