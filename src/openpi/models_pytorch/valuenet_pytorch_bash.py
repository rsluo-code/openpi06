# ---------------------------------------------------------
# valuenet_pytorch.py
# PI0.6 / RECAP 的 Value 网络实现（对应论文 Eq.(1)）
#
# 思路：
#   - 完全复用 pi0 的视觉 + 文本前缀编码（SigLIP + PaliGemma）
#   - 调用 PaliGemmaWithExpertModel 的 forward，只喂 prefix（images + tokens）
#   - 不进入动作 DiT，只取 prefix 的最后一层 hidden state
#   - 对所有 prefix token 做 masked mean pooling 得到一个 [B, D] 表示整个观察
#   - 用一个 Linear(D -> num_bins) 输出 201 维 logits（value 的离散分布）
#
# 特别说明：
#   - Pi05 格式下，机器人 state 已经在 tokenize 阶段被离散化并拼到 prompt 里，
#     所以这里不再单独处理 state，直接把它当作文本 token 的一部分即可。
# ---------------------------------------------------------

import math
from typing import Tuple

import torch
from torch import Tensor, nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


# ---------------------------------------------------------
# 从 pi0_pytorch 复制的 attention mask 构造函数
# ---------------------------------------------------------
def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """
    从 pad_masks（哪些位置是真实 token）和 att_masks（前缀/自回归结构）
    构造二维 attention mask，逻辑与 pi0_pytorch 完全一致。

    pad_masks: [B, N]，True/1 表示该位置是有效 token，False/0 表示 padding
    att_masks: [B, N]，整数/布尔，控制自回归结构
    返回:
        att_2d_masks: [B, N, N]，True 表示可以相互注意
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class ValueNetPytorch(nn.Module):
    """
    PI0.6 / RECAP 中的 Value 网络（对应论文 Eq.(1)）

    输入：Observation（结构与 pi0 完全一致）
      - 多视角图像 images
      - 文本 token（prompt / 指令）
      - （Pi05 情况下）state 已经被离散化后拼接到 prompt 里，
        所以这里不再单独处理 state。

    输出：
      - [B, num_bins] 的 logits（默认 201 个 bin），表示回报的离散分布。
        后续你可以：
          - 用 CrossEntropy 与离散 label（0 ~ 200）训练
          - 推理时对 softmax 概率取期望，反推 [-1, 0] 区间上的 value
    """

    def __init__(self, config, num_bins: int = 201):
        super().__init__()

        self.config = config
        self.num_bins = num_bins

        # --------------------------------------------------------------
        # 1. 读取与 pi0 相同的 paligemma / action_expert 配置
        #    注意：ValueNet 不需要动作 expert 的 DiT 部分，但 PaliGemmaWithExpertModel
        #    内部是把 VLM + LLM 封在一起的，这里依然按 pi0 的方式构造。
        # --------------------------------------------------------------
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 对比 pi0_pytorch：
        #   pi0 用 use_adarms=[False, True] (pi05 时) 作为动作 DiT 的时间条件；
        #   ValueNet 不做动作预测，所以统一设为 [False, False]。
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],
            precision=config.dtype,
        )

        # PaliGemma 文本隐藏维度（即 transformer 的 hidden_size）
        hidden_dim = paligemma_config.width

        # --------------------------------------------------------------
        # 2. Value head：将 pooled prefix 特征映射到 num_bins 维 logits
        # --------------------------------------------------------------
        self.value_head = nn.Linear(hidden_dim, num_bins)

        # 和 pi0 保持一致，提升 matmul 精度表现
        torch.set_float32_matmul_precision("high")

        # 是否启用 gradient checkpointing（由外部脚本设置）
        self.gradient_checkpointing_enabled = False

        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")


    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True


    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        print("in sample_actions")
        # import pdb; pdb.set_trace()

        logits = self.forward(observation=observation)

        return logits

    # ==================================================================
    # Helper：梯度 checkpoint 封装（与 pi0 一致）
    # ==================================================================
    def _apply_checkpoint(self, func, *args, **kwargs):
        """
        在训练模式且 gradient_checkpointing_enabled=True 时，
        使用 torch.utils.checkpoint.checkpoint 包一层，以减小显存占用。
        """
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        """
        将 [B, N, N] 的 2D attention mask 转成 transformer 需要的 4D mask：
            [B, 1, N, N]，True 位置为 0，False 位置为 -inf（一个大负数）。
        与 pi0_pytorch 完全一致。
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # 这里用与 pi0 相同的大负数 -2.3819763e38（近似 -inf）
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    # ==================================================================
    # 预处理：完全复用官方的 preprocess_observation_pytorch
    # ==================================================================
    def _preprocess_observation(self, observation, *, train: bool = True):
        """
        调用 openpi 自带的预处理函数，保证和 pi0 的 observation 处理完全一致。
        返回：
            images      : List[Tensor[B, C, H, W]]
            image_masks : List[Tensor[B]]
            lang_tokens : Tensor[B, T_txt]
            lang_masks  : Tensor[B, T_txt]
        """
        obs = _preprocessing.preprocess_observation_pytorch(observation, train=train)

        images = list(obs.images.values())
        image_masks = list(obs.image_masks.values())
        lang_tokens = obs.tokenized_prompt
        lang_masks = obs.tokenized_prompt_mask
        # 注意：Pi05 情况下，state 已经在 tokenize 阶段被编码进 lang_tokens 里，
        # 这里就不再单独返回 state。

        return images, image_masks, lang_tokens, lang_masks

    # ==================================================================
    # 前缀 embedding：图像 + 文本 → token embeddings + padding/attention masks
    # 这部分直接照抄 pi0 的 embed_prefix，只是封装成 ValueNet 的方法。
    # ==================================================================
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        返回：
            embs      : [B, N_total, D]  （图像 token + 文本 token 的拼接）
            pad_masks : [B, N_total]     （1 表示有效，0 表示 padding）
            att_masks : [B, N_total]     （用于构造自回归 / prefix-lm 的 attention）
        """
        embs = []
        pad_masks = []
        att_masks = []

        # ---------- 图像 embedding ----------
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(x):
                return self.paligemma_with_expert.embed_image(x)

            img_emb = self._apply_checkpoint(image_embed_func, img)  # [B, N_img, D]
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # image token 之间全互相可见
            att_masks += [0] * num_img_embs

        # ---------- 文本 embedding ----------
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)  # [B, T_txt, D]
            lang_emb_dim = lang_emb.shape[-1]
            # 与 pi0 一致，对文本 embedding 做 sqrt(dim) 的缩放
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像 + 文本之间 full attention
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # 拼接所有 token
        embs = torch.cat(embs, dim=1)           # [B, N_total, D]
        pad_masks = torch.cat(pad_masks, dim=1) # [B, N_total]

        # att_masks 是 1D list，这里变成 [B, N_total]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    # ==================================================================
    # 带 mask 的 mean pooling（对 transformer 输出进行池化）
    # ==================================================================
    @staticmethod
    def _masked_mean_pool(embs: Tensor, pad_masks: Tensor) -> Tensor:
        """
        embs      : [B, N, D]  transformer 输出的 token 特征
        pad_masks : [B, N]     1 表示有效 token，0 表示 padding

        返回：
            pooled : [B, D]    每个样本一个全局特征
        """
        pad = pad_masks.to(embs.dtype).unsqueeze(-1)   # [B, N, 1]
        embs_masked = embs * pad                       # padding 位置清零
        sum_embs = embs_masked.sum(dim=1)              # [B, D]
        denom = pad.sum(dim=1).clamp_min(1e-6)         # [B, 1] 防止除 0
        pooled = sum_embs / denom                      # [B, D]
        return pooled

    # ==================================================================
    # forward：Observation → value logits（201-bin）
    # ==================================================================
    def forward(self, observation) -> Tensor:
        """
        参数：
            observation: 与 pi0 相同的 Observation 对象或 dict
                         （能被 preprocess_observation_pytorch 接受）

        返回：
            logits: [B, num_bins]，每一维代表一个 return bin 的 logit
        """
        # 1. 预处理：得到多视角图像、mask 和 token
        images, img_masks, lang_tokens, lang_masks = self._preprocess_observation(
            observation,
            train=self.training,
        )

        # 2. 前缀 embedding（SigLIP + 文本 embedding）
        prefix_embs, pad_masks, att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )  # [B, N, D], [B, N], [B, N]

        # 3. 构造 attention mask / position_ids（与 pi0 完全一致）
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)         # [B, N, N]
        position_ids = torch.cumsum(pad_masks, dim=1) - 1              # [B, N]
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)  # [B, 1, N, N]

        att_2d_masks_4d = att_2d_masks_4d.to(prefix_embs.dtype)

        # 4. 通过 PaliGemma 的文本 transformer，仅使用 prefix 通路
        def forward_func(prefix_embs, attention_mask, position_ids):
            # PaliGemmaWithExpertModel.forward 返回：
            #   [prefix_output, suffix_output], prefix_past_key_values
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],  # 只喂 prefix，不喂动作 suffix
                use_cache=False,
                adarms_cond=None,      # ✅ ValueNet 不用 AdaRMS，传 None
                num_timesteps=0,       # ✅ 只走 prefix，不用 expert/时间步，给个 0 就行
            )
            return prefix_out  # [B, N, D]

        prefix_out = self._apply_checkpoint(
            forward_func, prefix_embs, att_2d_masks_4d, position_ids
        )  # [B, N, D]

        # 5. 对 prefix 的输出做 masked mean pooling，得到每个样本一个表示
        pooled = self._masked_mean_pool(prefix_out, pad_masks)  # [B, D]

        # 6. 注意 dtype：PaliGemma 可能是 bfloat16，而 Linear 的权重是 float32，
        #    这里显式升成 float32 再过 value_head，避免 dtype mismatch。
        pooled = pooled.to(torch.float32)

        # 7. Linear 投影到 201 维 logits（value 分类）
        logits = self.value_head(pooled)  # [B, num_bins]

        return logits


    pretrain_or_finetuning = "pretrain"
    '''
    假设 bins 在 [-1, 0]，模型输出了这样一个分布：
    bin value	prob
    -0.9	0.49
    -0.1	0.51
    argmax（False）
    → 选 -0.1

    expectation（True）
    → 0.49*(-0.9) + 0.51*(-0.1) = -0.49


    argmax：偏向“最可能的成功”

    expectation：被“失败那一侧的概率尾巴”强烈拉低
    '''
    use_expectation = False
    action_N = 30

    def logits_to_value(self,logits):
        # logits: [B, 201]  或者 [201]
        # 确保 logits 至少是 [B,201]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)   # [1,201]
        # probs: [B, 201] 或 [201]
        probs = torch.softmax(logits, dim=-1)

       

        # Rt1: [B] 或标量
        if self.use_expectation==True:
            value = (probs * self.bin_centers).sum(dim=-1) 
        else:
            # idx: [B] 或标量
            idx = torch.argmax(probs, dim=-1)
            value = self.bin_centers[idx]
        # import pdb; pdb.set_trace()
        return value

    def forward_cal_value(self,observation) -> Tensor:
        logits_t = self.forward(observation)
        value_t = self.logits_to_value(logits_t)
        return value_t

    def forward_cal_At(self, observation_t,observation_tN, step_index, episode_length, language_instruction_max_len) -> Tensor:
        value_t = self.forward_cal_value(observation_t)
        value_tN = self.forward_cal_value(observation_tN)
        device = next(self.parameters()).device
        step_index_t = torch.as_tensor(step_index, device=device, dtype=torch.float32)        # [B]，范围 0..L-1
        episode_length_t = torch.as_tensor(episode_length, device=device, dtype=torch.float32)  # [B]，每个都是 L
        lang_max_len_t = torch.as_tensor(language_instruction_max_len, device=device, dtype=torch.float32)
        denom = torch.clamp(lang_max_len_t, min=1.0)
        # 公式：R = -1 + (step_index_t - 1) / (L - 1)
        # Rt = -1.0 + (step_index_t - 1.0) / denom               # [B] ∈ [-1, 0]
        # --------------------------------------------------
        # 核心修改：替换Rt的计算逻辑
        # 公式：ratio = (episode_length_t - step_index_t) / language_instruction_max_len
        # Rt = -ratio → 占比越大（ratio→1），Rt→-1；占比越小（ratio→0），Rt→0
        # --------------------------------------------------
        # 处理边界：防止language_instruction_max_len为0导致除0，最小设为1.0
        denom = torch.clamp(lang_max_len_t, min=1.0)  # 替换原来的denom计算
        # 计算核心占比：(episode_length - step_index) / language_instruction_max_len
        ratio = (episode_length_t - (step_index_t + 1)) / denom
        ratioN = self.action_N / denom
        # 映射到[-1, 0]：ratio∈[0,1] → Rt∈[-1, 0]（如果ratio超出[0,1]，用clamp限制）
        Rt =  -1*torch.clamp(ratio, min=0.0, max=1.0)  
        RtN= -1*torch.clamp(ratioN, min=0.0, max=1.0)  

        if self.pretrain_or_finetuning == "pretrain":
            At = Rt - value_t
        elif self.pretrain_or_finetuning == "finetuning":
            At = RtN + value_tN - value_t
        else:
            At = Rt - value_t
        # import pdb; pdb.set_trace()

        return {
            "At":At,
            "value_t":value_t,
            "value_tN":value_tN,
            "Rt":Rt,
            "RtN":RtN,
        }

    def forward_cal_indicator(self, observation_t,observation_tN, step_index, episode_length, language_instruction_max_len,language_instruction_at_30precent) -> Tensor:

        forward_cal_At_result = self.forward_cal_At(observation_t,observation_tN, step_index, episode_length, language_instruction_max_len)
        At= forward_cal_At_result["At"]
        indicator = (At>language_instruction_at_30precent).long()
        return indicator
        