import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

sys.path.append(str(CURRENT_DIR))
# print(sys.path)

from math import sqrt
from typing import Optional, Unpack, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.cache_utils import Cache
from transformers import Qwen2_5_VLForConditionalGeneration
from quaternion.quaternion_layers import QuaternionLinear


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)
        self.num_head = num_head
        self.d = dim // num_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        q = (
            self.Wq(query)
            .reshape(query.shape[0], query.shape[1], self.num_head, self.d)
            .transpose(1, 2)
        )
        k = (
            self.Wk(key)
            .reshape(key.shape[0], key.shape[1], self.num_head, self.d)
            .transpose(1, 2)
        )
        v = (
            self.Wv(value)
            .reshape(value.shape[0], value.shape[1], self.num_head, self.d)
            .transpose(1, 2)
        )

        attn_score = (q @ k.transpose(-1, -2)) / sqrt(key.shape[-1])
        if mask is not None:
            attn_score.masked_fill(mask == 0, -1e9)
        attn_score = attn_score.softmax(dim=-1)
        attn = attn_score @ v
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(attn.shape[0], -1, self.d * self.num_head)
        )
        return self.dropout(self.Wo(attn))


class CrossAttention(nn.Module):
    def __init__(self, init_weights=True):
        super().__init__()
        self.msa = MultiHeadSelfAttention(dim=1280, num_head=8, dropout=0.1)
        self.quaternion = nn.Sequential(
            QuaternionLinear(in_features=1280, out_features=1280, seed=42),
            nn.Dropout(p=0.1),
            nn.GELU(),
        )
        self.norm1 = nn.LayerNorm(1280)
        self.norm2 = nn.LayerNorm(1280)
        if init_weights:
            self.init_weights()

    def forward(self, visual, text):
        visual, text = self.norm1(visual), self.norm1(text)
        output = self.msa(visual, text, text)
        output += self.quaternion(self.norm2(output))
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class CrossBlock(nn.Module):
    def __init__(self, num_depth):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttention() for _ in range(num_depth)])

    def forward(self, visual, text):
        x = visual
        for cross in self.layers:
            x = cross(x, text)
        return x


class TextEmbeddingProjection(nn.Module):
    def __init__(self, num_block=1, num_depth=1, init_weights=True):
        super().__init__()
        d = 2048
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(
            nn.Linear(in_features=d, out_features=1280), nn.Dropout(p=0.1), nn.GELU()
        )
        self.block = CrossBlock(num_depth=num_depth)
        if init_weights:
            self.init_weights()

    def forward(self, visual, text):
        text = self.proj(self.norm(text))
        return self.block(visual=visual, text=text)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class NewViT(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.inject_layer = config.text_cross_config["layer"]
        self.factor = config.text_cross_config["factor"]
        self.text_projection = nn.ModuleList(
            [
                TextEmbeddingProjection(num_depth=config.text_cross_config["depth"])
                for _ in range(len(self.inject_layer))
            ]
        )
        self.vision_layer = config.vision_layer_config["layer"]
        self.vision_merge = nn.Sequential(
            nn.Linear(
                in_features=len(self.vision_layer), out_features=len(self.vision_layer)
            ),
            nn.GELU(),
            nn.Linear(in_features=len(self.vision_layer), out_features=1),
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states.requires_grad = True
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        vision_info = []
        for layer_num, blk in enumerate(self.blocks):

            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            if layer_num in self.inject_layer:
                index_ = self.inject_layer.index(layer_num)
                cross_info = self.text_projection[index_](
                    visual=hidden_states.unsqueeze(0), text=kwargs["text_info"]
                )
                hidden_states += self.factor * cross_info.squeeze(0)  # (n, 1280)

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if layer_num in self.vision_layer:
                vision_info.append(hidden_states)

        multi_feature = torch.stack(vision_info)  # (v_layer, n, 1280)
        multi_feature = multi_feature.permute(1, 2, 0)
        multi_feature = self.vision_merge(multi_feature).squeeze(-1)  # (n, 1280)
        alpha = 0.5
        hidden_states = (1 - alpha) * hidden_states + alpha * multi_feature
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class NewVLModel(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = NewViT._from_config(config.vision_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor],
        text_info: torch.Tensor,
    ):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(
            pixel_values, grid_thw=image_grid_thw, text_info=text_info
        )
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                text_info=inputs_embeds,
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(
                        inputs_embeds.device
                    )
                else:
                    delta = torch.zeros(
                        (batch_size, seq_length), device=inputs_embeds.device
                    )
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class NewQwen(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = NewVLModel(config)
