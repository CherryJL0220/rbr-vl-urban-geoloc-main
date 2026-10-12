# # coding=utf-8
# # Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# #
# # This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# # and OPT implementations in this library. It has been modified from its
# # original forms to accommodate minor architectural differences compared
# # to GPT-NeoX and OPT used by the Meta AI team that trained the model.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# from typing import Callable, List, Optional, Tuple, Union

# import torch
# import torch.utils.checkpoint
# from torch import nn

# from ...activations import ACT2FN
# from ...cache_utils import Cache, DynamicCache, StaticCache
# from ...generation import GenerationMixin
# from ...modeling_attn_mask_utils import AttentionMaskConverter
# from ...modeling_flash_attention_utils import FlashAttentionKwargs
# from ...modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutputWithPast,
#     TokenClassifierOutput,
# )
# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from ...processing_utils import Unpack
# from ...pytorch_utils import ALL_LAYERNORM_LAYERS
# from ...utils import (
#     LossKwargs,
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     logging,
#     replace_return_docstrings,
# )
# from ...utils.deprecation import deprecate_kwarg
# from .configuration_llama import LlamaConfig


# logger = logging.get_logger(__name__)

# _CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
# _CONFIG_FOR_DOC = "LlamaConfig"


# class LlamaRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         LlamaRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

#     def extra_repr(self):
#         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(self, config: LlamaConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     def _dynamic_frequency_update(self, position_ids, device):
#         """
#         dynamic RoPE layers should recompute `inv_freq` in the following situations:
#         1 - growing beyond the cached sequence length (allow scaling)
#         2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
#         """
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_seq_len_cached:  # growth
#             inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
#             self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
#             self.max_seq_len_cached = seq_len

#         if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
#             # This .to() is needed if the model has been moved to a device after being initialized (because
#             # the buffer is automatically moved, but not the original copy)
#             self.original_inv_freq = self.original_inv_freq.to(device)
#             self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
#             self.max_seq_len_cached = self.original_max_seq_len

#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         if "dynamic" in self.rope_type:
#             self._dynamic_frequency_update(position_ids, device=x.device)

#         # Core RoPE block
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()

#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         cos = cos * self.attention_scaling
#         sin = sin * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# class LlamaMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(self, x):
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)

#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
#     if attention_mask is not None:
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask

#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
#     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()

#     return attn_output, attn_weights


# class LlamaAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = self.head_dim**-0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = True

#         self.q_proj = nn.Linear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.v_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         past_key_value: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)

#         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
#                 logger.warning_once(
#                     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
#                     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#                 )
#             else:
#                 attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights


# class LlamaDecoderLayer(nn.Module):
#     def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

#         self.mlp = LlamaMLP(config)
#         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)
#         if output_attentions:
#             outputs += (self_attn_weights,)

#         return outputs


# LLAMA_START_DOCSTRING = r"""
#     This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
#     library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
#     etc.)

#     This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
#     Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
#     and behavior.

#     Parameters:
#         config ([`LlamaConfig`]):
#             Model configuration class with all the parameters of the model. Initializing with a config file does not
#             load the weights associated with the model, only the configuration. Check out the
#             [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# """


# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )
# class LlamaPreTrainedModel(PreTrainedModel):
#     config_class = LlamaConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["LlamaDecoderLayer"]
#     _skip_keys_device_placement = ["past_key_values"]
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_flex_attn = True
#     _supports_cache_class = True
#     _supports_quantized_cache = True
#     _supports_static_cache = True
#     _supports_attention_backend = True

#     def _init_weights(self, module):
#         std = self.config.initializer_range
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()


# LLAMA_INPUTS_DOCSTRING = r"""
#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#             [What are attention masks?](../glossary#attention-mask)

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
#             `past_key_values`).

#             If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
#             and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
#             information on the default strategy.

#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
#             config.n_positions - 1]`.

#             [What are position IDs?](../glossary#position-ids)
#         past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
#             Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
#             blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
#             returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

#             Two formats are allowed:
#             - a [`~cache_utils.Cache`] instance, see our
#             [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
#             - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
#             shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
#             cache format.

#             The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
#             legacy cache format will be returned.

#             If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
#             have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
#             of shape `(batch_size, sequence_length)`.
#         inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
#             is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
#             model's internal embedding lookup matrix.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#             Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
#             this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
#             the complete sequence length.
# """


# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )
# class LlamaModel(LlamaPreTrainedModel):
#     """
#     Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

#     Args:
#         config: LlamaConfig
#     """

#     # def __init__(self, config: LlamaConfig):
#     #     super().__init__(config)
#     #     self.padding_idx = config.pad_token_id
#     #     self.vocab_size = config.vocab_size
#     #
#     #     self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#     #     self.layers = nn.ModuleList(
#     #         [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#     #     )
#     #     self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#     #     self.rotary_emb = LlamaRotaryEmbedding(config=config)
#     #     self.gradient_checkpointing = False
#     #
#     #     # Initialize weights and apply final processing
#     #     self.post_init()

#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = LlamaRotaryEmbedding(config=config)
#         self.gradient_checkpointing = False

#         # FastV 配置
#         self.use_fast_v = getattr(config, "use_fast_v", False)
#         self.fast_v_sys_length = getattr(config, "fast_v_sys_length", None)
#         self.fast_v_image_token_length = getattr(config, "fast_v_image_token_length", None)
#         self.fast_v_attention_rank = getattr(config, "fast_v_attention_rank", None)
#         self.fast_v_agg_layer = getattr(config, "fast_v_agg_layer", None)
#         self.fast_v_inplace = getattr(config, "fast_v_inplace", False)

#         # 验证 FastV 参数
#         if self.use_fast_v:
#             if self.fast_v_agg_layer is not None:
#                 assert self.fast_v_agg_layer > 0, "fast_v_agg_layer should be > 0"
#             if self.fast_v_inplace:
#                 # inplace 模式下需要禁用缓存和梯度检查点
#                 logger.warning_once("FastV inplace mode: use_cache and gradient_checkpointing will be disabled.")
#             else:
#                 logger.warning_once("FastV mask mode: use_cache may be disabled if attention mask changes.")

#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     def reset_fastv(self):
#         """重置 FastV 配置（用于动态调整）"""
#         self.use_fast_v = self.config.use_fast_v
#         self.fast_v_sys_length = self.config.fast_v_sys_length
#         self.fast_v_image_token_length = self.config.fast_v_image_token_length
#         self.fast_v_attention_rank = self.config.fast_v_attention_rank
#         self.fast_v_agg_layer = self.config.fast_v_agg_layer
#         self.fast_v_inplace = self.config.fast_v_inplace

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#             self,
#             input_ids: torch.LongTensor = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values: Optional[Cache] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#             cache_position: Optional[torch.LongTensor] = None,
#             **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False

#         # FastV 预处理：禁用缓存和梯度检查点（避免复杂性）
#         if self.use_fast_v and self.fast_v_agg_layer is not None:
#             if use_cache:
#                 logger.warning_once("FastV is enabled, setting use_cache=False to avoid KV cache issues.")
#                 use_cache = False
#             if self.gradient_checkpointing and self.training:
#                 logger.warning_once("FastV is enabled, disabling gradient checkpointing.")
#                 self.gradient_checkpointing = False

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         if use_cache and past_key_values is None:
#             past_key_values = DynamicCache()

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )

#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)

#         causal_mask = self._update_causal_mask(
#             attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
#         )

#         hidden_states = inputs_embeds
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         # 用于 FastV 的变量
#         current_attention_mask = causal_mask
#         current_hidden_states = hidden_states
#         current_position_ids = position_ids
#         current_cache_position = cache_position
#         current_past_key_values = past_key_values
#         use_fast_v_this_run = self.use_fast_v and self.fast_v_agg_layer is not None
#         agg_layer = self.fast_v_agg_layer if use_fast_v_this_run else None

#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None

#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (current_hidden_states,)

#             # 决定是否在这一层输出注意力（仅当 idx == agg_layer 且启用 FastV 时需要）
#             output_attentions_this = output_attentions
#             if use_fast_v_this_run and idx == agg_layer:
#                 output_attentions_this = True

#             # 选择当前层使用的输入变量
#             if use_fast_v_this_run and idx >= agg_layer:
#                 # 后续层使用更新后的变量
#                 cur_attn_mask = current_attention_mask
#                 cur_pos_ids = current_position_ids
#                 cur_cache_pos = current_cache_position
#                 cur_past_kv = current_past_key_values
#             else:
#                 # 前 agg_layer 层使用原始变量
#                 cur_attn_mask = causal_mask
#                 cur_pos_ids = position_ids
#                 cur_cache_pos = cache_position
#                 cur_past_kv = past_key_values

#             # 调用解码器层
#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     current_hidden_states,
#                     cur_attn_mask,
#                     cur_pos_ids,
#                     cur_past_kv,
#                     output_attentions_this,
#                     use_cache,
#                     cur_cache_pos,
#                     position_embeddings,
#                     **flash_attn_kwargs,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     current_hidden_states,
#                     attention_mask=cur_attn_mask,
#                     position_ids=cur_pos_ids,
#                     past_key_value=cur_past_kv,
#                     output_attentions=output_attentions_this,
#                     use_cache=use_cache,
#                     cache_position=cur_cache_pos,
#                     position_embeddings=position_embeddings,
#                     **flash_attn_kwargs,
#                 )

#             current_hidden_states = layer_outputs[0]

#             # FastV 核心逻辑：在第 agg_layer 层利用注意力权重选择 token
#             if use_fast_v_this_run and idx == agg_layer:
#                 attn_weights = layer_outputs[1]  # 获取注意力权重

#                 # 计算平均注意力（对头平均，再对 batch 平均）
#                 # attn_weights shape: [batch, num_heads, q_len, kv_len]
#                 batch_size, num_heads, q_len, kv_len = attn_weights.shape
#                 # 平均 head 维度 -> [batch, q_len, kv_len]
#                 attn_avg_heads = torch.mean(attn_weights, dim=1)
#                 # 平均 batch 维度 -> [q_len, kv_len]
#                 attn_avg = attn_avg_heads.mean(dim=0)

#                 # 取最后一个 query 位置的注意力
#                 last_token_attn = attn_avg[-1]  # [kv_len]

#                 sys_len = self.fast_v_sys_length
#                 img_len = self.fast_v_image_token_length
#                 rank = self.fast_v_attention_rank

#                 # 图像 token 的注意力
#                 image_attn = last_token_attn[sys_len: sys_len + img_len]
#                 topk_indices = image_attn.topk(rank).indices + sys_len

#                 if self.fast_v_inplace:
#                     # 保留的 token 索引：系统提示 + 选中的图像 token + 文本 token
#                     keep_indices = torch.cat([
#                         torch.arange(sys_len, device=current_hidden_states.device),
#                         topk_indices,
#                         torch.arange(sys_len + img_len, kv_len, device=current_hidden_states.device)
#                     ]).sort().values
#                     new_seq_len = keep_indices.shape[0]

#                     # 裁剪 hidden_states
#                     current_hidden_states = current_hidden_states[:, keep_indices, :]

#                     # 更新 position_ids 和 cache_position
#                     # 注意：假设 batch_size = 1 或所有样本使用相同索引
#                     # 这里扩展到 batch_size
#                     current_position_ids = keep_indices.unsqueeze(0).expand(batch_size, -1)
#                     current_cache_position = torch.arange(new_seq_len, device=current_hidden_states.device)

#                     # 生成新的 attention_mask（2D 全 1）
#                     new_attn_mask_2d = torch.ones(
#                         (batch_size, new_seq_len), dtype=torch.bool, device=current_hidden_states.device
#                     )
#                     # 重新生成 4D causal mask
#                     current_attention_mask = self._update_causal_mask(
#                         new_attn_mask_2d,
#                         current_hidden_states,
#                         current_cache_position,
#                         None,  # past_key_values 置空
#                         output_attentions,
#                     )

#                     # 缓存已失效，禁用后续缓存
#                     use_cache = False
#                     current_past_key_values = None
#                 else:
#                     # 非 inplace 模式：生成新的 attention_mask，屏蔽不重要的图像 token
#                     new_attn_mask_2d = torch.ones(
#                         (batch_size, kv_len), dtype=torch.bool, device=current_hidden_states.device
#                     )
#                     new_attn_mask_2d[:, sys_len: sys_len + img_len] = False
#                     new_attn_mask_2d[:, topk_indices] = True

#                     current_attention_mask = self._update_causal_mask(
#                         new_attn_mask_2d,
#                         current_hidden_states,
#                         current_cache_position,
#                         current_past_key_values,
#                         output_attentions,
#                     )
#                     # position_ids 和 cache_position 保持不变
#                     # current_position_ids 和 current_cache_position 仍使用原始值

#                 # 本层已经处理，跳过后续的注意力输出收集（已在外部处理）
#                 # 但注意：当前层输出的 hidden_states 已经是更新后的值
#                 # 我们可能需要在收集输出时保留原始的输出，这里不重复收集

#             # 收集输出（仅当 use_cache 或 output_attentions 启用时）
#             if use_cache:
#                 # 注意：当前层可能已经将 past_key_values 置空，需要根据情况获取
#                 kv_out = layer_outputs[2 if output_attentions_this else 1]
#                 next_decoder_cache += (kv_out,)
#             if output_attentions:
#                 # 只在需要时收集注意力权重
#                 if output_attentions_this:
#                     all_self_attns += (layer_outputs[1],)
#                 else:
#                     # 如果不需要输出，但收集列表仍保持长度一致，可以填充 None
#                     # 简单起见，如果不输出就不加入
#                     pass

#             # 注意：在 FastV 非 inplace 模式下，我们未更新 current_position_ids 和 current_cache_position，
#             # 因此后续层仍使用原始值，但 attention_mask 已更改。
#             # 对于 inplace 模式，所有后续变量都已更新。

#         # 最终层归一化
#         current_hidden_states = self.norm(current_hidden_states)

#         if output_hidden_states:
#             all_hidden_states += (current_hidden_states,)

#         output = BaseModelOutputWithPast(
#             last_hidden_state=current_hidden_states,
#             past_key_values=next_decoder_cache if use_cache else None,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )
#         return output if return_dict else output.to_tuple()
#     # def forward(
#     #     self,
#     #     input_ids: torch.LongTensor = None,
#     #     attention_mask: Optional[torch.Tensor] = None,
#     #     position_ids: Optional[torch.LongTensor] = None,
#     #     past_key_values: Optional[Cache] = None,
#     #     inputs_embeds: Optional[torch.FloatTensor] = None,
#     #     use_cache: Optional[bool] = None,
#     #     output_attentions: Optional[bool] = None,
#     #     output_hidden_states: Optional[bool] = None,
#     #     return_dict: Optional[bool] = None,
#     #     cache_position: Optional[torch.LongTensor] = None,
#     #     **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
#     # ) -> Union[Tuple, BaseModelOutputWithPast]:
#     #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     #     output_hidden_states = (
#     #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     #     )
#     #     use_cache = use_cache if use_cache is not None else self.config.use_cache
#     #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#     #
#     #     if (input_ids is None) ^ (inputs_embeds is not None):
#     #         raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
#     #
#     #     if self.gradient_checkpointing and self.training and use_cache:
#     #         logger.warning_once(
#     #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#     #         )
#     #         use_cache = False
#     #
#     #     if inputs_embeds is None:
#     #         inputs_embeds = self.embed_tokens(input_ids)
#     #
#     #     if use_cache and past_key_values is None:
#     #         past_key_values = DynamicCache()
#     #
#     #     if cache_position is None:
#     #         past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#     #         cache_position = torch.arange(
#     #             past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#     #         )
#     #
#     #     if position_ids is None:
#     #         position_ids = cache_position.unsqueeze(0)
#     #
#     #     causal_mask = self._update_causal_mask(
#     #         attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
#     #     )
#     #
#     #     hidden_states = inputs_embeds
#     #
#     #     # create position embeddings to be shared across the decoder layers
#     #     position_embeddings = self.rotary_emb(hidden_states, position_ids)
#     #
#     #     # decoder layers
#     #     all_hidden_states = () if output_hidden_states else None
#     #     all_self_attns = () if output_attentions else None
#     #
#     #     for decoder_layer in self.layers[: self.config.num_hidden_layers]:
#     #         if output_hidden_states:
#     #             all_hidden_states += (hidden_states,)
#     #
#     #         if self.gradient_checkpointing and self.training:
#     #             layer_outputs = self._gradient_checkpointing_func(
#     #                 decoder_layer.__call__,
#     #                 hidden_states,
#     #                 causal_mask,
#     #                 position_ids,
#     #                 past_key_values,
#     #                 output_attentions,
#     #                 use_cache,
#     #                 cache_position,
#     #                 position_embeddings,
#     #             )
#     #         else:
#     #             layer_outputs = decoder_layer(
#     #                 hidden_states,
#     #                 attention_mask=causal_mask,
#     #                 position_ids=position_ids,
#     #                 past_key_value=past_key_values,
#     #                 output_attentions=output_attentions,
#     #                 use_cache=use_cache,
#     #                 cache_position=cache_position,
#     #                 position_embeddings=position_embeddings,
#     #                 **flash_attn_kwargs,
#     #             )
#     #
#     #         hidden_states = layer_outputs[0]
#     #
#     #         if output_attentions:
#     #             all_self_attns += (layer_outputs[1],)
#     #
#     #     hidden_states = self.norm(hidden_states)
#     #
#     #     # add hidden states from the last decoder layer
#     #     if output_hidden_states:
#     #         all_hidden_states += (hidden_states,)
#     #
#     #     output = BaseModelOutputWithPast(
#     #         last_hidden_state=hidden_states,
#     #         past_key_values=past_key_values if use_cache else None,
#     #         hidden_states=all_hidden_states,
#     #         attentions=all_self_attns,
#     #     )
#     #     return output if return_dict else output.to_tuple()

#     def _update_causal_mask(
#         self,
#         attention_mask: torch.Tensor,
#         input_tensor: torch.Tensor,
#         cache_position: torch.Tensor,
#         past_key_values: Cache,
#         output_attentions: bool,
#     ):
#         if self.config._attn_implementation == "flash_attention_2":
#             if attention_mask is not None and (attention_mask == 0.0).any():
#                 return attention_mask
#             return None

#         # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
#         # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
#         # to infer the attention mask.
#         past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#         using_static_cache = isinstance(past_key_values, StaticCache)

#         # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
#         if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
#             if AttentionMaskConverter._ignore_causal_mask_sdpa(
#                 attention_mask,
#                 inputs_embeds=input_tensor,
#                 past_key_values_length=past_seen_tokens,
#                 is_training=self.training,
#             ):
#                 return None

#         dtype, device = input_tensor.dtype, input_tensor.device
#         sequence_length = input_tensor.shape[1]
#         if using_static_cache:
#             target_length = past_key_values.get_max_cache_shape()
#         else:
#             target_length = (
#                 attention_mask.shape[-1]
#                 if isinstance(attention_mask, torch.Tensor)
#                 else past_seen_tokens + sequence_length + 1
#             )

#         # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
#         causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
#             attention_mask,
#             sequence_length=sequence_length,
#             target_length=target_length,
#             dtype=dtype,
#             device=device,
#             cache_position=cache_position,
#             batch_size=input_tensor.shape[0],
#         )

#         if (
#             self.config._attn_implementation == "sdpa"
#             and attention_mask is not None
#             and attention_mask.device.type in ["cuda", "xpu"]
#             and not output_attentions
#         ):
#             # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
#             # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
#             # Details: https://github.com/pytorch/pytorch/issues/110213
#             min_dtype = torch.finfo(dtype).min
#             causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

#         return causal_mask

#     @staticmethod
#     def _prepare_4d_causal_attention_mask_with_cache_position(
#         attention_mask: torch.Tensor,
#         sequence_length: int,
#         target_length: int,
#         dtype: torch.dtype,
#         device: torch.device,
#         cache_position: torch.Tensor,
#         batch_size: int,
#         **kwargs,
#     ):
#         """
#         Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
#         `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

#         Args:
#             attention_mask (`torch.Tensor`):
#                 A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
#                 `(batch_size, 1, query_length, key_value_length)`.
#             sequence_length (`int`):
#                 The sequence length being processed.
#             target_length (`int`):
#                 The target length: when generating with static cache, the mask should be as long as the static cache,
#                 to account for the 0 padding, the part of the cache that is not filled yet.
#             dtype (`torch.dtype`):
#                 The dtype to use for the 4D attention mask.
#             device (`torch.device`):
#                 The device to plcae the 4D attention mask on.
#             cache_position (`torch.Tensor`):
#                 Indices depicting the position of the input sequence tokens in the sequence.
#             batch_size (`torch.Tensor`):
#                 Batch size.
#         """
#         if attention_mask is not None and attention_mask.dim() == 4:
#             # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
#             causal_mask = attention_mask
#         else:
#             min_dtype = torch.finfo(dtype).min
#             causal_mask = torch.full(
#                 (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
#             )
#             if sequence_length != 1:
#                 causal_mask = torch.triu(causal_mask, diagonal=1)
#             causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
#             causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
#             if attention_mask is not None:
#                 causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
#                 mask_length = attention_mask.shape[-1]
#                 padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
#                     causal_mask.device
#                 )
#                 padding_mask = padding_mask == 0
#                 causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
#                     padding_mask, min_dtype
#                 )

#         return causal_mask


# class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}
#     _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = LlamaModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def set_decoder(self, decoder):
#         self.model = decoder

#     def get_decoder(self):
#         return self.model

#     @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         **kwargs: Unpack[KwargsForCausalLM],
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#             logits_to_keep (`int` or `torch.Tensor`, *optional*):
#                 If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
#                 `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
#                 token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
#                 If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
#                 This is useful when using packed tensor format (single dimension for batch and sequence length).

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#             **kwargs,
#         )

#         hidden_states = outputs[0]
#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     The LLaMa Model transformer with a sequence classification head on top (linear layer).

#     [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-2) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """,
#     LLAMA_START_DOCSTRING,
# )
# class LlamaForSequenceClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = LlamaModel(config)
#         self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             last_non_pad_token = -1
#         elif input_ids is not None:
#             # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
#             non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
#             token_indices = torch.arange(input_ids.shape[-1], device=logits.device)
#             last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
#         else:
#             last_non_pad_token = -1
#             logger.warning_once(
#                 f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                 "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#             )

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @add_start_docstrings(
#     """
# The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
# SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     LLAMA_START_DOCSTRING,
# )
# class LlamaForQuestionAnswering(LlamaPreTrainedModel):
#     base_model_prefix = "transformer"

#     # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = LlamaModel(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.transformer.embed_tokens

#     def set_input_embeddings(self, value):
#         self.transformer.embed_tokens = value

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         start_positions: Optional[torch.LongTensor] = None,
#         end_positions: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[Tuple, QuestionAnsweringModelOutput]:
#         r"""
#         start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         loss = None
#         if start_positions is not None and end_positions is not None:
#             loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return QuestionAnsweringModelOutput(
#             loss=loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
#     output) e.g. for Named-Entity-Recognition (NER) tasks.
#     """,
#     LLAMA_START_DOCSTRING,
# )
# class LlamaForTokenClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = LlamaModel(config)
#         if getattr(config, "classifier_dropout", None) is not None:
#             classifier_dropout = config.classifier_dropout
#         elif getattr(config, "hidden_dropout", None) is not None:
#             classifier_dropout = config.hidden_dropout
#         else:
#             classifier_dropout = 0.1
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.score = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, TokenClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         logits = self.score(sequence_output)

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits, labels, self.config)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# __all__ = [
#     "LlamaForCausalLM",
#     "LlamaModel",
#     "LlamaPreTrainedModel",
#     "LlamaForSequenceClassification",
#     "LlamaForQuestionAnswering",
#     "LlamaForTokenClassification",
# ]

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    # def __init__(self, config: LlamaConfig):
    #     super().__init__(config)
    #     self.padding_idx = config.pad_token_id
    #     self.vocab_size = config.vocab_size
    #
    #     self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    #     self.layers = nn.ModuleList(
    #         [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    #     )
    #     self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    #     self.rotary_emb = LlamaRotaryEmbedding(config=config)
    #     self.gradient_checkpointing = False
    #
    #     # Initialize weights and apply final processing
    #     self.post_init()

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # FastV 配置
        self.use_fast_v = getattr(config, "use_fast_v", False)
        self.fast_v_sys_length = getattr(config, "fast_v_sys_length", None)
        self.fast_v_image_token_length = getattr(config, "fast_v_image_token_length", None)
        self.fast_v_attention_rank = getattr(config, "fast_v_attention_rank", None)
        self.fast_v_agg_layer = getattr(config, "fast_v_agg_layer", None)
        self.fast_v_inplace = getattr(config, "fast_v_inplace", False)

        # 验证 FastV 参数
        if self.use_fast_v:
            if self.fast_v_agg_layer is not None:
                assert self.fast_v_agg_layer > 0, "fast_v_agg_layer should be > 0"
            if self.fast_v_inplace:
                # inplace 模式下需要禁用缓存和梯度检查点
                logger.warning_once("FastV inplace mode: use_cache and gradient_checkpointing will be disabled.")
            else:
                logger.warning_once("FastV mask mode: use_cache may be disabled if attention mask changes.")

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def reset_fastv(self):
        """重置 FastV 配置（用于动态调整）"""
        self.use_fast_v = self.config.use_fast_v
        self.fast_v_sys_length = self.config.fast_v_sys_length
        self.fast_v_image_token_length = self.config.fast_v_image_token_length
        self.fast_v_attention_rank = self.config.fast_v_attention_rank
        self.fast_v_agg_layer = self.config.fast_v_agg_layer
        self.fast_v_inplace = self.config.fast_v_inplace

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # FastV 预处理：禁用缓存和梯度检查点（避免复杂性）
        if self.use_fast_v and self.fast_v_agg_layer is not None:
            if use_cache:
                logger.warning_once("FastV is enabled, setting use_cache=False to avoid KV cache issues.")
                use_cache = False
            if self.gradient_checkpointing and self.training:
                logger.warning_once("FastV is enabled, disabling gradient checkpointing.")
                self.gradient_checkpointing = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 用于 FastV 的变量
        current_attention_mask = causal_mask
        current_hidden_states = hidden_states
        current_position_ids = position_ids
        current_cache_position = cache_position
        current_past_key_values = past_key_values
        use_fast_v_this_run = self.use_fast_v and self.fast_v_agg_layer is not None
        agg_layer = self.fast_v_agg_layer if use_fast_v_this_run else None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (current_hidden_states,)

            # 决定是否在这一层输出注意力（仅当 idx == agg_layer 且启用 FastV 时需要）
            output_attentions_this = output_attentions
            if use_fast_v_this_run and idx == agg_layer:
                output_attentions_this = True

            # 选择当前层使用的输入变量
            if use_fast_v_this_run and idx >= agg_layer:
                # 后续层使用更新后的变量
                cur_attn_mask = current_attention_mask
                cur_pos_ids = current_position_ids
                cur_cache_pos = current_cache_position
                cur_past_kv = current_past_key_values
            else:
                # 前 agg_layer 层使用原始变量
                cur_attn_mask = causal_mask
                cur_pos_ids = position_ids
                cur_cache_pos = cache_position
                cur_past_kv = past_key_values

            # 调用解码器层
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    current_hidden_states,
                    cur_attn_mask,
                    cur_pos_ids,
                    cur_past_kv,
                    output_attentions_this,
                    use_cache,
                    cur_cache_pos,
                    position_embeddings,
                    **flash_attn_kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    current_hidden_states,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_pos_ids,
                    past_key_value=cur_past_kv,
                    output_attentions=output_attentions_this,
                    use_cache=use_cache,
                    cache_position=cur_cache_pos,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            current_hidden_states = layer_outputs[0]

            # FastV 核心逻辑：在第 agg_layer 层利用注意力权重选择 token
            if use_fast_v_this_run and idx == agg_layer:
                attn_weights = layer_outputs[1]  # 获取注意力权重

                # 计算平均注意力（对头平均，再对 batch 平均）
                # attn_weights shape: [batch, num_heads, q_len, kv_len]
                batch_size, num_heads, q_len, kv_len = attn_weights.shape
                # 平均 head 维度 -> [batch, q_len, kv_len]
                attn_avg_heads = torch.mean(attn_weights, dim=1)
                # 平均 batch 维度 -> [q_len, kv_len]
                attn_avg = attn_avg_heads.mean(dim=0)

                # 取最后一个 query 位置的注意力
                last_token_attn = attn_avg[-1]  # [kv_len]

                sys_len = self.fast_v_sys_length
                img_len = self.fast_v_image_token_length
                rank = self.fast_v_attention_rank

                # 图像 token 的注意力
                image_attn = last_token_attn[sys_len: sys_len + img_len]
                topk_indices = image_attn.topk(rank).indices + sys_len

                if self.fast_v_inplace:
                    # 保留的 token 索引：系统提示 + 选中的图像 token + 文本 token
                    keep_indices = torch.cat([
                        torch.arange(sys_len, device=current_hidden_states.device),
                        topk_indices,
                        torch.arange(sys_len + img_len, kv_len, device=current_hidden_states.device)
                    ]).sort().values
                    new_seq_len = keep_indices.shape[0]

                    # 裁剪 hidden_states
                    current_hidden_states = current_hidden_states[:, keep_indices, :]

                    # 更新 position_ids 和 cache_position
                    # 注意：假设 batch_size = 1 或所有样本使用相同索引
                    # 这里扩展到 batch_size
                    current_position_ids = keep_indices.unsqueeze(0).expand(batch_size, -1)
                    current_cache_position = torch.arange(new_seq_len, device=current_hidden_states.device)

                    # 生成新的 attention_mask（2D 全 1）
                    new_attn_mask_2d = torch.ones(
                        (batch_size, new_seq_len), dtype=torch.bool, device=current_hidden_states.device
                    )
                    # 重新生成 4D causal mask
                    current_attention_mask = self._update_causal_mask(
                        new_attn_mask_2d,
                        current_hidden_states,
                        current_cache_position,
                        None,  # past_key_values 置空
                        output_attentions,
                    )

                    # 缓存已失效，禁用后续缓存
                    use_cache = False
                    current_past_key_values = None
                else:
                    # 非 inplace 模式：生成新的 attention_mask，屏蔽不重要的图像 token
                    new_attn_mask_2d = torch.ones(
                        (batch_size, kv_len), dtype=torch.bool, device=current_hidden_states.device
                    )
                    new_attn_mask_2d[:, sys_len: sys_len + img_len] = False
                    new_attn_mask_2d[:, topk_indices] = True

                    current_attention_mask = self._update_causal_mask(
                        new_attn_mask_2d,
                        current_hidden_states,
                        current_cache_position,
                        current_past_key_values,
                        output_attentions,
                    )
                    # position_ids 和 cache_position 保持不变
                    # current_position_ids 和 current_cache_position 仍使用原始值

                # 本层已经处理，跳过后续的注意力输出收集（已在外部处理）
                # 但注意：当前层输出的 hidden_states 已经是更新后的值
                # 我们可能需要在收集输出时保留原始的输出，这里不重复收集

            # 收集输出（仅当 use_cache 或 output_attentions 启用时）
            if use_cache:
                # 注意：当前层可能已经将 past_key_values 置空，需要根据情况获取
                kv_out = layer_outputs[2 if output_attentions_this else 1]
                next_decoder_cache += (kv_out,)
            if output_attentions:
                # 只在需要时收集注意力权重
                if output_attentions_this:
                    all_self_attns += (layer_outputs[1],)
                else:
                    # 如果不需要输出，但收集列表仍保持长度一致，可以填充 None
                    # 简单起见，如果不输出就不加入
                    pass

            # 注意：在 FastV 非 inplace 模式下，我们未更新 current_position_ids 和 current_cache_position，
            # 因此后续层仍使用原始值，但 attention_mask 已更改。
            # 对于 inplace 模式，所有后续变量都已更新。

        # 最终层归一化
        current_hidden_states = self.norm(current_hidden_states)

        if output_hidden_states:
            all_hidden_states += (current_hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=current_hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    # ) -> Union[Tuple, BaseModelOutputWithPast]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     if (input_ids is None) ^ (inputs_embeds is not None):
    #         raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    #
    #     if self.gradient_checkpointing and self.training and use_cache:
    #         logger.warning_once(
    #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
    #         )
    #         use_cache = False
    #
    #     if inputs_embeds is None:
    #         inputs_embeds = self.embed_tokens(input_ids)
    #
    #     if use_cache and past_key_values is None:
    #         past_key_values = DynamicCache()
    #
    #     if cache_position is None:
    #         past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    #         cache_position = torch.arange(
    #             past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    #         )
    #
    #     if position_ids is None:
    #         position_ids = cache_position.unsqueeze(0)
    #
    #     causal_mask = self._update_causal_mask(
    #         attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    #     )
    #
    #     hidden_states = inputs_embeds
    #
    #     # create position embeddings to be shared across the decoder layers
    #     position_embeddings = self.rotary_emb(hidden_states, position_ids)
    #
    #     # decoder layers
    #     all_hidden_states = () if output_hidden_states else None
    #     all_self_attns = () if output_attentions else None
    #
    #     for decoder_layer in self.layers[: self.config.num_hidden_layers]:
    #         if output_hidden_states:
    #             all_hidden_states += (hidden_states,)
    #
    #         if self.gradient_checkpointing and self.training:
    #             layer_outputs = self._gradient_checkpointing_func(
    #                 decoder_layer.__call__,
    #                 hidden_states,
    #                 causal_mask,
    #                 position_ids,
    #                 past_key_values,
    #                 output_attentions,
    #                 use_cache,
    #                 cache_position,
    #                 position_embeddings,
    #             )
    #         else:
    #             layer_outputs = decoder_layer(
    #                 hidden_states,
    #                 attention_mask=causal_mask,
    #                 position_ids=position_ids,
    #                 past_key_value=past_key_values,
    #                 output_attentions=output_attentions,
    #                 use_cache=use_cache,
    #                 cache_position=cache_position,
    #                 position_embeddings=position_embeddings,
    #                 **flash_attn_kwargs,
    #             )
    #
    #         hidden_states = layer_outputs[0]
    #
    #         if output_attentions:
    #             all_self_attns += (layer_outputs[1],)
    #
    #     hidden_states = self.norm(hidden_states)
    #
    #     # add hidden states from the last decoder layer
    #     if output_hidden_states:
    #         all_hidden_states += (hidden_states,)
    #
    #     output = BaseModelOutputWithPast(
    #         last_hidden_state=hidden_states,
    #         past_key_values=past_key_values if use_cache else None,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attns,
    #     )
    #     return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
