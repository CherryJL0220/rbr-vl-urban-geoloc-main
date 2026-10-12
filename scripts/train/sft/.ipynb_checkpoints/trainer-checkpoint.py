# import json
# import os
# from types import MethodType
# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
# from typing_extensions import override
# from torch.nn import functional as F


# import numpy as np
# import torch
# from transformers import Seq2SeqTrainer

# from ...extras.constants import IGNORE_INDEX
# from ...extras.logging import get_logger
# from ..callbacks import PissaConvertCallback, SaveProcessorCallback
# from ..trainer_utils import create_custom_optimzer, create_custom_scheduler


# if TYPE_CHECKING:
#     from torch.utils.data import Dataset
#     from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
#     from transformers.trainer import PredictionOutput
#     from ...hparams import FinetuningArguments


# logger = get_logger(__name__)


# class CustomSeq2SeqTrainer(Seq2SeqTrainer):
#     r"""
#     Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
#     """

#     def __init__(
#             self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
#     ) -> None:
#         gen_kwargs = kwargs.pop("gen_kwargs", None)
#         super().__init__(**kwargs)
#         self.finetuning_args = finetuning_args

#         self._init_contrastive_projections()

#         # 添加调试日志
#         logger.info(f"Training arguments: {self.args}")
#         logger.info(f"SWFT loss weight: {getattr(self.args, 'swft_loss_weight', 'not set')}")
#         logger.info(f"Image-address loss weight: {getattr(self.args, 'img_addr_loss_weight', 'not set')}")
#         logger.info(f"Image-description loss weight: {getattr(self.args, 'img_desc_loss_weight', 'not set')}")
#         logger.info(f"Contrastive temperature: {getattr(self.args, 'contrastive_temperature', 'not set')}")

#         if processor is not None:
#             self.add_callback(SaveProcessorCallback(processor))

#         if finetuning_args.pissa_convert:
#             self.add_callback(PissaConvertCallback)

#         if finetuning_args.use_badam:
#             from badam import BAdamCallback, clip_grad_norm_old_version

#             self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
#             self.add_callback(BAdamCallback)

#         # 记录模型类型
#         logger.info(f"Model type: {type(self.model)}")
#         if hasattr(self.model, 'config'):
#             logger.info(f"Model config type: {getattr(self.model.config, 'model_type', 'unknown')}")

#         # 检查模型是否支持特征提取
#         logger.info(f"Model has vision_tower: {hasattr(self.model, 'vision_tower')}")
#         logger.info(f"Model has language_model: {hasattr(self.model, 'language_model')}")
#         logger.info(f"Model has get_vision_features: {hasattr(self.model, 'get_vision_features')}")
#         logger.info(f"Model has get_text_features: {hasattr(self.model, 'get_text_features')}")

#     def _init_contrastive_projections(self):
#         """初始化对比损失投影层"""
#         # 这些投影层将在第一次计算对比损失时根据实际特征维度创建
#         self.addr_projection = None
#         self.desc_projection = None
#         print(f"[INIT] Initialized contrastive projection layers")

#     def _ensure_projection_layers(self, vision_dim, text_dim, device):
#         """确保投影层存在，如果不存在则创建"""
#         # 获取视觉特征的数据类型
#         vision_dtype = torch.float16  # 默认使用 float16
#         if hasattr(self.model, 'dtype'):
#             vision_dtype = self.model.dtype
#         elif hasattr(self.model, 'config'):
#             if hasattr(self.model.config, 'torch_dtype'):
#                 vision_dtype = self.model.config.torch_dtype
        
#         print(f"[PROJECTION] Vision dtype: {vision_dtype}")
        
#         # 创建地址特征投影层
#         if self.addr_projection is None:
#             print(f"[PROJECTION] Creating address projection: {text_dim} -> {vision_dim}")
#             self.addr_projection = torch.nn.Linear(text_dim, vision_dim).to(device)
#             print(f"[PROJECTION] Address projection created")

#         # 创建描述特征投影层
#         if self.desc_projection is None:
#             print(f"[PROJECTION] Creating description projection: {text_dim} -> {vision_dim}")
#             self.desc_projection = torch.nn.Linear(text_dim, vision_dim).to(device)
#             print(f"[PROJECTION] Description projection created")

#     def _project_text_features(self, text_features, projection_type="address"):
#         """投影文本特征到视觉特征维度"""
#         if projection_type == "address" and self.addr_projection is not None:
#             projected = self.addr_projection(text_features)
#             # 确保投影后的特征与原始文本特征在同一设备上
#             if projected.device != text_features.device:
#                 projected = projected.to(text_features.device)
#             return projected
#         elif projection_type == "description" and self.desc_projection is not None:
#             projected = self.desc_projection(text_features)
#             # 确保投影后的特征与原始文本特征在同一设备上
#             if projected.device != text_features.device:
#                 projected = projected.to(text_features.device)
#             return projected
#         return text_features  # 如果没有投影层，返回原特征

#     def _get_vision_features(self, pixel_values: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
#         """
#         提取视觉特征

#         Args:
#             pixel_values: 图像像素值张量，形状为 [batch_size, channels, height, width]

#         Returns:
#             visual_features: 视觉特征，形状为 [batch_size, feature_dim]
#             如果无法提取，返回None
#         """
#         if pixel_values is None:
#             return None

#         try:
#             # 确保 torch 已导入
#             import torch

#             # 方法1: 如果模型本身有视觉特征提取方法
#             if hasattr(self.model, 'get_vision_features'):
#                 vision_features = self.model.get_vision_features(pixel_values)
#                 if vision_features is not None:
#                     return vision_features

#             # 方法2: 对于LLaVA类模型，使用视觉编码器
#             if hasattr(self.model, 'vision_tower'):
#                 # 使用 no_grad 避免梯度计算影响
#                 with torch.no_grad():
#                     vision_outputs = self.model.vision_tower(pixel_values)

#                 # 提取特征 - 尝试不同的方法
#                 if hasattr(vision_outputs, 'last_hidden_state'):
#                     # LLaVA 1.5/1.6 风格
#                     visual_features = vision_outputs.last_hidden_state
#                     # 取全局平均池化
#                     visual_features = visual_features.mean(dim=1)
#                     logger.debug(
#                         f"Extracted vision features from vision_tower.last_hidden_state, shape: {visual_features.shape}")
#                     return visual_features
#                 elif hasattr(vision_outputs, 'pooler_output'):
#                     # 某些模型使用pooler输出
#                     logger.debug(
#                         f"Extracted vision features from vision_tower.pooler_output, shape: {vision_outputs.pooler_output.shape}")
#                     return vision_outputs.pooler_output
#                 elif isinstance(vision_outputs, torch.Tensor):
#                     # 如果输出直接是张量
#                     if vision_outputs.dim() == 4:
#                         # [batch, channels, height, width] -> 全局平均池化
#                         visual_features = vision_outputs.mean(dim=[2, 3])
#                     elif vision_outputs.dim() == 3:
#                         # [batch, seq_len, hidden_dim] -> 取平均
#                         visual_features = vision_outputs.mean(dim=1)
#                     else:
#                         visual_features = vision_outputs
#                     logger.debug(f"Extracted vision features from vision_tower tensor, shape: {visual_features.shape}")
#                     return visual_features
#                 else:
#                     # 尝试获取第一个元素（有些模型返回元组）
#                     if isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
#                         vision_outputs = vision_outputs[0]
#                         if isinstance(vision_outputs, torch.Tensor):
#                             if vision_outputs.dim() == 4:
#                                 visual_features = vision_outputs.mean(dim=[2, 3])
#                             elif vision_outputs.dim() == 3:
#                                 visual_features = vision_outputs.mean(dim=1)
#                             else:
#                                 visual_features = vision_outputs
#                             logger.debug(
#                                 f"Extracted vision features from vision_tower tuple, shape: {visual_features.shape}")
#                             return visual_features

#             # 方法3: 尝试通过forward获取特征
#             if hasattr(self.model, 'get_image_features'):
#                 with torch.no_grad():
#                     visual_features = self.model.get_image_features(pixel_values)
#                 if visual_features is not None:
#                     logger.debug(f"Extracted vision features from get_image_features, shape: {visual_features.shape}")
#                     return visual_features

#             # 方法4: 对于Qwen-VL类模型
#             model_type = getattr(self.model.config, "model_type", "unknown") if hasattr(self.model,
#                                                                                         'config') else "unknown"
#             if model_type in ['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl', 'qwen-vl']:
#                 # Qwen-VL可能有不同的视觉编码器结构
#                 if hasattr(self.model, 'visual'):
#                     with torch.no_grad():
#                         visual_outputs = self.model.visual(pixel_values)
#                     if hasattr(visual_outputs, 'last_hidden_state'):
#                         visual_features = visual_outputs.last_hidden_state.mean(dim=1)
#                         logger.debug(
#                             f"Extracted vision features from Qwen-VL visual.last_hidden_state, shape: {visual_features.shape}")
#                         return visual_features
#                     elif isinstance(visual_outputs, tuple) and len(visual_outputs) > 0:
#                         visual_features = visual_outputs[0].mean(dim=1)
#                         logger.debug(
#                             f"Extracted vision features from Qwen-VL visual tuple, shape: {visual_features.shape}")
#                         return visual_features

#             # 方法5: 尝试从模型配置中推断视觉特征提取方法
#             if hasattr(self.model, 'config'):
#                 # 检查是否有视觉模型名称
#                 vision_model_name = getattr(self.model.config, 'vision_model_name', None)
#                 if vision_model_name:
#                     logger.debug(f"Model has vision_model_name: {vision_model_name}")

#             # 如果没有提取到特征，尝试更通用的方法
#             # 查找可能包含视觉特征的属性
#             vision_attrs = ['vision_model', 'vit', 'clip', 'visual_transformer']
#             for attr in vision_attrs:
#                 if hasattr(self.model, attr):
#                     try:
#                         vision_module = getattr(self.model, attr)
#                         with torch.no_grad():
#                             vision_outputs = vision_module(pixel_values)
#                         if hasattr(vision_outputs, 'last_hidden_state'):
#                             visual_features = vision_outputs.last_hidden_state.mean(dim=1)
#                             logger.debug(
#                                 f"Extracted vision features from {attr}.last_hidden_state, shape: {visual_features.shape}")
#                             return visual_features
#                         elif isinstance(vision_outputs, torch.Tensor):
#                             if vision_outputs.dim() == 4:
#                                 visual_features = vision_outputs.mean(dim=[2, 3])
#                             elif vision_outputs.dim() == 3:
#                                 visual_features = vision_outputs.mean(dim=1)
#                             else:
#                                 visual_features = vision_outputs
#                             logger.debug(
#                                 f"Extracted vision features from {attr} tensor, shape: {visual_features.shape}")
#                             return visual_features
#                     except Exception as e:
#                         logger.debug(f"Failed to extract vision features from {attr}: {e}")
#                         continue

#         except Exception as e:
#             logger.warning(f"Failed to extract vision features: {e}")
#             import traceback
#             logger.debug(f"Traceback: {traceback.format_exc()}")
#             return None

#         logger.warning("Could not extract vision features using any method")
#         return None

#     def _get_text_features(self, input_ids: Optional[torch.Tensor] = None,
#                            attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
#         """
#         提取文本特征

#         Args:
#             input_ids: 输入token IDs
#             attention_mask: 注意力掩码

#         Returns:
#             text_features: 文本特征，形状为 [batch_size, feature_dim]
#             如果无法提取，返回None
#         """
#         if input_ids is None:
#             return None

#         try:
#             # 确保 torch 已导入
#             import torch

#             # 方法1: 如果模型本身有文本特征提取方法
#             if hasattr(self.model, 'get_text_features'):
#                 text_features = self.model.get_text_features(input_ids, attention_mask)
#                 if text_features is not None:
#                     return text_features

#             # 方法2: 对于LLaVA类模型，使用语言模型
#             if hasattr(self.model, 'language_model'):
#                 # 获取语言模型的输出
#                 with torch.no_grad():
#                     text_outputs = self.model.language_model(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         output_hidden_states=True
#                     )

#                 # 获取最后一层隐藏状态
#                 if hasattr(text_outputs, 'hidden_states') and len(text_outputs.hidden_states) > 0:
#                     last_hidden_state = text_outputs.hidden_states[-1]

#                     # 根据attention_mask计算加权平均
#                     if attention_mask is not None:
#                         attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
#                         sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
#                         sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
#                         text_features = sum_embeddings / sum_mask
#                     else:
#                         # 如果没有attention_mask，取平均
#                         text_features = last_hidden_state.mean(dim=1)

#                     logger.debug(f"Extracted text features from language_model, shape: {text_features.shape}")
#                     return text_features

#                 # 如果没有hidden_states，尝试其他属性
#                 if hasattr(text_outputs, 'pooler_output'):
#                     logger.debug(
#                         f"Extracted text features from language_model.pooler_output, shape: {text_outputs.pooler_output.shape}")
#                     return text_outputs.pooler_output

#                 if hasattr(text_outputs, 'last_hidden_state'):
#                     last_hidden_state = text_outputs.last_hidden_state
#                     if attention_mask is not None:
#                         attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
#                         sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
#                         sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
#                         text_features = sum_embeddings / sum_mask
#                     else:
#                         text_features = last_hidden_state.mean(dim=1)
#                     logger.debug(
#                         f"Extracted text features from language_model.last_hidden_state, shape: {text_features.shape}")
#                     return text_features

#             # 方法3: 使用语言模型的pooler输出（如果有）
#             if hasattr(self.model, 'text_pooler'):
#                 with torch.no_grad():
#                     text_outputs = self.model.language_model(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask
#                     )
#                 if hasattr(text_outputs, 'pooler_output'):
#                     logger.debug(f"Extracted text features from text_pooler, shape: {text_outputs.pooler_output.shape}")
#                     return text_outputs.pooler_output

#             # 方法4: 对于没有明确语言模型的模型，尝试直接调用模型
#             # 但注意：这可能会影响梯度，所以使用no_grad
#             with torch.no_grad():
#                 text_outputs = self.model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     output_hidden_states=True
#                 )

#             if hasattr(text_outputs, 'hidden_states') and len(text_outputs.hidden_states) > 0:
#                 last_hidden_state = text_outputs.hidden_states[-1]
#                 if attention_mask is not None:
#                     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
#                     sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
#                     sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
#                     text_features = sum_embeddings / sum_mask
#                 else:
#                     text_features = last_hidden_state.mean(dim=1)
#                 logger.debug(f"Extracted text features from model.hidden_states, shape: {text_features.shape}")
#                 return text_features

#             if hasattr(text_outputs, 'last_hidden_state'):
#                 last_hidden_state = text_outputs.last_hidden_state
#                 if attention_mask is not None:
#                     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
#                     sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
#                     sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
#                     text_features = sum_embeddings / sum_mask
#                 else:
#                     text_features = last_hidden_state.mean(dim=1)
#                 logger.debug(f"Extracted text features from model.last_hidden_state, shape: {text_features.shape}")
#                 return text_features

#             if hasattr(text_outputs, 'pooler_output'):
#                 logger.debug(
#                     f"Extracted text features from model.pooler_output, shape: {text_outputs.pooler_output.shape}")
#                 return text_outputs.pooler_output

#             # 方法5: 对于多模态模型，可能有单独的文本编码器
#             text_attrs = ['text_model', 'text_encoder', 'bert', 'roberta']
#             for attr in text_attrs:
#                 if hasattr(self.model, attr):
#                     try:
#                         text_module = getattr(self.model, attr)
#                         with torch.no_grad():
#                             text_outputs = text_module(
#                                 input_ids=input_ids,
#                                 attention_mask=attention_mask,
#                                 output_hidden_states=True
#                             )

#                         if hasattr(text_outputs, 'pooler_output'):
#                             logger.debug(
#                                 f"Extracted text features from {attr}.pooler_output, shape: {text_outputs.pooler_output.shape}")
#                             return text_outputs.pooler_output

#                         if hasattr(text_outputs, 'last_hidden_state'):
#                             last_hidden_state = text_outputs.last_hidden_state
#                             if attention_mask is not None:
#                                 attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
#                                 sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
#                                 sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
#                                 text_features = sum_embeddings / sum_mask
#                             else:
#                                 text_features = last_hidden_state.mean(dim=1)
#                             logger.debug(
#                                 f"Extracted text features from {attr}.last_hidden_state, shape: {text_features.shape}")
#                             return text_features
#                     except Exception as e:
#                         logger.debug(f"Failed to extract text features from {attr}: {e}")
#                         continue

#         except Exception as e:
#             logger.warning(f"Failed to extract text features: {e}")
#             import traceback
#             logger.debug(f"Traceback: {traceback.format_exc()}")
#             return None

#         logger.warning("Could not extract text features using any method")
#         return None

#     def create_optimizer(self) -> "torch.optim.Optimizer":
#         if self.optimizer is None:
#             self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
#         return super().create_optimizer()

#     def create_scheduler(
#         self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
#     ) -> "torch.optim.lr_scheduler.LRScheduler":
#         create_custom_scheduler(self.args, num_training_steps, optimizer)
#         return super().create_scheduler(num_training_steps, optimizer)

#     @override
#     def compute_loss(
#             self,
#             model: "PreTrainedModel",
#             inputs: Dict[str, "torch.Tensor"],
#             return_outputs: bool = False,
#             **kwargs
#     ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
#         r"""
#         计算总损失，包括SWFT损失和对比损失。
#         """
#         ### 1. 提取ratio值（如果有）
#         ratio = inputs.pop("ratio", torch.tensor(1.0)).item()  # 获取SWFT权重比
#         print("ratio", ratio)

#         ### 调试：打印输入键名
#         print(f"[DEBUG] Input keys: {list(inputs.keys())}")

#         ### 2. 计算基础SWFT损失
#         loss_output = super().compute_loss(model, inputs, return_outputs, **kwargs)
#         print("loss_output", loss_output)

#         if return_outputs:
#             loss_swft, outputs = loss_output
#             # 应用梯度累积调整
#             if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
#                 loss_swft = loss_swft / self.args.gradient_accumulation_steps
#             # 应用ratio加权
#             loss_swft = loss_swft * ratio
#         else:
#             loss_swft = loss_output
#             # 应用梯度累积调整
#             if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
#                 loss_swft = loss_swft / self.args.gradient_accumulation_steps
#             # 应用ratio加权
#             loss_swft = loss_swft * ratio

#         ### 3. 计算对比损失（图像-地址、图像-描述）
#         loss_contrastive = torch.tensor(0.0, device=loss_swft.device)
#         if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
#             loss_contrastive = loss_contrastive / self.args.gradient_accumulation_steps
#         print("loss_contrastive", loss_contrastive)

#         # 检查是否处于训练阶段并启用了对比损失
#         print(f"[DEBUG] Model training mode: {model.training}")
#         print(f"[DEBUG] Should compute contrastive: {self._should_compute_contrastive_loss()}")

#         if model.training and self._should_compute_contrastive_loss():
#             print("[DEBUG] Computing contrastive losses...")
#             loss_contrastive = self._compute_contrastive_losses(inputs)
#             print(f"[DEBUG] Contrastive loss computed: {loss_contrastive.item()}")

#         ### 4. 融合总损失
#         lambda1 = getattr(self.args, "swft_loss_weight", 1.0)
#         print("lambda1", lambda1)
#         total_loss = lambda1 * loss_swft + loss_contrastive
#         print("total_loss", total_loss)

#         logger.debug(f"SWFT loss: {loss_swft.item() if isinstance(loss_swft, torch.Tensor) else loss_swft}")
#         logger.debug(f"Contrastive loss: {loss_contrastive.item()}")
#         logger.debug(f"Total loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")

#         if return_outputs:
#             return (total_loss, outputs)
#         else:
#             return total_loss

#     def _should_compute_contrastive_loss(self) -> bool:
#         """检查是否应该计算对比损失"""
#         # 检查是否有对比损失权重设置
#         lambda2 = getattr(self.args, "img_addr_loss_weight", 0.0)
#         lambda3 = getattr(self.args, "img_desc_loss_weight", 0.0)
        
#         # 只要有一个对比损失权重大于0，就应该计算
#         return lambda2 > 0 or lambda3 > 0

#     def _compute_contrastive_losses(self, inputs: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
#         """计算所有对比损失的总和"""
#         total_contrastive_loss = torch.tensor(0.0, device=inputs["input_ids"].device)
        
#         # 尝试获取视觉特征
#         vision_features = self._extract_vision_features(inputs)
#         if vision_features is None:
#             logger.warning("Could not extract vision features. Skipping contrastive loss.")
#             return total_contrastive_loss
        
#         # 计算图像-地址对比损失
#         lambda2 = getattr(self.args, "img_addr_loss_weight", 0.0)
#         if lambda2 > 0:
#             addr_loss = self._compute_image_address_loss(vision_features, inputs)
#             print("addr_loss", addr_loss)
#             if addr_loss is not None:
#                 total_contrastive_loss = total_contrastive_loss + lambda2 * addr_loss
        
#         # 计算图像-描述对比损失
#         lambda3 = getattr(self.args, "img_desc_loss_weight", 0.0)
#         if lambda3 > 0:
#             desc_loss = self._compute_image_description_loss(vision_features, inputs)
#             print("desc_loss", desc_loss)

#             if desc_loss is not None:
#                 total_contrastive_loss = total_contrastive_loss + lambda3 * desc_loss
        
#         return total_contrastive_loss

#     def _extract_vision_features(self, inputs: Dict[str, "torch.Tensor"]) -> Optional["torch.Tensor"]:
#         """从输入中提取视觉特征"""
#         try:
#             print(f"[DEBUG] Looking for vision features in inputs...")

#             # 尝试从不同的键中获取像素值
#             vision_keys = ["pixel_values", "vision_feats", "image_features"]

#             found_key = None
#             for key in vision_keys:
#                 if key in inputs:
#                     print(f"[DEBUG] Found key '{key}' in inputs")
#                     if inputs[key] is not None:
#                         print(f"[DEBUG] Key '{key}' has shape: {inputs[key].shape}, dtype: {inputs[key].dtype}")
#                         found_key = key
#                         break

#             if found_key is None:
#                 print("[WARNING] No vision features found in inputs")
#                 print(f"[DEBUG] All input keys: {list(inputs.keys())}")
#                 return None

#             print(f"[DEBUG] Extracting vision features from key: {found_key}")
#             print(f"[DEBUG] Model type: {type(self.model)}")

#             # 直接使用内部方法提取视觉特征
#             vision_features = self._get_vision_features(inputs[found_key])
#             if vision_features is not None:
#                 print(f"[DEBUG] Vision features extracted successfully, shape: {vision_features.shape}, dtype: {vision_features.dtype}")
#                 return vision_features
#             else:
#                 print("[WARNING] _get_vision_features returned None")

#             return None

#         except Exception as e:
#             print(f"[ERROR] Failed to extract vision features: {e}")
#             import traceback
#             print(f"[ERROR] Traceback: {traceback.format_exc()}")
#             return None

#     def _compute_image_address_loss(self, vision_features: "torch.Tensor", inputs: Dict[str, "torch.Tensor"]) -> \
#             Optional["torch.Tensor"]:
#         """计算图像-地址对比损失"""
#         try:
#             print(f"[DEBUG] Looking for address features in inputs...")

#             # 尝试获取地址特征
#             addr_input_keys = ["addr_input_ids", "address_input_ids", "addr_text_ids"]
#             addr_mask_keys = ["addr_attention_mask", "address_attention_mask", "addr_text_mask"]

#             addr_features = None
#             found_input_key = None
#             found_mask_key = None

#             for input_key, mask_key in zip(addr_input_keys, addr_mask_keys):
#                 if input_key in inputs and mask_key in inputs:
#                     print(f"[DEBUG] Found address keys: {input_key}, {mask_key}")
#                     if inputs[input_key] is not None and inputs[mask_key] is not None:
#                         print(f"[DEBUG] {input_key} shape: {inputs[input_key].shape}, dtype: {inputs[input_key].dtype}")
#                         print(f"[DEBUG] {mask_key} shape: {inputs[mask_key].shape}, dtype: {inputs[mask_key].dtype}")
#                         found_input_key = input_key
#                         found_mask_key = mask_key
#                         break

#             if found_input_key is None:
#                 print("[WARNING] No address features found in inputs")
#                 return None

#             print(f"[DEBUG] Extracting address features from keys: {found_input_key}, {found_mask_key}")

#             # 直接使用内部方法提取文本特征
#             addr_features = self._get_text_features(
#                 inputs[found_input_key],
#                 inputs[found_mask_key]
#             )
#             if addr_features is not None:
#                 print(f"[DEBUG] Address features extracted successfully, shape: {addr_features.shape}, dtype: {addr_features.dtype}")
#             else:
#                 print("[WARNING] _get_text_features returned None")
#                 return None

#             # 确保投影层存在
#             vision_dim = vision_features.shape[1]
#             text_dim = addr_features.shape[1]
#             self._ensure_projection_layers(vision_dim, text_dim, vision_features.device)

#             # 投影地址特征到视觉特征维度
#             addr_features_projected = self._project_text_features(addr_features, "address")
#             print(f"[DEBUG] Address features projected from {text_dim} to {addr_features_projected.shape[1]}, dtype: {addr_features_projected.dtype}")
#             print(f"[DEBUG] Vision features dtype: {vision_features.dtype}")

#             # 确保数据类型匹配
#             if addr_features_projected.dtype != vision_features.dtype:
#                 print(f"[DEBUG] Converting projected features dtype from {addr_features_projected.dtype} to {vision_features.dtype}")
#                 addr_features_projected = addr_features_projected.to(vision_features.dtype)

#             # 确保批次大小匹配
#             print(f"[DEBUG] Vision features batch size: {vision_features.shape[0]}")
#             print(f"[DEBUG] Address features batch size: {addr_features_projected.shape[0]}")

#             if vision_features.shape[0] != addr_features_projected.shape[0]:
#                 print(f"[WARNING] Batch size mismatch: vision {vision_features.shape[0]}, address {addr_features_projected.shape[0]}")
#                 return None

#             # 计算对比损失
#             temperature = getattr(self.args, "contrastive_temperature", 0.07)
#             print(f"[DEBUG] Contrastive temperature: {temperature}")
#             loss = self._compute_contrastive_loss_pair(vision_features, addr_features_projected, temperature)

#             print(f"[DEBUG] Image-address contrastive loss: {loss.item()}")
#             return loss

#         except Exception as e:
#             print(f"[ERROR] Failed to compute image-address loss: {e}")
#             import traceback
#             print(f"[ERROR] Traceback: {traceback.format_exc()}")
#             return None

#     def _compute_image_description_loss(self, vision_features: "torch.Tensor", inputs: Dict[str, "torch.Tensor"]) -> \
#     Optional["torch.Tensor"]:
#         """计算图像-描述对比损失"""
#         try:
#             print(f"[DEBUG] Looking for description features in inputs...")

#             # 尝试获取描述特征
#             desc_input_keys = ["desc_input_ids", "description_input_ids", "desc_text_ids"]
#             desc_mask_keys = ["desc_attention_mask", "description_attention_mask", "desc_text_mask"]

#             desc_features = None
#             found_input_key = None
#             found_mask_key = None

#             for input_key, mask_key in zip(desc_input_keys, desc_mask_keys):
#                 if input_key in inputs and mask_key in inputs:
#                     print(f"[DEBUG] Found description keys: {input_key}, {mask_key}")
#                     if inputs[input_key] is not None and inputs[mask_key] is not None:
#                         print(f"[DEBUG] {input_key} shape: {inputs[input_key].shape}")
#                         print(f"[DEBUG] {mask_key} shape: {inputs[mask_key].shape}")
#                         found_input_key = input_key
#                         found_mask_key = mask_key
#                         break

#             if found_input_key is None:
#                 print("[WARNING] No description features found in inputs")
#                 return None

#             print(f"[DEBUG] Extracting description features from keys: {found_input_key}, {found_mask_key}")

#             # 直接使用内部方法提取文本特征
#             desc_features = self._get_text_features(
#                 inputs[found_input_key],
#                 inputs[found_mask_key]
#             )
#             if desc_features is not None:
#                 print(f"[DEBUG] Description features extracted successfully, shape: {desc_features.shape}")
#             else:
#                 print("[WARNING] _get_text_features returned None")
#                 return None

#             # 确保投影层存在
#             vision_dim = vision_features.shape[1]
#             text_dim = desc_features.shape[1]
#             self._ensure_projection_layers(vision_dim, text_dim, vision_features.device)

#             # 投影描述特征到视觉特征维度
#             desc_features_projected = self._project_text_features(desc_features, "description")
#             print(f"[DEBUG] Description features projected from {text_dim} to {desc_features_projected.shape[1]}")

#             # 确保批次大小匹配
#             print(f"[DEBUG] Vision features batch size: {vision_features.shape[0]}")
#             print(f"[DEBUG] Description features batch size: {desc_features_projected.shape[0]}")

#             if vision_features.shape[0] != desc_features_projected.shape[0]:
#                 print(f"[WARNING] Batch size mismatch: vision {vision_features.shape[0]}, description {desc_features_projected.shape[0]}")
#                 return None

#             # 计算对比损失
#             temperature = getattr(self.args, "contrastive_temperature", 0.07)
#             print(f"[DEBUG] Contrastive temperature: {temperature}")
#             loss = self._compute_contrastive_loss_pair(vision_features, desc_features_projected, temperature)

#             print(f"[DEBUG] Image-description contrastive loss: {loss.item()}")
#             return loss

#         except Exception as e:
#             print(f"[ERROR] Failed to compute image-description loss: {e}")
#             import traceback
#             print(f"[ERROR] Traceback: {traceback.format_exc()}")
#             return None

#     def _compute_contrastive_loss_pair(self, features1: "torch.Tensor", features2: "torch.Tensor",
#                                        tau: float) -> "torch.Tensor":
#         """计算一对特征之间的对比损失"""
#         # 检查维度是否匹配
#         if features1.shape[1] != features2.shape[1]:
#             print(f"[WARNING] Feature dimension mismatch: {features1.shape} vs {features2.shape}")
#             # 这里可以添加投影逻辑，但现在我们先跳过
#             return torch.tensor(0.0, device=features1.device)

#         # 确保特征数据类型一致
#         if features1.dtype != features2.dtype:
#             print(f"[DEBUG] Converting features2 dtype from {features2.dtype} to {features1.dtype}")
#             features2 = features2.to(features1.dtype)

#         # 归一化特征
#         features1_norm = F.normalize(features1, dim=-1)
#         features2_norm = F.normalize(features2, dim=-1)

#         # 计算相似度矩阵
#         logits = torch.matmul(features1_norm, features2_norm.T) / tau
#         batch_size = features1_norm.shape[0]

#         # 创建标签
#         labels = torch.arange(batch_size, device=features1_norm.device)

#         # 计算双向交叉熵损失
#         loss_i2t = F.cross_entropy(logits, labels)
#         loss_t2i = F.cross_entropy(logits.T, labels)

#         # 返回平均损失
#         return (loss_i2t + loss_t2i) / 2
    
#     @override
#     def _set_signature_columns_if_needed(self):
#         super()._set_signature_columns_if_needed()
#         # 添加对比损失相关的列到签名列中
#         contrastive_columns = ["ratio", "addr_input_ids", "addr_attention_mask", 
#                               "desc_input_ids", "desc_attention_mask", "pixel_values"]
#         self._signature_columns = list(set(self._signature_columns + contrastive_columns))
#         logger.debug(f"Signature columns: {self._signature_columns}")

#     def prediction_step(
#         self,
#         model: "torch.nn.Module",
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         r"""
#         Removes the prompt part in the generated tokens.

#         Subclass and override to inject custom behavior.
#         """
#         labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
#         if self.args.predict_with_generate:
#             assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
#             prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
#             if prompt_len > label_len:
#                 inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
#             if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
#                 inputs["labels"] = inputs["labels"][:, :prompt_len]

#         # 在预测步骤中移除对比损失相关的输入，以免干扰正常的预测
#         prediction_inputs = {k: v for k, v in inputs.items() 
#                            if not k.startswith(('addr_', 'desc_', 'pixel_values', 'vision_'))}
        
#         loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
#             model, prediction_inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
#         )
#         if generated_tokens is not None and self.args.predict_with_generate:
#             generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
#             generated_tokens = generated_tokens.contiguous()

#         return loss, generated_tokens, labels

#     def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
#         r"""
#         Pads the tensor to the same length as the target tensor.
#         """
#         assert self.tokenizer.pad_token_id is not None, "Pad token is required."
#         padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
#         padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
#         return padded_tensor.contiguous()  # in contiguous memory

#     def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
#         r"""
#         Saves model predictions to `output_dir`.

#         A custom behavior that not contained in Seq2SeqTrainer.
#         """
#         if not self.is_world_process_zero():
#             return

#         output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
#         logger.info(f"Saving prediction results to {output_prediction_file}")

#         labels = np.where(
#             predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
#         )
#         preds = np.where(
#             predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
#         )

#         for i in range(len(preds)):
#             pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
#             if len(pad_len):  # move pad token to last
#                 preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

#         decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
#         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

#         with open(output_prediction_file, "w", encoding="utf-8") as writer:
#             res: List[str] = []
#             for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
#                 res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            
#             writer.write("\n".join(res))

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing_extensions import override
from torch.nn import functional as F


import numpy as np
import torch
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
            self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        gen_kwargs = kwargs.pop("gen_kwargs", None)
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        self._init_contrastive_projections()

        # 添加调试日志
        logger.info(f"Training arguments: {self.args}")
        logger.info(f"SWFT loss weight: {getattr(self.args, 'swft_loss_weight', 'not set')}")
        logger.info(f"Image-address loss weight: {getattr(self.args, 'img_addr_loss_weight', 'not set')}")
        logger.info(f"Image-description loss weight: {getattr(self.args, 'img_desc_loss_weight', 'not set')}")
        logger.info(f"Contrastive temperature: {getattr(self.args, 'contrastive_temperature', 'not set')}")

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # 记录模型类型
        logger.info(f"Model type: {type(self.model)}")
        if hasattr(self.model, 'config'):
            logger.info(f"Model config type: {getattr(self.model.config, 'model_type', 'unknown')}")

        # 检查模型是否支持特征提取
        logger.info(f"Model has vision_tower: {hasattr(self.model, 'vision_tower')}")
        logger.info(f"Model has language_model: {hasattr(self.model, 'language_model')}")
        logger.info(f"Model has get_vision_features: {hasattr(self.model, 'get_vision_features')}")
        logger.info(f"Model has get_text_features: {hasattr(self.model, 'get_text_features')}")

    def _init_contrastive_projections(self):
        """初始化对比损失投影层"""
        # 这些投影层将在第一次计算对比损失时根据实际特征维度创建
        self.addr_projection = None
        self.desc_projection = None
        print(f"[INIT] Initialized contrastive projection layers")

    def _ensure_projection_layers(self, vision_dim, text_dim, device):
        """确保投影层存在，如果不存在则创建"""
        # 获取视觉特征的数据类型
        vision_dtype = torch.float16  # 默认使用 float16
        if hasattr(self.model, 'dtype'):
            vision_dtype = self.model.dtype
        elif hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'torch_dtype'):
                vision_dtype = self.model.config.torch_dtype
        
        print(f"[PROJECTION] Vision dtype: {vision_dtype}")
        
        # 创建地址特征投影层
        if self.addr_projection is None:
            print(f"[PROJECTION] Creating address projection: {text_dim} -> {vision_dim}")
            self.addr_projection = torch.nn.Linear(text_dim, vision_dim).to(device)
            print(f"[PROJECTION] Address projection created")

        # 创建描述特征投影层
        if self.desc_projection is None:
            print(f"[PROJECTION] Creating description projection: {text_dim} -> {vision_dim}")
            self.desc_projection = torch.nn.Linear(text_dim, vision_dim).to(device)
            print(f"[PROJECTION] Description projection created")

    def _project_text_features(self, text_features, projection_type="address"):
        """投影文本特征到视觉特征维度"""
        if projection_type == "address" and self.addr_projection is not None:
            projected = self.addr_projection(text_features)
            # 确保投影后的特征与原始文本特征在同一设备上
            if projected.device != text_features.device:
                projected = projected.to(text_features.device)
            return projected
        elif projection_type == "description" and self.desc_projection is not None:
            projected = self.desc_projection(text_features)
            # 确保投影后的特征与原始文本特征在同一设备上
            if projected.device != text_features.device:
                projected = projected.to(text_features.device)
            return projected
        return text_features  # 如果没有投影层，返回原特征

    def _get_vision_features(self, pixel_values: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        提取视觉特征

        Args:
            pixel_values: 图像像素值张量，形状为 [batch_size, channels, height, width]

        Returns:
            visual_features: 视觉特征，形状为 [batch_size, feature_dim]
            如果无法提取，返回None
        """
        if pixel_values is None:
            return None

        try:
            # 确保 torch 已导入
            import torch

            # 方法1: 如果模型本身有视觉特征提取方法
            if hasattr(self.model, 'get_vision_features'):
                vision_features = self.model.get_vision_features(pixel_values)
                if vision_features is not None:
                    return vision_features

            # 方法2: 对于LLaVA类模型，使用视觉编码器
            if hasattr(self.model, 'vision_tower'):
                # 使用 no_grad 避免梯度计算影响
                with torch.no_grad():
                    vision_outputs = self.model.vision_tower(pixel_values)

                # 提取特征 - 尝试不同的方法
                if hasattr(vision_outputs, 'last_hidden_state'):
                    # LLaVA 1.5/1.6 风格
                    visual_features = vision_outputs.last_hidden_state
                    # 取全局平均池化
                    visual_features = visual_features.mean(dim=1)
                    logger.debug(
                        f"Extracted vision features from vision_tower.last_hidden_state, shape: {visual_features.shape}")
                    return visual_features
                elif hasattr(vision_outputs, 'pooler_output'):
                    # 某些模型使用pooler输出
                    logger.debug(
                        f"Extracted vision features from vision_tower.pooler_output, shape: {vision_outputs.pooler_output.shape}")
                    return vision_outputs.pooler_output
                elif isinstance(vision_outputs, torch.Tensor):
                    # 如果输出直接是张量
                    if vision_outputs.dim() == 4:
                        # [batch, channels, height, width] -> 全局平均池化
                        visual_features = vision_outputs.mean(dim=[2, 3])
                    elif vision_outputs.dim() == 3:
                        # [batch, seq_len, hidden_dim] -> 取平均
                        visual_features = vision_outputs.mean(dim=1)
                    else:
                        visual_features = vision_outputs
                    logger.debug(f"Extracted vision features from vision_tower tensor, shape: {visual_features.shape}")
                    return visual_features
                else:
                    # 尝试获取第一个元素（有些模型返回元组）
                    if isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
                        vision_outputs = vision_outputs[0]
                        if isinstance(vision_outputs, torch.Tensor):
                            if vision_outputs.dim() == 4:
                                visual_features = vision_outputs.mean(dim=[2, 3])
                            elif vision_outputs.dim() == 3:
                                visual_features = vision_outputs.mean(dim=1)
                            else:
                                visual_features = vision_outputs
                            logger.debug(
                                f"Extracted vision features from vision_tower tuple, shape: {visual_features.shape}")
                            return visual_features

            # 方法3: 尝试通过forward获取特征
            if hasattr(self.model, 'get_image_features'):
                with torch.no_grad():
                    visual_features = self.model.get_image_features(pixel_values)
                if visual_features is not None:
                    logger.debug(f"Extracted vision features from get_image_features, shape: {visual_features.shape}")
                    return visual_features

            # 方法4: 对于Qwen-VL类模型
            model_type = getattr(self.model.config, "model_type", "unknown") if hasattr(self.model,
                                                                                        'config') else "unknown"
            if model_type in ['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl', 'qwen-vl']:
                # Qwen-VL可能有不同的视觉编码器结构
                if hasattr(self.model, 'visual'):
                    with torch.no_grad():
                        visual_outputs = self.model.visual(pixel_values)
                    if hasattr(visual_outputs, 'last_hidden_state'):
                        visual_features = visual_outputs.last_hidden_state.mean(dim=1)
                        logger.debug(
                            f"Extracted vision features from Qwen-VL visual.last_hidden_state, shape: {visual_features.shape}")
                        return visual_features
                    elif isinstance(visual_outputs, tuple) and len(visual_outputs) > 0:
                        visual_features = visual_outputs[0].mean(dim=1)
                        logger.debug(
                            f"Extracted vision features from Qwen-VL visual tuple, shape: {visual_features.shape}")
                        return visual_features

            # 方法5: 尝试从模型配置中推断视觉特征提取方法
            if hasattr(self.model, 'config'):
                # 检查是否有视觉模型名称
                vision_model_name = getattr(self.model.config, 'vision_model_name', None)
                if vision_model_name:
                    logger.debug(f"Model has vision_model_name: {vision_model_name}")

            # 如果没有提取到特征，尝试更通用的方法
            # 查找可能包含视觉特征的属性
            vision_attrs = ['vision_model', 'vit', 'clip', 'visual_transformer']
            for attr in vision_attrs:
                if hasattr(self.model, attr):
                    try:
                        vision_module = getattr(self.model, attr)
                        with torch.no_grad():
                            vision_outputs = vision_module(pixel_values)
                        if hasattr(vision_outputs, 'last_hidden_state'):
                            visual_features = vision_outputs.last_hidden_state.mean(dim=1)
                            logger.debug(
                                f"Extracted vision features from {attr}.last_hidden_state, shape: {visual_features.shape}")
                            return visual_features
                        elif isinstance(vision_outputs, torch.Tensor):
                            if vision_outputs.dim() == 4:
                                visual_features = vision_outputs.mean(dim=[2, 3])
                            elif vision_outputs.dim() == 3:
                                visual_features = vision_outputs.mean(dim=1)
                            else:
                                visual_features = vision_outputs
                            logger.debug(
                                f"Extracted vision features from {attr} tensor, shape: {visual_features.shape}")
                            return visual_features
                    except Exception as e:
                        logger.debug(f"Failed to extract vision features from {attr}: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Failed to extract vision features: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

        logger.warning("Could not extract vision features using any method")
        return None

    def _get_text_features(self, input_ids: Optional[torch.Tensor] = None,
                           attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        提取文本特征

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码

        Returns:
            text_features: 文本特征，形状为 [batch_size, feature_dim]
            如果无法提取，返回None
        """
        if input_ids is None:
            return None

        try:
            # 确保 torch 已导入
            import torch

            # 方法1: 如果模型本身有文本特征提取方法
            if hasattr(self.model, 'get_text_features'):
                text_features = self.model.get_text_features(input_ids, attention_mask)
                if text_features is not None:
                    return text_features

            # 方法2: 对于LLaVA类模型，使用语言模型
            if hasattr(self.model, 'language_model'):
                # 获取语言模型的输出
                with torch.no_grad():
                    text_outputs = self.model.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                # 获取最后一层隐藏状态
                if hasattr(text_outputs, 'hidden_states') and len(text_outputs.hidden_states) > 0:
                    last_hidden_state = text_outputs.hidden_states[-1]

                    # 根据attention_mask计算加权平均
                    if attention_mask is not None:
                        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                        text_features = sum_embeddings / sum_mask
                    else:
                        # 如果没有attention_mask，取平均
                        text_features = last_hidden_state.mean(dim=1)

                    logger.debug(f"Extracted text features from language_model, shape: {text_features.shape}")
                    return text_features

                # 如果没有hidden_states，尝试其他属性
                if hasattr(text_outputs, 'pooler_output'):
                    logger.debug(
                        f"Extracted text features from language_model.pooler_output, shape: {text_outputs.pooler_output.shape}")
                    return text_outputs.pooler_output

                if hasattr(text_outputs, 'last_hidden_state'):
                    last_hidden_state = text_outputs.last_hidden_state
                    if attention_mask is not None:
                        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                        text_features = sum_embeddings / sum_mask
                    else:
                        text_features = last_hidden_state.mean(dim=1)
                    logger.debug(
                        f"Extracted text features from language_model.last_hidden_state, shape: {text_features.shape}")
                    return text_features

            # 方法3: 使用语言模型的pooler输出（如果有）
            if hasattr(self.model, 'text_pooler'):
                with torch.no_grad():
                    text_outputs = self.model.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                if hasattr(text_outputs, 'pooler_output'):
                    logger.debug(f"Extracted text features from text_pooler, shape: {text_outputs.pooler_output.shape}")
                    return text_outputs.pooler_output

            # 方法4: 对于没有明确语言模型的模型，尝试直接调用模型
            # 但注意：这可能会影响梯度，所以使用no_grad
            with torch.no_grad():
                text_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

            if hasattr(text_outputs, 'hidden_states') and len(text_outputs.hidden_states) > 0:
                last_hidden_state = text_outputs.hidden_states[-1]
                if attention_mask is not None:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                    sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                    text_features = sum_embeddings / sum_mask
                else:
                    text_features = last_hidden_state.mean(dim=1)
                logger.debug(f"Extracted text features from model.hidden_states, shape: {text_features.shape}")
                return text_features

            if hasattr(text_outputs, 'last_hidden_state'):
                last_hidden_state = text_outputs.last_hidden_state
                if attention_mask is not None:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                    sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                    text_features = sum_embeddings / sum_mask
                else:
                    text_features = last_hidden_state.mean(dim=1)
                logger.debug(f"Extracted text features from model.last_hidden_state, shape: {text_features.shape}")
                return text_features

            if hasattr(text_outputs, 'pooler_output'):
                logger.debug(
                    f"Extracted text features from model.pooler_output, shape: {text_outputs.pooler_output.shape}")
                return text_outputs.pooler_output

            # 方法5: 对于多模态模型，可能有单独的文本编码器
            text_attrs = ['text_model', 'text_encoder', 'bert', 'roberta']
            for attr in text_attrs:
                if hasattr(self.model, attr):
                    try:
                        text_module = getattr(self.model, attr)
                        with torch.no_grad():
                            text_outputs = text_module(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True
                            )

                        if hasattr(text_outputs, 'pooler_output'):
                            logger.debug(
                                f"Extracted text features from {attr}.pooler_output, shape: {text_outputs.pooler_output.shape}")
                            return text_outputs.pooler_output

                        if hasattr(text_outputs, 'last_hidden_state'):
                            last_hidden_state = text_outputs.last_hidden_state
                            if attention_mask is not None:
                                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                                text_features = sum_embeddings / sum_mask
                            else:
                                text_features = last_hidden_state.mean(dim=1)
                            logger.debug(
                                f"Extracted text features from {attr}.last_hidden_state, shape: {text_features.shape}")
                            return text_features
                    except Exception as e:
                        logger.debug(f"Failed to extract text features from {attr}: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Failed to extract text features: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

        logger.warning("Could not extract text features using any method")
        return None

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def compute_loss(
            self,
            model: "PreTrainedModel",
            inputs: Dict[str, "torch.Tensor"],
            return_outputs: bool = False,
            **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        计算总损失，包括SWFT损失和对比损失。
        """
        print("\n" + "=" * 80)
        print(f"[DEBUG] ===== 开始计算第 {self.state.global_step} 步的损失 =====")

        ### 1. 提取ratio值（如果有）
        ratio = inputs.pop("ratio", torch.tensor(1.0)).item()  # 获取SWFT权重比
        # ratio = 1.0
        # inputs.pop("ratio", None)  # 移除 ratio 键（如果存在），防止干扰后续逻辑
        print(f"[DEBUG] 1. ratio值: {ratio:.4f}, return_outputs: {return_outputs}")
        print(f"[DEBUG]   num_items_in_batch: {kwargs.get('num_items_in_batch')}")
        print(f"[DEBUG]   model_accepts_loss_kwargs: {getattr(self, 'model_accepts_loss_kwargs', False)}")
        print(f"[DEBUG]   gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")

        ### 调试：打印输入键名
        print(f"[DEBUG] 2. 输入键名: {list(inputs.keys())}")

        ### 2. 计算基础SWFT损失
        print(f"[DEBUG] 3. 开始计算基础SWFT损失...")
        loss_output = super().compute_loss(model, inputs, return_outputs, **kwargs)
        print(f"[DEBUG]   loss_output类型: {type(loss_output)}, 原始值: {loss_output}")

        if return_outputs:
            loss_swft, outputs = loss_output
            print(f"[DEBUG]   return_outputs=True路径")
            print(f"[DEBUG]   1原始SWFT损失: {loss_swft.item():.6f}")
            # 应用梯度累积调整
            if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
                loss_swft = loss_swft / self.args.gradient_accumulation_steps
                print(f"[DEBUG]   梯度累积调整后SWFT损失: {loss_swft.item():.6f} (除以{self.args.gradient_accumulation_steps})")
            # 应用ratio加权
            original_loss_swft = loss_swft.item()
            loss_swft = loss_swft * ratio
            print(f"[DEBUG]   ratio加权后SWFT损失: {loss_swft.item():.6f} (乘以ratio={ratio:.4f})")
        else:
            loss_swft = loss_output
            print(f"[DEBUG]   return_outputs=False路径")
            print(f"[DEBUG]   2原始SWFT损失: {loss_swft.item():.6f}")
            # # 应用梯度累积调整
            # if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
            #     loss_swft = loss_swft / self.args.gradient_accumulation_steps
            #     print(f"[DEBUG]   梯度累积调整后SWFT损失: {loss_swft.item():.6f} (除以{self.args.gradient_accumulation_steps})")
            # 应用ratio加权
            original_loss_swft = loss_swft.item()
            loss_swft = loss_swft * ratio/8.0
            loss_swft = loss_swft
            print(f"[DEBUG]   ratio加权后SWFT损失: {loss_swft.item():.6f} (乘以ratio={ratio:.4f})")

        ### 3. 计算对比损失（图像-地址、图像-描述）
        loss_contrastive = torch.tensor(0.0, device=loss_swft.device)
        print(f"[DEBUG] 4. 检查对比损失计算条件...")
        print(f"[DEBUG]   模型训练模式: {model.training}")
        print(f"[DEBUG]   应该计算对比损失: {self._should_compute_contrastive_loss()}")

        if model.training and self._should_compute_contrastive_loss():
            print(f"[DEBUG] 5. 开始计算对比损失...")
            # 计算原始对比损失
            contrastive_loss_raw = self._compute_contrastive_losses(inputs)
            print(f"[DEBUG]   原始对比损失值: {contrastive_loss_raw.item():.6f}")

            # 应用梯度累积调整（与SWFT损失保持一致）
            # 注意：这里我们始终应用梯度累积调整，不管 model_accepts_loss_kwargs 的值
            if kwargs.get("num_items_in_batch"):
                loss_contrastive = contrastive_loss_raw / self.args.gradient_accumulation_steps
                print(
                    f"[DEBUG]   22222222梯度累积调整后对比损失: {loss_contrastive.item():.6f} (除以{self.args.gradient_accumulation_steps})")
            else:
                loss_contrastive = contrastive_loss_raw
                print(f"[DEBUG]   11111梯度累积调整后对比损失: {loss_contrastive.item():.6f} (未调整)")
            # loss_contrastive  = ratio * loss_contrastive
            # loss_contrastive  = loss_contrastive
            # print(f"[DEBUG]   梯度累积调整后对比损失: {loss_contrastive.item():.6f} (未调整)")
            # print(f"[DEBUG]   最终对比损失值: {loss_contrastive.item():.6f}")

        ### 4. 融合总损失
        print(f"[DEBUG] 6. 融合总损失...")
        lambda1 = getattr(self.args, "swft_loss_weight", 1.0)
        print(f"[DEBUG]   SWFT损失权重(lambda1): {lambda1}")

        lambda2 = getattr(self.args, "img_addr_loss_weight", 0.0)
        lambda3 = getattr(self.args, "img_desc_loss_weight", 0.0)
        print(f"[DEBUG]   图像-地址对比损失权重(lambda2): {lambda2}")
        print(f"[DEBUG]   图像-描述对比损失权重(lambda3): {lambda3}")

        total_loss = lambda1 * loss_swft + loss_contrastive
        print(
            f"[DEBUG]   总损失计算: {lambda1} * {loss_swft.item():.6f} + {loss_contrastive.item():.6f} = {total_loss.item():.6f}")
        print(f"[DEBUG]   分解: SWFT贡献: {lambda1 * loss_swft.item():.6f}, 对比损失贡献: {loss_contrastive.item():.6f}")

        logger.debug(f"SWFT loss: {loss_swft.item() if isinstance(loss_swft, torch.Tensor) else loss_swft}")
        logger.debug(f"Contrastive loss: {loss_contrastive.item()}")
        logger.debug(f"Total loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")

        print(f"[DEBUG] ===== 第 {self.state.global_step} 步损失计算完成 =====")
        print("=" * 80 + "\n")

        if return_outputs:
            return (total_loss, outputs)
        else:
            return total_loss

    def _should_compute_contrastive_loss(self) -> bool:
        """检查是否应该计算对比损失"""
        # 检查是否有对比损失权重设置
        lambda2 = getattr(self.args, "img_addr_loss_weight", 0.0)
        lambda3 = getattr(self.args, "img_desc_loss_weight", 0.0)
        
        # 只要有一个对比损失权重大于0，就应该计算
        return lambda2 > 0 or lambda3 > 0

    def _compute_contrastive_losses(self, inputs: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """计算所有对比损失的总和"""
        total_contrastive_loss = torch.tensor(0.0, device=inputs["input_ids"].device)
        print(f"[DEBUG]   开始计算对比损失...")

        # 尝试获取视觉特征
        vision_features = self._extract_vision_features(inputs)
        if vision_features is None:
            logger.warning("Could not extract vision features. Skipping contrastive loss.")
            print(f"[DEBUG]   视觉特征提取失败，跳过对比损失")
            return total_contrastive_loss

        # 计算图像-地址对比损失
        lambda2 = getattr(self.args, "img_addr_loss_weight", 0.0)
        print(f"[DEBUG]   图像-地址对比损失权重(lambda2): {lambda2}")

        if lambda2 > 0:
            print(f"[DEBUG]   计算图像-地址对比损失...")
            addr_loss = self._compute_image_address_loss(vision_features, inputs)
            if addr_loss is not None:
                addr_contribution = lambda2 * addr_loss
                total_contrastive_loss = total_contrastive_loss + addr_contribution
                print(f"[DEBUG]     图像-地址对比损失: {addr_loss.item():.6f}")
                print(f"[DEBUG]     权重调整后贡献: {addr_contribution.item():.6f} (lambda2={lambda2})")
                print(f"[DEBUG]     当前总对比损失: {total_contrastive_loss.item():.6f}")

        # 计算图像-描述对比损失
        lambda3 = getattr(self.args, "img_desc_loss_weight", 0.0)
        print(f"[DEBUG]   图像-描述对比损失权重(lambda3): {lambda3}")

        if lambda3 > 0:
            print(f"[DEBUG]   计算图像-描述对比损失...")
            desc_loss = self._compute_image_description_loss(vision_features, inputs)
            if desc_loss is not None:
                desc_contribution = lambda3 * desc_loss
                total_contrastive_loss = total_contrastive_loss + desc_contribution
                print(f"[DEBUG]     图像-描述对比损失: {desc_loss.item():.6f}")
                print(f"[DEBUG]     权重调整后贡献: {desc_contribution.item():.6f} (lambda3={lambda3})")
                print(f"[DEBUG]     当前总对比损失: {total_contrastive_loss.item():.6f}")

        print(f"[DEBUG]   对比损失子项计算完成:")
        print(f"[DEBUG]     - 图像-地址对比损失: {addr_loss.item() if lambda2 > 0 and addr_loss is not None else 0:.6f}")
        print(f"[DEBUG]     - 图像-描述对比损失: {desc_loss.item() if lambda3 > 0 and desc_loss is not None else 0:.6f}")
        print(f"[DEBUG]     - 加权后总对比损失: {total_contrastive_loss.item():.6f}")

        return total_contrastive_loss

    def _extract_vision_features(self, inputs: Dict[str, "torch.Tensor"]) -> Optional["torch.Tensor"]:
        """从输入中提取视觉特征"""
        try:
            print(f"[DEBUG] Looking for vision features in inputs...")

            # 尝试从不同的键中获取像素值
            vision_keys = ["pixel_values", "vision_feats", "image_features"]

            found_key = None
            for key in vision_keys:
                if key in inputs:
                    print(f"[DEBUG] Found key '{key}' in inputs")
                    if inputs[key] is not None:
                        print(f"[DEBUG] Key '{key}' has shape: {inputs[key].shape}, dtype: {inputs[key].dtype}")
                        found_key = key
                        break

            if found_key is None:
                print("[WARNING] No vision features found in inputs")
                print(f"[DEBUG] All input keys: {list(inputs.keys())}")
                return None

            print(f"[DEBUG] Extracting vision features from key: {found_key}")
            print(f"[DEBUG] Model type: {type(self.model)}")

            # 直接使用内部方法提取视觉特征
            vision_features = self._get_vision_features(inputs[found_key])
            if vision_features is not None:
                print(f"[DEBUG] Vision features extracted successfully, shape: {vision_features.shape}, dtype: {vision_features.dtype}")
                return vision_features
            else:
                print("[WARNING] _get_vision_features returned None")

            return None

        except Exception as e:
            print(f"[ERROR] Failed to extract vision features: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return None

    def _compute_image_address_loss(self, vision_features: "torch.Tensor", inputs: Dict[str, "torch.Tensor"]) -> \
            Optional["torch.Tensor"]:
        """计算图像-地址对比损失"""
        try:
            print(f"[DEBUG]     提取地址特征...")

            # 尝试获取地址特征
            addr_input_keys = ["addr_input_ids", "address_input_ids", "addr_text_ids"]
            addr_mask_keys = ["addr_attention_mask", "address_attention_mask", "addr_text_mask"]

            addr_features = None
            found_input_key = None
            found_mask_key = None

            for input_key, mask_key in zip(addr_input_keys, addr_mask_keys):
                if input_key in inputs and mask_key in inputs:
                    print(f"[DEBUG]     找到地址键: {input_key}, {mask_key}")
                    if inputs[input_key] is not None and inputs[mask_key] is not None:
                        print(f"[DEBUG]       {input_key} 形状: {inputs[input_key].shape}, dtype: {inputs[input_key].dtype}")
                        print(f"[DEBUG]       {mask_key} 形状: {inputs[mask_key].shape}, dtype: {inputs[mask_key].dtype}")
                        found_input_key = input_key
                        found_mask_key = mask_key
                        break

            if found_input_key is None:
                print("[WARNING] No address features found in inputs")
                return None

            print(f"[DEBUG]     从键中提取地址特征: {found_input_key}, {found_mask_key}")

            # 直接使用内部方法提取文本特征
            addr_features = self._get_text_features(
                inputs[found_input_key],
                inputs[found_mask_key]
            )
            if addr_features is not None:
                print(f"[DEBUG]     地址特征提取成功, 形状: {addr_features.shape}, dtype: {addr_features.dtype}")
            else:
                print("[WARNING] _get_text_features returned None")
                return None

            # 确保投影层存在
            vision_dim = vision_features.shape[1]
            text_dim = addr_features.shape[1]
            self._ensure_projection_layers(vision_dim, text_dim, vision_features.device)

            # 投影地址特征到视觉特征维度
            addr_features_projected = self._project_text_features(addr_features, "address")
            print(f"[DEBUG]     地址特征投影从 {text_dim} 到 {addr_features_projected.shape[1]}, dtype: {addr_features_projected.dtype}")
            print(f"[DEBUG]     视觉特征 dtype: {vision_features.dtype}")

            # 确保数据类型匹配
            if addr_features_projected.dtype != vision_features.dtype:
                print(f"[DEBUG]     转换投影特征 dtype 从 {addr_features_projected.dtype} 到 {vision_features.dtype}")
                addr_features_projected = addr_features_projected.to(vision_features.dtype)

            # 确保批次大小匹配
            print(f"[DEBUG]     视觉特征批次大小: {vision_features.shape[0]}")
            print(f"[DEBUG]     地址特征批次大小: {addr_features_projected.shape[0]}")

            if vision_features.shape[0] != addr_features_projected.shape[0]:
                print(f"[WARNING] Batch size mismatch: vision {vision_features.shape[0]}, address {addr_features_projected.shape[0]}")
                return None

            # 计算对比损失
            temperature = getattr(self.args, "contrastive_temperature", 0.07)
            print(f"[DEBUG]     对比温度: {temperature}")
            loss = self._compute_contrastive_loss_pair(vision_features, addr_features_projected, temperature)

            print(f"[DEBUG]     图像-地址对比损失计算完成: {loss.item():.6f}")
            return loss

        except Exception as e:
            print(f"[ERROR] Failed to compute image-address loss: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return None

    def _compute_image_description_loss(self, vision_features: "torch.Tensor", inputs: Dict[str, "torch.Tensor"]) -> \
    Optional["torch.Tensor"]:
        """计算图像-描述对比损失"""
        try:
            print(f"[DEBUG] Looking for description features in inputs...")
            print(f"[DEBUG] Available keys: {list(inputs.keys())}")
            if "desc_input_ids" not in inputs:
                print(f"[ERROR] desc_input_ids not found in inputs!")
                print(f"[ERROR] Available keys with 'desc': {[k for k in inputs.keys() if 'desc' in k]}")
                return None
            if "desc_attention_mask" not in inputs:
                print(f"[ERROR] desc_attention_mask not found in inputs!")
                return None
            print(f"[DEBUG] desc_input_ids shape: {inputs['desc_input_ids'].shape}")
            print(f"[DEBUG] desc_attention_mask shape: {inputs['desc_attention_mask'].shape}")
            print(f"[DEBUG] desc_input_ids first few: {inputs['desc_input_ids'][0][:10] if inputs['desc_input_ids'].shape[1] > 0 else 'empty'}")

            print(f"[DEBUG]     提取描述特征...")

            # 尝试获取描述特征
            desc_input_keys = ["desc_input_ids", "description_input_ids", "desc_text_ids"]
            desc_mask_keys = ["desc_attention_mask", "description_attention_mask", "desc_text_mask"]

            desc_features = None
            found_input_key = None
            found_mask_key = None

            for input_key, mask_key in zip(desc_input_keys, desc_mask_keys):
                if input_key in inputs and mask_key in inputs:
                    print(f"[DEBUG]     找到描述键: {input_key}, {mask_key}")
                    if inputs[input_key] is not None and inputs[mask_key] is not None:
                        print(f"[DEBUG]       {input_key} 形状: {inputs[input_key].shape}")
                        print(f"[DEBUG]       {mask_key} 形状: {inputs[mask_key].shape}")
                        found_input_key = input_key
                        found_mask_key = mask_key
                        break

            if found_input_key is None:
                print("[WARNING] No description features found in inputs")
                return None

            print(f"[DEBUG]     从键中提取描述特征: {found_input_key}, {found_mask_key}")

            # 直接使用内部方法提取文本特征
            desc_features = self._get_text_features(
                inputs[found_input_key],
                inputs[found_mask_key]
            )
            if desc_features is not None:
                print(f"[DEBUG]     描述特征提取成功, 形状: {desc_features.shape}")
            else:
                print("[WARNING] _get_text_features returned None")
                return None

            # 确保投影层存在
            vision_dim = vision_features.shape[1]
            text_dim = desc_features.shape[1]
            self._ensure_projection_layers(vision_dim, text_dim, vision_features.device)

            # 投影描述特征到视觉特征维度
            desc_features_projected = self._project_text_features(desc_features, "description")
            print(f"[DEBUG]     描述特征投影从 {text_dim} 到 {desc_features_projected.shape[1]}")

            # 确保批次大小匹配
            print(f"[DEBUG]     视觉特征批次大小: {vision_features.shape[0]}")
            print(f"[DEBUG]     描述特征批次大小: {desc_features_projected.shape[0]}")

            if vision_features.shape[0] != desc_features_projected.shape[0]:
                print(f"[WARNING] Batch size mismatch: vision {vision_features.shape[0]}, description {desc_features_projected.shape[0]}")
                return None

            # 计算对比损失
            temperature = getattr(self.args, "contrastive_temperature", 0.07)
            print(f"[DEBUG]     对比温度: {temperature}")
            loss = self._compute_contrastive_loss_pair(vision_features, desc_features_projected, temperature)

            print(f"[DEBUG]     图像-描述对比损失计算完成: {loss.item():.6f}")
            return loss

        except Exception as e:
            print(f"[ERROR] Failed to compute image-description loss: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return None

    def _compute_contrastive_loss_pair(self, features1: "torch.Tensor", features2: "torch.Tensor",
                                       tau: float) -> "torch.Tensor":
        """计算一对特征之间的对比损失"""
        # 检查维度是否匹配
        if features1.shape[1] != features2.shape[1]:
            print(f"[WARNING] Feature dimension mismatch: {features1.shape} vs {features2.shape}")
            # 这里可以添加投影逻辑，但现在我们先跳过
            return torch.tensor(0.0, device=features1.device)

        # 确保特征数据类型一致
        if features1.dtype != features2.dtype:
            print(f"[DEBUG] Converting features2 dtype from {features2.dtype} to {features1.dtype}")
            features2 = features2.to(features1.dtype)

        # 归一化特征
        features1_norm = F.normalize(features1, dim=-1)
        features2_norm = F.normalize(features2, dim=-1)

        # 计算相似度矩阵
        logits = torch.matmul(features1_norm, features2_norm.T) / tau
        batch_size = features1_norm.shape[0]

        # 创建标签
        labels = torch.arange(batch_size, device=features1_norm.device)

        # 计算双向交叉熵损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        # 返回平均损失
        return (loss_i2t + loss_t2i) / 2
    
    @override
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # 添加对比损失相关的列到签名列中
        contrastive_columns = ["ratio", "addr_input_ids", "addr_attention_mask", 
                              "desc_input_ids", "desc_attention_mask", "pixel_values"]
        self._signature_columns = list(set(self._signature_columns + contrastive_columns))
        logger.debug(f"Signature columns: {self._signature_columns}")

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        # 在预测步骤中移除对比损失相关的输入，以免干扰正常的预测
        prediction_inputs = {k: v for k, v in inputs.items() 
                           if not k.startswith(('addr_', 'desc_', 'pixel_values', 'vision_'))}
        
        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, prediction_inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            
            writer.write("\n".join(res))
# limitations under the License.

# import json
# import os
# from types import MethodType
# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
# from typing_extensions import override

# import numpy as np
# import torch
# from transformers import Seq2SeqTrainer

# from ...extras.constants import IGNORE_INDEX
# from ...extras.logging import get_logger
# from ..callbacks import PissaConvertCallback, SaveProcessorCallback
# from ..trainer_utils import create_custom_optimzer, create_custom_scheduler


# if TYPE_CHECKING:
#     from torch.utils.data import Dataset
#     from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
#     from transformers.trainer import PredictionOutput
#     from ...hparams import FinetuningArguments


# logger = get_logger(__name__)


# class CustomSeq2SeqTrainer(Seq2SeqTrainer):
#     r"""
#     Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
#     """

#     def __init__(
#         self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
#     ) -> None:
#         kwargs.pop("gen_kwargs", None)   # ← 新增这一行
#         super().__init__(**kwargs)
#         self.finetuning_args = finetuning_args

#         if processor is not None:
#             self.add_callback(SaveProcessorCallback(processor))

#         if finetuning_args.pissa_convert:
#             self.add_callback(PissaConvertCallback)

#         if finetuning_args.use_badam:
#             from badam import BAdamCallback, clip_grad_norm_old_version

#             self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
#             self.add_callback(BAdamCallback)

#     def create_optimizer(self) -> "torch.optim.Optimizer":
#         if self.optimizer is None:
#             self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
#         return super().create_optimizer()

#     def create_scheduler(
#         self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
#     ) -> "torch.optim.lr_scheduler.LRScheduler":
#         create_custom_scheduler(self.args, num_training_steps, optimizer)
#         return super().create_scheduler(num_training_steps, optimizer)


#     @override
#     def compute_loss(
#         self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
#     ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
#         r"""
#         Fixes the loss value. See https://github.com/huggingface/transformers/pull/35438 for details.

#         It should be removed after https://github.com/huggingface/transformers/pull/35651 is merged.
#         """
#         ### change by lth on 20250224
#         # print("inputs", inputs)
#         ratio = inputs.pop("ratio").item()
#         # print("ratio", ratio)
#         loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
#         # print("loss_before", loss)
#         if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
#             # print("change loss before going")
#             if return_outputs:
#                 loss = (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
#             else:
#                 loss = loss / self.args.gradient_accumulation_steps

#         if return_outputs:
#             # print("return outputs")
#             loss = (loss[0] * ratio, *loss[1:])
#         else:
#             # print("not return outputs")
#             loss = loss * ratio
#         # print("loss_after", loss)
#         return loss
    
#     @override
#     def _set_signature_columns_if_needed(self):
#         super()._set_signature_columns_if_needed()
#         self._signature_columns = list(set(self._signature_columns + ["ratio"]))


#     def prediction_step(
#         self,
#         model: "torch.nn.Module",
#         inputs: Dict[str, Union[torch.Tensor, Any]],
#         prediction_loss_only: bool,
#         ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         r"""
#         Removes the prompt part in the generated tokens.

#         Subclass and override to inject custom behavior.
#         """
#         labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
#         if self.args.predict_with_generate:
#             assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
#             prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
#             if prompt_len > label_len:
#                 inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
#             if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
#                 inputs["labels"] = inputs["labels"][:, :prompt_len]

#         loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
#             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
#         )
#         if generated_tokens is not None and self.args.predict_with_generate:
#             generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
#             generated_tokens = generated_tokens.contiguous()

#         return loss, generated_tokens, labels

#     def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
#         r"""
#         Pads the tensor to the same length as the target tensor.
#         """
#         assert self.tokenizer.pad_token_id is not None, "Pad token is required."
#         padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
#         padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
#         return padded_tensor.contiguous()  # in contiguous memory

#     def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
#         r"""
#         Saves model predictions to `output_dir`.

#         A custom behavior that not contained in Seq2SeqTrainer.
#         """
#         if not self.is_world_process_zero():
#             return

#         output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
#         logger.info(f"Saving prediction results to {output_prediction_file}")

#         labels = np.where(
#             predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
#         )
#         preds = np.where(
#             predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
#         )

#         for i in range(len(preds)):
#             pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
#             if len(pad_len):  # move pad token to last
#                 preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

#         decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
#         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

#         with open(output_prediction_file, "w", encoding="utf-8") as writer:
#             res: List[str] = []
#             for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
#                 res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

#             writer.write("\n".join(res))