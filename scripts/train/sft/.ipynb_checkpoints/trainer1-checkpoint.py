# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
        kwargs.pop("gen_kwargs", None)   # ← 新增这一行
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        
        
        # 检查是否需要包装模型以支持对比特征提取
        self._wrap_model_for_contrastive_learning()
        
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

    def _wrap_model_for_contrastive_learning(self):
        """包装模型以支持对比特征提取"""
        try:
            # 尝试导入自定义模型包装器
            from ...model.custom_model import CustomLlavaModel

            # 检查模型是否已经被包装
            if not isinstance(self.model, CustomLlavaModel):
                print(f"[DEBUG] Wrapping model. Current model type: {type(self.model)}")
                print(f"[DEBUG] Model attributes: {dir(self.model)[:20]}...")  # 打印前20个属性

                # 检查模型是否有视觉编码器和文本编码器
                has_vision_encoder = hasattr(self.model, 'vision_tower') or hasattr(self.model, 'visual_encoder')
                has_text_encoder = hasattr(self.model, 'language_model') or hasattr(self.model, 'text_encoder')
                print(f"[DEBUG] Has vision encoder: {has_vision_encoder}")
                print(f"[DEBUG] Has text encoder: {has_text_encoder}")

                original_model = self.model
                self.model = CustomLlavaModel(original_model)
                print(f"[DEBUG] Model wrapped successfully. New model type: {type(self.model)}")

                # 检查包装后的模型是否有特征提取方法
                has_vision_features = hasattr(self.model, 'get_vision_features')
                has_text_features = hasattr(self.model, 'get_text_features')
                print(f"[DEBUG] Wrapped model has get_vision_features: {has_vision_features}")
                print(f"[DEBUG] Wrapped model has get_text_features: {has_text_features}")
            else:
                print("[DEBUG] Model is already wrapped with CustomLlavaModel.")

        except ImportError as e:
            print(f"[ERROR] Cannot import CustomLlavaModel: {e}")
            print("[WARNING] Contrastive learning features will be disabled.")

            # 尝试检查原始模型是否有特征提取方法
            has_vision_features = hasattr(self.model, 'get_vision_features')
            has_text_features = hasattr(self.model, 'get_text_features')
            print(f"[DEBUG] Original model has get_vision_features: {has_vision_features}")
            print(f"[DEBUG] Original model has get_text_features: {has_text_features}")

            if has_vision_features and has_text_features:
                print("[INFO] Original model already has feature extraction methods.")
        except Exception as e:
            print(f"[ERROR] Failed to wrap model: {e}")
            print("[WARNING] Contrastive learning features will be disabled.")

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
        ### 1. 提取ratio值（如果有）
        ratio = inputs.pop("ratio", torch.tensor(1.0)).item()  # 获取SWFT权重比
        print("ratio", ratio)

        ### 调试：打印输入键名
        print(f"[DEBUG] Input keys: {list(inputs.keys())}")

        ### 2. 计算基础SWFT损失
        loss_output = super().compute_loss(model, inputs, return_outputs, **kwargs)
        print("loss_output", loss_output)

        if return_outputs:
            loss_swft, outputs = loss_output
            # 应用梯度累积调整
            if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
                loss_swft = loss_swft / self.args.gradient_accumulation_steps
            # 应用ratio加权
            loss_swft = loss_swft * ratio
        else:
            loss_swft = loss_output
            # 应用梯度累积调整
            if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
                loss_swft = loss_swft / self.args.gradient_accumulation_steps
            # 应用ratio加权
            loss_swft = loss_swft * ratio

        ### 3. 计算对比损失（图像-地址、图像-描述）
        loss_contrastive = torch.tensor(0.0, device=loss_swft.device)
        print("loss_contrastive", loss_contrastive)

        # 检查是否处于训练阶段并启用了对比损失
        print(f"[DEBUG] Model training mode: {model.training}")
        print(f"[DEBUG] Should compute contrastive: {self._should_compute_contrastive_loss()}")

        if model.training and self._should_compute_contrastive_loss():
            print("[DEBUG] Computing contrastive losses...")
            loss_contrastive = self._compute_contrastive_losses(inputs)
            print(f"[DEBUG] Contrastive loss computed: {loss_contrastive.item()}")

        ### 4. 融合总损失
        lambda1 = getattr(self.args, "swft_loss_weight", 1.0)
        print("lambda1", lambda1)
        total_loss = lambda1 * loss_swft + loss_contrastive
        print("total_loss", total_loss)

        logger.debug(f"SWFT loss: {loss_swft.item() if isinstance(loss_swft, torch.Tensor) else loss_swft}")
        logger.debug(f"Contrastive loss: {loss_contrastive.item()}")
        logger.debug(f"Total loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")

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
        
        # 检查模型是否有特征提取方法
        has_vision_features = hasattr(self.model, 'get_vision_features')
        has_text_features = hasattr(self.model, 'get_text_features')
        
        if not (has_vision_features and has_text_features):
            logger.warning("Model does not have required feature extraction methods for contrastive learning.")
            return total_contrastive_loss
        
        # 尝试获取视觉特征
        vision_features = self._extract_vision_features(inputs)
        if vision_features is None:
            logger.warning("Could not extract vision features. Skipping contrastive loss.")
            return total_contrastive_loss
        
        # 计算图像-地址对比损失
        lambda2 = getattr(self.args, "img_addr_loss_weight", 0.0)
        if lambda2 > 0:
            addr_loss = self._compute_image_address_loss(vision_features, inputs)
            print("addr_loss", addr_loss)
            if addr_loss is not None:
                total_contrastive_loss = total_contrastive_loss + lambda2 * addr_loss
        
        # 计算图像-描述对比损失
        lambda3 = getattr(self.args, "img_desc_loss_weight", 0.0)
        if lambda3 > 0:
            desc_loss = self._compute_image_description_loss(vision_features, inputs)
            print("desc_loss", desc_loss)

            if desc_loss is not None:
                total_contrastive_loss = total_contrastive_loss + lambda3 * desc_loss
        
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
                        print(f"[DEBUG] Key '{key}' has shape: {inputs[key].shape}")
                        found_key = key
                        break

            if found_key is None:
                print("[WARNING] No vision features found in inputs")
                print(f"[DEBUG] All input keys: {list(inputs.keys())}")
                return None

            print(f"[DEBUG] Extracting vision features from key: {found_key}")
            print(f"[DEBUG] Model type: {type(self.model)}")
            print(f"[DEBUG] Model has get_vision_features: {hasattr(self.model, 'get_vision_features')}")

            if hasattr(self.model, 'get_vision_features'):
                vision_features = self.model.get_vision_features(inputs[found_key])
                if vision_features is not None:
                    print(f"[DEBUG] Vision features extracted successfully, shape: {vision_features.shape}")
                    return vision_features
                else:
                    print("[WARNING] get_vision_features returned None")
            else:
                print("[WARNING] Model does not have get_vision_features method")

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
            print(f"[DEBUG] Looking for address features in inputs...")

            # 尝试获取地址特征
            addr_input_keys = ["addr_input_ids", "address_input_ids", "addr_text_ids"]
            addr_mask_keys = ["addr_attention_mask", "address_attention_mask", "addr_text_mask"]

            addr_features = None
            found_input_key = None
            found_mask_key = None

            for input_key, mask_key in zip(addr_input_keys, addr_mask_keys):
                if input_key in inputs and mask_key in inputs:
                    print(f"[DEBUG] Found address keys: {input_key}, {mask_key}")
                    if inputs[input_key] is not None and inputs[mask_key] is not None:
                        print(f"[DEBUG] {input_key} shape: {inputs[input_key].shape}")
                        print(f"[DEBUG] {mask_key} shape: {inputs[mask_key].shape}")
                        found_input_key = input_key
                        found_mask_key = mask_key
                        break

            if found_input_key is None:
                print("[WARNING] No address features found in inputs")
                return None

            print(f"[DEBUG] Extracting address features from keys: {found_input_key}, {found_mask_key}")
            print(f"[DEBUG] Model has get_text_features: {hasattr(self.model, 'get_text_features')}")

            if hasattr(self.model, 'get_text_features'):
                addr_features = self.model.get_text_features(
                    inputs[found_input_key],
                    inputs[found_mask_key]
                )
                if addr_features is not None:
                    print(f"[DEBUG] Address features extracted successfully, shape: {addr_features.shape}")
                else:
                    print("[WARNING] get_text_features returned None")
                    return None
            else:
                print("[WARNING] Model does not have get_text_features method")
                return None

            # 确保批次大小匹配
            print(f"[DEBUG] Vision features batch size: {vision_features.shape[0]}")
            print(f"[DEBUG] Address features batch size: {addr_features.shape[0]}")

            if vision_features.shape[0] != addr_features.shape[0]:
                print(
                    f"[WARNING] Batch size mismatch: vision {vision_features.shape[0]}, address {addr_features.shape[0]}")
                return None

            # 计算对比损失
            temperature = getattr(self.args, "contrastive_temperature", 0.07)
            print(f"[DEBUG] Contrastive temperature: {temperature}")
            loss = self._compute_contrastive_loss_pair(vision_features, addr_features, temperature)

            print(f"[DEBUG] Image-address contrastive loss: {loss.item()}")
            return loss

        except Exception as e:
            print(f"[ERROR] Failed to compute image-address loss: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return None

    def _compute_image_description_loss(self, vision_features: "torch.Tensor", inputs: Dict[str, "torch.Tensor"]) -> Optional["torch.Tensor"]:
        """计算图像-描述对比损失"""
        try:
            # 尝试获取描述特征
            desc_input_keys = ["desc_input_ids", "description_input_ids", "desc_text_ids"]
            desc_mask_keys = ["desc_attention_mask", "description_attention_mask", "desc_text_mask"]
            
            desc_features = None
            for input_key, mask_key in zip(desc_input_keys, desc_mask_keys):
                if input_key in inputs and mask_key in inputs:
                    if inputs[input_key] is not None and inputs[mask_key] is not None:
                        logger.debug(f"Extracting description features from keys: {input_key}, {mask_key}")
                        desc_features = self.model.get_text_features(
                            inputs[input_key],
                            inputs[mask_key]
                        )
                        if desc_features is not None:
                            logger.debug(f"Description features shape: {desc_features.shape}")
                            break
            
            if desc_features is None:
                logger.debug("No description features found in inputs")
                return None
            
            # 确保批次大小匹配
            if vision_features.shape[0] != desc_features.shape[0]:
                logger.warning(f"Batch size mismatch: vision {vision_features.shape[0]}, description {desc_features.shape[0]}")
                return None
            
            # 计算对比损失
            temperature = getattr(self.args, "contrastive_temperature", 0.07)
            loss = self._compute_contrastive_loss_pair(vision_features, desc_features, temperature)
            
            logger.debug(f"Image-description contrastive loss: {loss.item()}")
            return loss
            
        except Exception as e:
            logger.warning(f"Failed to compute image-description loss: {e}")
            return None

    def _compute_contrastive_loss_pair(self, features1: "torch.Tensor", features2: "torch.Tensor", tau: float) -> "torch.Tensor":
        """计算一对特征之间的对比损失"""
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

            # writer.write("\n".join(res))
 ensure_ascii=False))

            writer.write("\n".join(res))