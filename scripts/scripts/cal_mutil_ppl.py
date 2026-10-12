# # coding=utf-8
# # Copyright 2024 the LlamaFactory team.
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
#
# import os
# import json
# from dataclasses import dataclass
# from typing import Any, Dict, Literal, Optional, Sequence
#
# import fire
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
#
# # 在 cal_mutil_ppl.py 开头添加导入
# from llamafactory.data.template import TEMPLATES
# from llamafactory.data.mm_plugin import get_mm_plugin
# from llamafactory.data import get_dataset
# from llamafactory.extras.constants import IGNORE_INDEX
# from llamafactory.hparams import get_train_args
# from llamafactory.model import load_model, load_tokenizer
# from llamafactory.data.template import get_template_and_fix_tokenizer
# import setproctitle
# from transformers import AutoModel, AutoTokenizer
# from transformers import AutoModelForCausalLM, AutoProcessor, AutoImageProcessor
# import torch
# from PIL import Image
# import requests
# from io import BytesIO
#
# from transformers import modeling_utils
#
# # os.environ["LLAMAFACTORY_FORCE_SINGLE_GPU"] = "1"
#
# if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
#     modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
#
# # 在脚本开始时设置进程名
# setproctitle.setproctitle("@liutianhui")
#
#
# @dataclass
# class MultimodalDataCollator:
#     """
#     多模态数据整理器
#     处理图像-文本对，支持多模态输入
#     """
#
#     def __init__(self, tokenizer, image_processor=None, template=None, multimodal_mode=True):
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
#         self.template = template
#         self.multimodal_mode = multimodal_mode
#
#     def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         """
#         处理多模态数据批次
#         """
#         batch = {
#             'input_ids': [],
#             'attention_mask': [],
#             'labels': [],
#             'pixel_values': []  # 使用pixel_values作为键，这是HuggingFace的标准
#         }
#
#         for feature in features:
#             # 处理图像
#             if 'images' in feature and feature['images'] is not None:
#                 images = feature['images']
#                 if isinstance(images, list) and len(images) > 0:
#                     # 只处理第一张图像（多图像支持可以后续添加）
#                     img = images[0]
#                     if isinstance(img, str):
#                         # 如果是图像路径，加载图像
#                         try:
#                             if os.path.exists(img):
#                                 image = Image.open(img).convert('RGB')
#                             else:
#                                 # 尝试从URL加载
#                                 response = requests.get(img, timeout=10)
#                                 image = Image.open(BytesIO(response.content)).convert('RGB')
#
#                             if self.image_processor:
#                                 processed_img = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
#                             else:
#                                 # 如果没有图像处理器，使用简单的预处理
#                                 import torchvision.transforms as transforms
#                                 transform = transforms.Compose([
#                                     transforms.Resize((336, 336)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#                                 ])
#                                 processed_img = transform(image)
#
#                             batch['pixel_values'].append(processed_img)
#                         except Exception as e:
#                             print(f"Error loading image {img}: {e}")
#                             # 创建一个占位符张量
#                             batch['pixel_values'].append(torch.zeros(3, 336, 336))
#                     elif torch.is_tensor(img):
#                         batch['pixel_values'].append(img)
#                     else:
#                         batch['pixel_values'].append(torch.zeros(3, 336, 336))
#                 else:
#                     batch['pixel_values'].append(torch.zeros(3, 336, 336))
#             else:
#                 batch['pixel_values'].append(torch.zeros(3, 336, 336))
#
#             # 处理文本
#             batch['input_ids'].append(feature['input_ids'])
#             batch['attention_mask'].append(feature['attention_mask'])
#             batch['labels'].append(feature.get('labels', feature['input_ids']))
#
#         # 对文本部分进行填充
#         max_len = max(len(ids) for ids in batch['input_ids'])
#
#         padded_input_ids = []
#         padded_attention_mask = []
#         padded_labels = []
#
#         for i in range(len(batch['input_ids'])):
#             input_ids = batch['input_ids'][i]
#             attention_mask = batch['attention_mask'][i]
#             labels = batch['labels'][i]
#
#             # 填充
#             pad_len = max_len - len(input_ids)
#             padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
#             padded_attention_mask.append(attention_mask + [0] * pad_len)
#
#             # 对labels进行特殊处理，保持与输入相同长度
#             if len(labels) < max_len:
#                 padded_labels.append(labels + [IGNORE_INDEX] * (max_len - len(labels)))
#             else:
#                 padded_labels.append(labels[:max_len])
#
#         # 转换为张量
#         result = {
#             'input_ids': torch.tensor(padded_input_ids),
#             'attention_mask': torch.tensor(padded_attention_mask),
#             'labels': torch.tensor(padded_labels)
#         }
#
#         # 处理图像张量
#         if batch['pixel_values']:
#             # 堆叠所有图像
#             result['pixel_values'] = torch.stack(batch['pixel_values'])
#
#         return result
#
#
# def debug_dataset_loading(model_args, data_args, training_args, stage, tokenizer_module):
#     """调试数据集加载过程"""
#     print("=== 开始调试数据集加载 ===")
#     print(f"model_args: {model_args}")
#     print(f"data_args: {data_args}")
#     print(f"training_args: {training_args}")
#     print(f"stage: {stage}")
#     print(f"tokenizer_module keys: {list(tokenizer_module.keys())}")
#
#     # 检查tokenizer_module中的内容
#     for key, value in tokenizer_module.items():
#         print(f"tokenizer_module['{key}'] type: {type(value)}")
#         if hasattr(value, '__dict__'):
#             print(
#                 f"tokenizer_module['{key}'] attributes: {[attr for attr in dir(value) if not attr.startswith('_')][:10]}")
#
#     # 尝试逐步加载数据集
#     try:
#         print("=== 尝试获取数据集 ===")
#         dataset_result = get_dataset(
#             template=getattr(data_args, 'template', 'default'),
#             model_args=model_args,
#             data_args=data_args,
#             training_args=training_args,
#             stage=stage,
#             **tokenizer_module
#         )
#         print(f"数据集获取成功，返回结果类型: {type(dataset_result)}")
#         if isinstance(dataset_result, dict):
#             print(f"数据集结果keys: {list(dataset_result.keys())}")
#             if "train_dataset" in dataset_result:
#                 print(f"train_dataset类型: {type(dataset_result['train_dataset'])}")
#         return dataset_result
#     except Exception as e:
#         print(f"获取数据集时出错: {e}")
#         print(f"错误类型: {type(e)}")
#         import traceback
#         traceback.print_exc()
#         raise
#
#
# def cal_ppl(
#         model_name_or_path: str,
#         save_name: str,
#         ### batch size 不等同于训练时的 per_device_train_batch_size，这里是一张卡同时处理多少数据，最终结果是分开的
#         batch_size: int = 2,
#         stage: Literal["pt", "sft", "rm"] = "sft",
#         dataset: str = "alpaca_en",
#         dataset_dir: str = "data",
#         template: str = "default",
#         # cutoff_len需与训练时保持一致
#         cutoff_len: int = 4096,
#         # 一个很大的数
#         max_samples: Optional[int] = 100000000000000000000000000000000000000000000,
#         train_on_prompt: bool = False,
#         # 多模态相关参数 - 使用LLaMA-Factory支持的参数名
#         multimodal: bool = False,
#         # 图像最大像素数
#         image_max_pixels: int = 1024 * 1024,
#         # 图像最小像素数
#         image_min_pixels: int = 32 * 32,
# ):
#     r"""
#     Calculates the ppl on the dataset of the pre-trained models.
#     Support for multimodal models.
#
#     Usage for multimodal models:
#     python cal_mutil_ppl.py \
#       --model_name_or_path /path/to/llava1.5-7B \
#       --dataset test \
#       --template llava \
#       --multimodal True \
#       --save_name /path/to/save/ppl.json \
#       --dataset_dir /path/to/data
#
#     The relation between ppl and loss: loss = ln(ppl)
#     """
#     print("=== 开始参数解析 ===")
#
#     # 构建参数字典 - 使用LLaMA-Factory支持的参数名
#     args_dict = dict(
#         stage=stage,
#         model_name_or_path=model_name_or_path,
#         dataset=dataset,
#         dataset_dir=dataset_dir,
#         template=template,
#         cutoff_len=cutoff_len,
#         max_samples=max_samples,
#         train_on_prompt=train_on_prompt,
#         preprocessing_num_workers=1,  # 先设置为1以便调试
#         output_dir="dummy_dir",
#         overwrite_cache=True,
#         do_train=True,
#     )
#
#     # 关键：如果是多模态模型，强制使用llava模板
#     if multimodal:
#         # 首先尝试使用llava模板
#         args_dict['template'] = 'llava'
#         print(f"多模态模式，强制使用模板: llava")
#
#         # 尝试添加多模态参数
#         try:
#             args_dict['visual_inputs'] = True
#             print("添加 visual_inputs 参数")
#         except:
#             print("无法添加 visual_inputs 参数，将在后续手动设置")
#
#     try:
#         model_args, data_args, training_args, finetuning_args, _ = get_train_args(args_dict)
#         print("参数解析成功")
#     except Exception as e:
#         print(f"参数解析失败: {e}")
#         print("尝试使用最小参数集重新解析...")
#         # 使用最小参数集
#         minimal_args = {
#             'stage': stage,
#             'model_name_or_path': model_name_or_path,
#             'dataset': dataset,
#             'dataset_dir': dataset_dir,
#             'template': args_dict['template'],
#             'cutoff_len': cutoff_len,
#             'output_dir': "dummy_dir",
#             'do_train': True,
#         }
#         model_args, data_args, training_args, finetuning_args, _ = get_train_args(minimal_args)
#
#     print("=== 参数解析完成 ===")
#     print(f"模型路径: {model_args.model_name_or_path}")
#     print(f"数据集: {data_args.dataset}")
#     print(f"模板: {getattr(data_args, 'template', template)}")
#
#     # 关键：手动设置多模态标志
#     if multimodal:
#         # 确保模型参数包含视觉输入标志
#         if not hasattr(model_args, 'visual_inputs'):
#             model_args.visual_inputs = True
#         else:
#             model_args.visual_inputs = True
#
#         # 确保数据参数包含模板
#         if not hasattr(data_args, 'template') or data_args.template != 'llava':
#             data_args.template = 'llava'
#
#         print(f"手动设置: model_args.visual_inputs = {model_args.visual_inputs}")
#         print(f"手动设置: data_args.template = {data_args.template}")
#
#     # 加载tokenizer和处理器
#     tokenizer_module = load_tokenizer(model_args)
#     print(f"tokenizer_module类型: {type(tokenizer_module)}")
#     print(f"tokenizer_module包含: {list(tokenizer_module.keys())}")
#
#     tokenizer = tokenizer_module["tokenizer"]
#     print(f"tokenizer类型: {type(tokenizer)}")
#
#     # 获取处理器（对于多模态模型）
#     processor = tokenizer_module.get("processor")
#     if processor:
#         print(f"处理器类型: {type(processor)}")
#         # 检查处理器是否有图像处理器
#         if hasattr(processor, 'image_processor'):
#             print(f"图像处理器类型: {type(processor.image_processor)}")
#     else:
#         print("没有找到处理器，如果是多模态模型这可能是问题")
#
#     # 获取模板对象
#     print(f"Data args template: {getattr(data_args, 'template', 'Not found')}")
#
#     try:
#         template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
#         print(f"模板对象类型: {type(template_obj)}")
#         print(f"模板名称: {getattr(template_obj, 'name', 'unknown')}")
#
#         # 检查模板是否有多模态插件
#         if multimodal:
#             if not hasattr(template_obj, 'mm_plugin'):
#                 print("错误：模板没有多模态插件！")
#                 print(f"模板对象属性: {[attr for attr in dir(template_obj) if not attr.startswith('_')]}")
#
#                 # 尝试直接创建llava模板
#                 from llamafactory.data.template import TEMPLATES
#                 if 'llava' in TEMPLATES:
#                     print("尝试直接创建llava模板...")
#                     template_info = TEMPLATES['llava']
#                     # 创建模板实例
#                     template_obj = template_info['class'](
#                         tokenizer=tokenizer,
#                         **template_info.get('kwargs', {})
#                     )
#                     template_obj.name = 'llava'
#                     print(f"重新创建模板: {template_obj.name}")
#             else:
#                 print(f"模板有多模态插件: {type(template_obj.mm_plugin)}")
#                 # 检查插件属性
#                 if hasattr(template_obj.mm_plugin, 'image_token'):
#                     print(f"图像令牌: {template_obj.mm_plugin.image_token}")
#
#     except Exception as e:
#         print(f"模板处理异常: {e}")
#         import traceback
#         traceback.print_exc()
#         # 回退到默认模板
#         template_obj = template
#
#     # 更新data_args中的模板
#     data_args.template = template_obj
#     print(f"最终模板: {getattr(data_args.template, 'name', 'unknown')}")
#
#     print("=== 开始加载数据集 ===")
#
#     # 准备数据集参数
#     dataset_kwargs = {
#         'template': data_args.template,
#         'model_args': model_args,
#         'data_args': data_args,
#         'training_args': training_args,
#         'stage': stage,
#     }
#
#     # 确保传递正确的参数
#     dataset_kwargs['tokenizer'] = tokenizer
#     if processor:
#         dataset_kwargs['processor'] = processor
#
#     # 尝试加载数据集
#     try:
#         dataset_result = get_dataset(**dataset_kwargs)
#
#         if isinstance(dataset_result, dict) and "train_dataset" in dataset_result:
#             trainset = dataset_result["train_dataset"]
#             print(f"从字典获取训练集")
#         else:
#             trainset = dataset_result
#             print(f"直接获取训练集")
#
#         print(f"数据集类型: {type(trainset)}")
#         if hasattr(trainset, '__len__'):
#             print(f"训练集样本数量: {len(trainset)}")
#
#     except Exception as e:
#         print(f"数据集加载失败: {e}")
#         import traceback
#         traceback.print_exc()
#
#         # 尝试手动加载数据
#         print("尝试手动加载数据集...")
#         try:
#             import json
#             from datasets import Dataset
#
#             # 查找数据文件
#             data_file = os.path.join(dataset_dir, dataset, "test.json")
#             if not os.path.exists(data_file):
#                 # 尝试其他可能的路径
#                 data_file = os.path.join(dataset_dir, f"{dataset}.json")
#
#             if os.path.exists(data_file):
#                 with open(data_file, 'r') as f:
#                     data_list = json.load(f)
#
#                 print(f"从文件加载数据: {data_file}")
#                 print(f"样本数量: {len(data_list)}")
#
#                 # 转换为Dataset
#                 trainset = Dataset.from_list(data_list)
#                 print(f"手动创建数据集成功，样本数: {len(trainset)}")
#             else:
#                 raise FileNotFoundError(f"数据文件不存在: {data_file}")
#
#         except Exception as e2:
#             print(f"手动加载也失败: {e2}")
#             raise
#
#     print("=== 开始加载模型 ===")
#
#     # 根据是否多模态选择不同的加载方式
#     if multimodal:
#         print("加载多模态模型...")
#         try:
#             # 方法1：使用LLaMA-Factory的load_model
#             model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
#             print("=== 使用LLaMA-Factory加载多模态模型成功 ===")
#
#             # 检查模型是否有多模态能力
#             model_forward_params = model.forward.__code__.co_varnames
#             print(f"模型前向参数: {model_forward_params[:20]}...")
#
#             if hasattr(model, 'vision_tower'):
#                 print("模型具有视觉塔 (vision_tower)")
#             else:
#                 print("模型可能没有视觉塔，但可能以其他方式支持多模态")
#
#         except Exception as e:
#             print(f"LLaMA-Factory加载失败: {e}")
#             print("尝试使用标准方式加载...")
#             try:
#                 # 方法2：使用标准的CausalLM加载
#                 model = AutoModelForCausalLM.from_pretrained(
#                     model_name_or_path,
#                     torch_dtype=torch.bfloat16,
#                     device_map="auto",
#                     trust_remote_code=True,
#                     attn_implementation="eager"
#                 )
#                 print("=== 使用标准方式加载模型成功 ===")
#             except Exception as e2:
#                 print(f"标准方式加载也失败: {e2}")
#                 print("尝试使用float16加载...")
#                 try:
#                     model = AutoModelForCausalLM.from_pretrained(
#                         model_name_or_path,
#                         torch_dtype=torch.float16,
#                         device_map="auto",
#                         trust_remote_code=True,
#                         attn_implementation="eager"
#                     )
#                     print("=== 使用float16加载模型成功 ===")
#                 except Exception as e3:
#                     print(f"所有加载方式都失败: {e3}")
#                     raise
#     else:
#         # 非多模态模型
#         try:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name_or_path,
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 attn_implementation="eager"
#             )
#             print("=== 标准模型加载完成 ===")
#         except Exception as e:
#             print(f"直接加载失败: {e}")
#             print("尝试使用LLaMA-Factory的load_model...")
#             try:
#                 model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
#                 print("=== 使用LLaMA-Factory加载模型成功 ===")
#             except Exception as e2:
#                 print(f"LLaMA-Factory加载也失败: {e2}")
#                 raise
#
#     # 创建数据整理器
#     if multimodal:
#         # 尝试获取图像处理器
#         image_processor = None
#         if processor and hasattr(processor, 'image_processor'):
#             image_processor = processor.image_processor
#             print(f"使用图像处理器: {type(image_processor)}")
#         elif hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'image_processor'):
#             image_processor = model.vision_tower.image_processor
#             print(f"从视觉塔获取图像处理器: {type(image_processor)}")
#
#         data_collator = MultimodalDataCollator(
#             tokenizer=tokenizer,
#             image_processor=image_processor,
#             template=template_obj,
#             multimodal_mode=True
#         )
#         print("创建多模态数据整理器")
#     elif stage == "pt":
#         data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#     elif stage == "sft":
#         data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
#     else:
#         raise NotImplementedError("Stage does not supported: {}.".format(stage))
#
#     print("=== 调试数据集样本 ===")
#
#     def debug_dataset_samples(dataset, num_samples=3):
#         """调试数据集样本"""
#         print(f"\n正在检查前 {num_samples} 个样本...")
#
#         for i in range(min(num_samples, len(dataset))):
#             print(f"\n{'=' * 50}")
#             print(f"样本 {i} 详细分析")
#             print(f"{'=' * 50}")
#
#             try:
#                 sample = dataset[i]
#
#                 # 1. 基本信息
#                 print("📋 基本信息:")
#                 print(f"  所有字段: {list(sample.keys())}")
#
#                 # 2. 检查多模态字段
#                 if 'images' in sample:
#                     print(f"\n🖼️  图像字段:")
#                     images = sample['images']
#                     if isinstance(images, list):
#                         print(f"  图像列表长度: {len(images)}")
#                         for j, img in enumerate(images[:2]):  # 只显示前两个
#                             if isinstance(img, str):
#                                 print(f"    图像 {j}: 路径 - {img}")
#                                 # 检查路径是否存在
#                                 if os.path.exists(img):
#                                     print(f"      路径存在: 是")
#                                 else:
#                                     print(f"      路径存在: 否")
#                             elif torch.is_tensor(img):
#                                 print(f"    图像 {j}: 张量形状 - {img.shape}")
#                 else:
#                     print(f"\n🖼️  没有找到图像字段")
#
#                 # 3. 检查文本字段
#                 fields_to_check = ['input_ids', 'attention_mask', 'labels']
#                 for field in fields_to_check:
#                     print(f"\n🔍 分析字段 '{field}':")
#
#                     if field not in sample:
#                         print(f"  ❌ 字段不存在")
#                         continue
#
#                     value = sample[field]
#                     print(f"  类型: {type(value)}")
#
#                     if value is None:
#                         print(f"  ❌ 值为 None")
#                         continue
#
#                     if hasattr(value, '__len__'):
#                         print(f"  长度: {len(value)}")
#                         if len(value) > 0:
#                             print(f"  前10个元素: {value[:10]}")
#
#                 # 4. 尝试解码查看文本内容
#                 print(f"\n📝 文本内容预览:")
#                 try:
#                     if 'input_ids' in sample and sample['input_ids'] is not None:
#                         input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
#                         print(f"  input_ids解码: {input_text[:200]}...")
#                 except Exception as decode_error:
#                     print(f"  解码失败: {decode_error}")
#
#                 print(f"✅ 样本 {i} 检查完成")
#
#             except Exception as e:
#                 print(f"❌ 样本 {i} 检查失败: {e}")
#                 import traceback
#                 traceback.print_exc()
#
#         print(f"\n{'=' * 50}")
#         print("调试完成")
#         print(f"{'=' * 50}")
#
#     # 执行调试
#     debug_dataset_samples(trainset, num_samples=3)
#
#     print("=== 创建数据加载器 ===")
#     dataloader = DataLoader(trainset, batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
#     criterion = torch.nn.CrossEntropyLoss(reduction="none")
#     total_ppl = 0
#     perplexities = []
#     batch: Dict[str, "torch.Tensor"]
#
#     print("=== 开始计算困惑度 ===")
#
#     model.eval()  # 确保模型在评估模式
#
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(tqdm(dataloader, desc="计算困惑度")):
#             try:
#                 # 准备输入数据
#                 input_ids = batch['input_ids'].to(model.device)
#                 attention_mask = batch['attention_mask'].to(model.device)
#                 labels = batch['labels'].to(model.device)
#
#                 # 准备模型输入
#                 model_inputs = {
#                     'input_ids': input_ids,
#                     'attention_mask': attention_mask,
#                     'labels': labels,
#                 }
#
#                 # 如果有图像，添加到输入中
#                 if multimodal and 'pixel_values' in batch and batch['pixel_values'] is not None:
#                     pixel_values = batch['pixel_values'].to(model.device)
#
#                     # 检查模型是否接受pixel_values参数
#                     model_forward_params = model.forward.__code__.co_varnames
#                     if 'pixel_values' in model_forward_params:
#                         model_inputs['pixel_values'] = pixel_values
#                         print(f"批次 {batch_idx}: 添加像素值输入，形状: {pixel_values.shape}")
#                     elif 'images' in model_forward_params:
#                         model_inputs['images'] = pixel_values
#                         print(f"批次 {batch_idx}: 添加图像输入，形状: {pixel_values.shape}")
#                     elif hasattr(model, 'vision_tower'):
#                         # 对于LLaVA类模型，可能直接使用pixel_values
#                         model_inputs['pixel_values'] = pixel_values
#                         print(f"批次 {batch_idx}: 添加像素值到视觉塔，形状: {pixel_values.shape}")
#                     else:
#                         # 如果不支持像素值输入，尝试查找其他可能的参数名
#                         for param_name in ['vision_inputs', 'image_features', 'visual_features', 'image_patches']:
#                             if param_name in model_forward_params:
#                                 model_inputs[param_name] = pixel_values
#                                 print(f"批次 {batch_idx}: 添加 {param_name} 输入")
#                                 break
#                         else:
#                             print(f"批次 {batch_idx}: 模型不支持像素值输入，跳过图像")
#
#                 # 前向传播
#                 outputs = model(**model_inputs)
#
#                 # 计算损失和困惑度
#                 shift_logits = outputs.logits[..., :-1, :].contiguous()
#                 shift_labels = labels[..., 1:].contiguous()
#
#                 loss_mask = shift_labels != IGNORE_INDEX
#
#                 # 展平计算
#                 flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
#                 flatten_labels = shift_labels.view(-1)
#
#                 # 计算每个token的损失
#                 token_losses = criterion(flatten_logits, flatten_labels)
#
#                 # 重塑回批次维度
#                 token_losses = token_losses.view(shift_labels.size(0), -1)
#
#                 # 计算每个句子的平均损失（忽略IGNORE_INDEX）
#                 sentence_losses = (token_losses * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
#
#                 # 从损失计算困惑度：ppl = exp(loss)
#                 sentence_ppls = torch.exp(sentence_losses)
#
#                 total_ppl += sentence_ppls.sum().item()
#                 perplexities.extend(sentence_ppls.cpu().tolist())
#
#                 # 每10个批次打印一次进度
#                 if (batch_idx + 1) % 10 == 0:
#                     if perplexities:
#                         avg_ppl = sum(perplexities) / len(perplexities)
#                         print(f"批次 {batch_idx + 1}, 当前平均困惑度: {avg_ppl:.2f}")
#
#             except Exception as e:
#                 print(f"批次 {batch_idx} 处理失败: {e}")
#                 print(f"批次内容键: {list(batch.keys())}")
#                 if 'pixel_values' in batch:
#                     print(f"像素值形状: {batch['pixel_values'].shape}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
#
#     # 保存结果
#     with open(save_name, "w", encoding="utf-8") as f:
#         json.dump(perplexities, f, indent=2)
#
#     if perplexities:
#         avg_ppl = total_ppl / len(perplexities)
#         print(f"平均困惑度: {avg_ppl:.2f}")
#         print(f"最小困惑度: {min(perplexities):.2f}")
#         print(f"最大困惑度: {max(perplexities):.2f}")
#         print(f"总样本数: {len(perplexities)}")
#     else:
#         print("警告：没有计算出任何困惑度值")
#
#     print(f"困惑度已保存到: {save_name}")
#
#
# if __name__ == "__main__":
#     fire.Fire(cal_ppl)

# coding=utf-8
# Copyright 2024 the LlamaFactory team.
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

import os
import json  # 确保在文件顶部导入 json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

# 在 cal_mutil_ppl.py 开头添加导入
from llamafactory.data.template import TEMPLATES
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data import get_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
import setproctitle
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoProcessor, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO
from transformers import modeling_utils

# os.environ["LLAMAFACTORY_FORCE_SINGLE_GPU"] = "1"

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']

# 在脚本开始时设置进程名
setproctitle.setproctitle("@liutianhui")


@dataclass
class MultimodalDataCollator:
    """
    多模态数据整理器
    处理图像-文本对，支持多模态输入
    """

    def __init__(self, tokenizer, image_processor=None, template=None, multimodal_mode=True):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.template = template
        self.multimodal_mode = multimodal_mode

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理多模态数据批次
        """
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'pixel_values': []  # 使用pixel_values作为键，这是HuggingFace的标准
        }

        for feature in features:
            # 处理图像
            if 'images' in feature and feature['images'] is not None:
                images = feature['images']
                if isinstance(images, list) and len(images) > 0:
                    # 只处理第一张图像（多图像支持可以后续添加）
                    img = images[0]
                    if isinstance(img, str):
                        # 如果是图像路径，加载图像
                        try:
                            if os.path.exists(img):
                                image = Image.open(img).convert('RGB')
                            else:
                                # 尝试从URL加载
                                response = requests.get(img, timeout=10)
                                image = Image.open(BytesIO(response.content)).convert('RGB')

                            if self.image_processor:
                                processed_img = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
                            else:
                                # 如果没有图像处理器，使用简单的预处理
                                import torchvision.transforms as transforms
                                transform = transforms.Compose([
                                    transforms.Resize((336, 336)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
                                processed_img = transform(image)

                            batch['pixel_values'].append(processed_img)
                        except Exception as e:
                            print(f"Error loading image {img}: {e}")
                            # 创建一个占位符张量
                            batch['pixel_values'].append(torch.zeros(3, 336, 336))
                    elif torch.is_tensor(img):
                        batch['pixel_values'].append(img)
                    else:
                        batch['pixel_values'].append(torch.zeros(3, 336, 336))
                else:
                    batch['pixel_values'].append(torch.zeros(3, 336, 336))
            else:
                batch['pixel_values'].append(torch.zeros(3, 336, 336))

            # 处理文本
            # 检查特征中是否包含必要的字段
            if 'input_ids' in feature and 'attention_mask' in feature:
                batch['input_ids'].append(feature['input_ids'])
                batch['attention_mask'].append(feature['attention_mask'])
                batch['labels'].append(feature.get('labels', feature['input_ids']))
            else:
                # 如果没有处理过的特征，打印警告
                print(f"警告: 样本缺少输入字段，跳过: {feature.keys()}")
                continue

        if not batch['input_ids']:  # 如果没有有效样本
            return {}

        # 对文本部分进行填充
        max_len = max(len(ids) for ids in batch['input_ids'])

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i in range(len(batch['input_ids'])):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            labels = batch['labels'][i]

            # 填充
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)

            # 对labels进行特殊处理，保持与输入相同长度
            if len(labels) < max_len:
                padded_labels.append(labels + [IGNORE_INDEX] * (max_len - len(labels)))
            else:
                padded_labels.append(labels[:max_len])

        # 转换为张量
        result = {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(padded_attention_mask),
            'labels': torch.tensor(padded_labels)
        }

        # 处理图像张量
        if batch['pixel_values']:
            # 堆叠所有图像
            result['pixel_values'] = torch.stack(batch['pixel_values'])

        return result


def cal_ppl(
        model_name_or_path: str,
        save_name: str,
        ### batch size 不等同于训练时的 per_device_train_batch_size，这里是一张卡同时处理多少数据，最终结果是分开的
        batch_size: int = 2,
        stage: Literal["pt", "sft", "rm"] = "sft",
        dataset: str = "alpaca_en",
        dataset_dir: str = "data",
        template: str = "default",
        # cutoff_len需与训练时保持一致
        cutoff_len: int = 4096,
        # 一个很大的数
        max_samples: Optional[int] = 100000000000000000000000000000000000000000000,
        train_on_prompt: bool = False,
        # 多模态相关参数 - 使用LLaMA-Factory支持的参数名
        multimodal: bool = False,
        # 图像最大像素数
        image_max_pixels: int = 1024 * 1024,
        # 图像最小像素数
        image_min_pixels: int = 32 * 32,
):
    r"""
    Calculates the ppl on the dataset of the pre-trained models.
    Support for multimodal models.

    Usage for multimodal models:
    python cal_mutil_ppl.py \
      --model_name_or_path /path/to/llava1.5-7B \
      --dataset test \
      --template llava \
      --multimodal True \
      --save_name /path/to/save/ppl.json \
      --dataset_dir /path/to/data

    The relation between ppl and loss: loss = ln(ppl)
    """
    print("=== 开始参数解析 ===")

    # 构建参数字典 - 使用LLaMA-Factory支持的参数名
    args_dict = dict(
        stage=stage,
        model_name_or_path=model_name_or_path,
        dataset=dataset,
        dataset_dir=dataset_dir,
        template=template,
        cutoff_len=cutoff_len,
        max_samples=max_samples,
        train_on_prompt=train_on_prompt,
        preprocessing_num_workers=1,  # 先设置为1以便调试
        output_dir="dummy_dir",
        overwrite_cache=True,
        do_train=True,
    )

    # 关键：如果是多模态模型，强制使用llava模板
    if multimodal:
        # 首先尝试使用llava模板
        args_dict['template'] = 'llava'
        print(f"多模态模式，强制使用模板: llava")

    try:
        model_args, data_args, training_args, finetuning_args, _ = get_train_args(args_dict)
        print("参数解析成功")
    except Exception as e:
        print(f"参数解析失败: {e}")
        print("尝试使用最小参数集重新解析...")
        # 使用最小参数集
        minimal_args = {
            'stage': stage,
            'model_name_or_path': model_name_or_path,
            'dataset': dataset,
            'dataset_dir': dataset_dir,
            'template': args_dict['template'],
            'cutoff_len': cutoff_len,
            'output_dir': "dummy_dir",
            'do_train': True,
        }
        model_args, data_args, training_args, finetuning_args, _ = get_train_args(minimal_args)

    print("=== 参数解析完成 ===")
    print(f"模型路径: {model_args.model_name_or_path}")
    print(f"数据集: {data_args.dataset}")
    print(f"模板: {getattr(data_args, 'template', template)}")

    # 关键：手动设置多模态标志
    if multimodal:
        # 确保模型参数包含视觉输入标志
        if not hasattr(model_args, 'visual_inputs'):
            model_args.visual_inputs = True
        else:
            model_args.visual_inputs = True

        # 确保数据参数包含模板
        if not hasattr(data_args, 'template') or data_args.template != 'llava':
            data_args.template = 'llava'

        print(f"手动设置: model_args.visual_inputs = {model_args.visual_inputs}")
        print(f"手动设置: data_args.template = {data_args.template}")

    # 加载tokenizer和处理器
    tokenizer_module = load_tokenizer(model_args)
    print(f"tokenizer_module类型: {type(tokenizer_module)}")
    print(f"tokenizer_module包含: {list(tokenizer_module.keys())}")

    tokenizer = tokenizer_module["tokenizer"]
    print(f"tokenizer类型: {type(tokenizer)}")

    # 获取处理器（对于多模态模型）
    processor = tokenizer_module.get("processor")
    if processor:
        print(f"处理器类型: {type(processor)}")
        # 检查处理器是否有图像处理器
        if hasattr(processor, 'image_processor'):
            print(f"图像处理器类型: {type(processor.image_processor)}")
    else:
        print("没有找到处理器，如果是多模态模型这可能是问题")

    # 获取模板对象
    print(f"Data args template: {getattr(data_args, 'template', 'Not found')}")

    try:
        template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
        print(f"模板对象类型: {type(template_obj)}")
        print(f"模板名称: {getattr(template_obj, 'name', 'unknown')}")

        # 检查模板是否有多模态插件
        if multimodal:
            if not hasattr(template_obj, 'mm_plugin'):
                print("错误：模板没有多模态插件！")
                print(f"模板对象属性: {[attr for attr in dir(template_obj) if not attr.startswith('_')][:20]}")

                # 尝试直接创建llava模板
                from llamafactory.data.template import TEMPLATES
                if 'llava' in TEMPLATES:
                    print("尝试直接创建llava模板...")
                    template_info = TEMPLATES['llava']
                    # 创建模板实例
                    template_obj = template_info['class'](
                        tokenizer=tokenizer,
                        **template_info.get('kwargs', {})
                    )
                    template_obj.name = 'llava'
                    print(f"重新创建模板: {template_obj.name}")
            else:
                print(f"模板有多模态插件: {type(template_obj.mm_plugin)}")
                # 检查插件属性
                if hasattr(template_obj.mm_plugin, 'image_token'):
                    print(f"图像令牌: {template_obj.mm_plugin.image_token}")

    except Exception as e:
        print(f"模板处理异常: {e}")
        import traceback
        traceback.print_exc()
        # 回退到默认模板
        template_obj = None
        print("使用默认模板处理")

    # 更新data_args中的模板
    if template_obj:
        data_args.template = template_obj
        print(f"最终模板: {getattr(data_args.template, 'name', 'unknown')}")
    else:
        data_args.template = template
        print(f"最终模板: {data_args.template}")

    print("=== 开始加载数据集 ===")

    # 准备数据集参数
    dataset_kwargs = {
        'template': data_args.template,
        'model_args': model_args,
        'data_args': data_args,
        'training_args': training_args,
        'stage': stage,
    }

    # 确保传递正确的参数
    dataset_kwargs['tokenizer'] = tokenizer
    if processor:
        dataset_kwargs['processor'] = processor

    # 尝试加载数据集
    try:
        dataset_result = get_dataset(**dataset_kwargs)

        if isinstance(dataset_result, dict) and "train_dataset" in dataset_result:
            trainset = dataset_result["train_dataset"]
            print(f"从字典获取训练集")
        else:
            trainset = dataset_result
            print(f"直接获取训练集")

        print(f"数据集类型: {type(trainset)}")
        if hasattr(trainset, '__len__'):
            print(f"训练集样本数量: {len(trainset)}")

    except Exception as e:
        print(f"数据集加载失败: {e}")
        import traceback
        traceback.print_exc()

        # 尝试手动加载数据
        print("尝试手动加载数据集...")
        try:
            # 注意：这里我们移除了重复的 import json，使用文件顶部导入的 json
            from datasets import Dataset

            # 查找数据文件
            data_file = os.path.join(dataset_dir, dataset, "test.json")
            if not os.path.exists(data_file):
                # 尝试其他可能的路径
                data_file = os.path.join(dataset_dir, f"{dataset}.json")

            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)

                print(f"从文件加载数据: {data_file}")
                print(f"样本数量: {len(data_list)}")

                # 转换为Dataset
                trainset = Dataset.from_list(data_list)
                print(f"手动创建数据集成功，样本数: {len(trainset)}")
            else:
                raise FileNotFoundError(f"数据文件不存在: {data_file}")

        except Exception as e2:
            print(f"手动加载也失败: {e2}")
            raise

    print("=== 开始加载模型 ===")

    # 根据是否多模态选择不同的加载方式
    if multimodal:
        print("加载多模态模型...")
        try:
            # 方法1：使用LLaMA-Factory的load_model
            model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
            print("=== 使用LLaMA-Factory加载多模态模型成功 ===")

            # 检查模型是否有多模态能力
            model_forward_params = model.forward.__code__.co_varnames
            print(f"模型前向参数: {model_forward_params[:20]}...")

            if hasattr(model, 'vision_tower'):
                print("模型具有视觉塔 (vision_tower)")
            else:
                print("模型可能没有视觉塔，但可能以其他方式支持多模态")

        except Exception as e:
            print(f"LLaMA-Factory加载失败: {e}")
            print("尝试使用标准方式加载...")
            try:
                # 方法2：使用标准的CausalLM加载
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                print("=== 使用标准方式加载模型成功 ===")
            except Exception as e2:
                print(f"标准方式加载也失败: {e2}")
                print("尝试使用float16加载...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="eager"
                    )
                    print("=== 使用float16加载模型成功 ===")
                except Exception as e3:
                    print(f"所有加载方式都失败: {e3}")
                    raise
    else:
        # 非多模态模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
            print("=== 标准模型加载完成 ===")
        except Exception as e:
            print(f"直接加载失败: {e}")
            print("尝试使用LLaMA-Factory的load_model...")
            try:
                model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
                print("=== 使用LLaMA-Factory加载模型成功 ===")
            except Exception as e2:
                print(f"LLaMA-Factory加载也失败: {e2}")
                raise

    # 创建数据整理器
    if multimodal:
        # 尝试获取图像处理器
        image_processor = None
        if processor and hasattr(processor, 'image_processor'):
            image_processor = processor.image_processor
            print(f"使用图像处理器: {type(image_processor)}")
        elif hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'image_processor'):
            image_processor = model.vision_tower.image_processor
            print(f"从视觉塔获取图像处理器: {type(image_processor)}")

        data_collator = MultimodalDataCollator(
            tokenizer=tokenizer,
            image_processor=image_processor,
            template=template_obj,
            multimodal_mode=True
        )
        print("创建多模态数据整理器")
    elif stage == "pt":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif stage == "sft":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    else:
        raise NotImplementedError("Stage does not supported: {}.".format(stage))

    print("=== 调试数据集样本 ===")

    def debug_dataset_samples(dataset, num_samples=3):
        """调试数据集样本"""
        print(f"\n正在检查前 {num_samples} 个样本...")

        for i in range(min(num_samples, len(dataset))):
            print(f"\n{'=' * 50}")
            print(f"样本 {i} 详细分析")
            print(f"{'=' * 50}")

            try:
                sample = dataset[i]

                # 1. 基本信息
                print("📋 基本信息:")
                print(f"  所有字段: {list(sample.keys())}")

                # 2. 检查多模态字段
                if 'images' in sample:
                    print(f"\n🖼️  图像字段:")
                    images = sample['images']
                    if isinstance(images, list):
                        print(f"  图像列表长度: {len(images)}")
                        for j, img in enumerate(images[:2]):  # 只显示前两个
                            if isinstance(img, str):
                                print(f"    图像 {j}: 路径 - {img}")
                                # 检查路径是否存在
                                if os.path.exists(img):
                                    print(f"      路径存在: 是")
                                else:
                                    print(f"      路径存在: 否")
                            elif torch.is_tensor(img):
                                print(f"    图像 {j}: 张量形状 - {img.shape}")
                else:
                    print(f"\n🖼️  没有找到图像字段")

                # 3. 检查文本字段
                fields_to_check = ['input_ids', 'attention_mask', 'labels']
                for field in fields_to_check:
                    print(f"\n🔍 分析字段 '{field}':")

                    if field not in sample:
                        print(f"  ❌ 字段不存在")
                        continue

                    value = sample[field]
                    print(f"  类型: {type(value)}")

                    if value is None:
                        print(f"  ❌ 值为 None")
                        continue

                    if hasattr(value, '__len__'):
                        print(f"  长度: {len(value)}")
                        if len(value) > 0:
                            print(f"  前10个元素: {value[:10]}")

                # 4. 尝试解码查看文本内容
                print(f"\n📝 文本内容预览:")
                try:
                    if 'input_ids' in sample and sample['input_ids'] is not None:
                        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
                        print(f"  input_ids解码: {input_text[:200]}...")
                except Exception as decode_error:
                    print(f"  解码失败: {decode_error}")

                print(f"✅ 样本 {i} 检查完成")

            except Exception as e:
                print(f"❌ 样本 {i} 检查失败: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 50}")
        print("调试完成")
        print(f"{'=' * 50}")

    # 执行调试
    debug_dataset_samples(trainset, num_samples=3)

    print("=== 创建数据加载器 ===")

    # 检查数据集是否为空
    if len(trainset) == 0:
        print("警告：数据集为空！")
        perplexities = []
    else:
        dataloader = DataLoader(trainset, batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        total_ppl = 0
        perplexities = []
        batch: Dict[str, "torch.Tensor"]

        print("=== 开始计算困惑度 ===")

        model.eval()  # 确保模型在评估模式

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="计算困惑度")):
                try:
                    # 检查批次是否为空
                    if not batch:
                        print(f"批次 {batch_idx} 为空，跳过")
                        continue

                    # 准备输入数据
                    if 'input_ids' not in batch:
                        print(f"批次 {batch_idx} 缺少 input_ids，跳过")
                        continue

                    input_ids = batch['input_ids'].to(model.device)
                    attention_mask = batch['attention_mask'].to(model.device)
                    labels = batch['labels'].to(model.device)

                    # 准备模型输入
                    model_inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels,
                    }

                    # 如果有图像，添加到输入中
                    if multimodal and 'pixel_values' in batch and batch['pixel_values'] is not None:
                        pixel_values = batch['pixel_values'].to(model.device)

                        # 检查模型是否接受pixel_values参数
                        model_forward_params = model.forward.__code__.co_varnames
                        if 'pixel_values' in model_forward_params:
                            model_inputs['pixel_values'] = pixel_values
                            print(f"批次 {batch_idx}: 添加像素值输入，形状: {pixel_values.shape}")
                        elif 'images' in model_forward_params:
                            model_inputs['images'] = pixel_values
                            print(f"批次 {batch_idx}: 添加图像输入，形状: {pixel_values.shape}")
                        elif hasattr(model, 'vision_tower'):
                            # 对于LLaVA类模型，可能直接使用pixel_values
                            model_inputs['pixel_values'] = pixel_values
                            print(f"批次 {batch_idx}: 添加像素值到视觉塔，形状: {pixel_values.shape}")
                        else:
                            # 如果不支持像素值输入，尝试查找其他可能的参数名
                            for param_name in ['vision_inputs', 'image_features', 'visual_features', 'image_patches']:
                                if param_name in model_forward_params:
                                    model_inputs[param_name] = pixel_values
                                    print(f"批次 {batch_idx}: 添加 {param_name} 输入")
                                    break
                            else:
                                print(f"批次 {batch_idx}: 模型不支持像素值输入，跳过图像")

                    # 前向传播
                    outputs = model(**model_inputs)

                    # 计算损失和困惑度
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss_mask = shift_labels != IGNORE_INDEX

                    # 展平计算
                    flatten_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flatten_labels = shift_labels.view(-1)

                    # 计算每个token的损失
                    token_losses = criterion(flatten_logits, flatten_labels)

                    # 重塑回批次维度
                    token_losses = token_losses.view(shift_labels.size(0), -1)

                    # 计算每个句子的平均损失（忽略IGNORE_INDEX）
                    sentence_losses = (token_losses * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)

                    # 从损失计算困惑度：ppl = exp(loss)
                    sentence_ppls = torch.exp(sentence_losses)

                    total_ppl += sentence_ppls.sum().item()
                    perplexities.extend(sentence_ppls.cpu().tolist())

                    # 每10个批次打印一次进度
                    if (batch_idx + 1) % 10 == 0 and perplexities:
                        avg_ppl = sum(perplexities) / len(perplexities)
                        print(f"批次 {batch_idx + 1}, 当前平均困惑度: {avg_ppl:.2f}")

                except Exception as e:
                    print(f"批次 {batch_idx} 处理失败: {e}")
                    print(f"批次内容键: {list(batch.keys()) if batch else '空批次'}")
                    if 'pixel_values' in batch:
                        print(f"像素值形状: {batch['pixel_values'].shape}")
                    import traceback
                    traceback.print_exc()
                    continue

    # 保存结果
    try:
        with open(save_name, "w", encoding="utf-8") as f:
            json.dump(perplexities, f, indent=2)
        print(f"困惑度已保存到: {save_name}")
    except Exception as e:
        print(f"保存结果失败: {e}")

    if perplexities:
        avg_ppl = sum(perplexities) / len(perplexities)
        print(f"平均困惑度: {avg_ppl:.2f}")
        print(f"最小困惑度: {min(perplexities):.2f}")
        print(f"最大困惑度: {max(perplexities):.2f}")
        print(f"总样本数: {len(perplexities)}")
    else:
        print("警告：没有计算出任何困惑度值")


if __name__ == "__main__":
    fire.Fire(cal_ppl)