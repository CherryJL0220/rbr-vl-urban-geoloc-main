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
import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from llamafactory.data import get_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
import setproctitle
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

from transformers import modeling_utils


# os.environ["LLAMAFACTORY_FORCE_SINGLE_GPU"] = "1"


if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

# 在脚本开始时设置进程名
setproctitle.setproctitle("@liutianhui")


@dataclass
class FilteredDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator for seq2seq that filters out multimodal fields.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Filter out multimodal fields before padding.
        """
        # 过滤掉多模态字段
        filtered_features = []
        for feature in features:
            filtered_feature = {
                'input_ids': feature['input_ids'],
                'attention_mask': feature['attention_mask'],
                'labels': feature['labels']
            }
            filtered_features.append(filtered_feature)

        return super().__call__(filtered_features)


@dataclass
class FilteredDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    r"""
    Data collator for language modeling that filters out multimodal fields.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Filter out multimodal fields before padding.
        """
        # 过滤掉多模态字段
        filtered_features = []
        for feature in features:
            filtered_feature = {
                'input_ids': feature['input_ids'],
                'attention_mask': feature['attention_mask'],
                'labels': feature.get('labels', feature['input_ids'])  # 对于PT，labels可能不存在
            }
            filtered_features.append(filtered_feature)

        return super().__call__(filtered_features)



@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    train_on_prompt: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        chosen_features = []
        for feature in features:
            prompt_len, answer_len = len(feature["prompt_ids"]), len(feature["chosen_ids"])
            input_ids = feature["prompt_ids"] + feature["chosen_ids"]
            attention_mask = [1] * (prompt_len + answer_len)
            labels = input_ids if self.train_on_prompt else [IGNORE_INDEX] * prompt_len + feature["chosen_ids"]
            chosen_features.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

        return super().__call__(chosen_features)




def debug_dataset_loading(model_args, data_args, training_args, stage, tokenizer_module):
    """调试数据集加载过程"""
    print("=== 开始调试数据集加载 ===")
    print(f"model_args: {model_args}")
    print(f"data_args: {data_args}")
    print(f"training_args: {training_args}")
    print(f"stage: {stage}")
    print(f"tokenizer_module keys: {list(tokenizer_module.keys())}")

    # 检查tokenizer_module中的内容
    for key, value in tokenizer_module.items():
        print(f"tokenizer_module['{key}'] type: {type(value)}")
        if hasattr(value, '__dict__'):
            print(
                f"tokenizer_module['{key}'] attributes: {[attr for attr in dir(value) if not attr.startswith('_')][:10]}")

    # 尝试逐步加载数据集
    try:
        print("=== 尝试获取数据集 ===")
        dataset_result = get_dataset(
            template=getattr(data_args, 'template', 'default'),
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage=stage,
            **tokenizer_module
        )
        print(f"数据集获取成功，返回结果类型: {type(dataset_result)}")
        if isinstance(dataset_result, dict):
            print(f"数据集结果keys: {list(dataset_result.keys())}")
            if "train_dataset" in dataset_result:
                print(f"train_dataset类型: {type(dataset_result['train_dataset'])}")
        return dataset_result
    except Exception as e:
        print(f"获取数据集时出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

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
):
    r"""
    Calculates the ppl on the dataset of the pre-trained models.
    Usage: python cal_ppl.py --model_name_or_path /data4/liutianhui/model_zoo/citygpt-qwen2.5-7b-London-v24.5-SF-v2.3-un-pk-20250213 --dataset citygpt-London-v24.5-SF-v2.3-alpaca_part1 --save_name /data4/liutianhui/LLAMA-Factory-v0.8.3-merge/data/citygpt-London-v24.5-SF-v2.3-ppl-part1.json --dataset_dir /data4/liutianhui/LLAMA-Factory-v0.8.3-merge/data
    The relation between ppl and loss: loss = ln(ppl)
    """
    print("=== 开始参数解析 ===")
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage=stage,
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            train_on_prompt=train_on_prompt,
            preprocessing_num_workers=64,
            output_dir="dummy_dir",
            overwrite_cache=True,
            do_train=True,
        )
    )

    print("=== 参数解析完成 ===")
    print(f"model_args类型: {type(model_args)}")
    print(f"data_args类型: {type(data_args)}")

    tokenizer_module = load_tokenizer(model_args)
    print(f"tokenizer_module类型: {type(tokenizer_module)}")
    tokenizer = tokenizer_module["tokenizer"]
    print(f"tokenizer类型: {type(tokenizer)}")

    print(f"Data args template: {getattr(data_args, 'template', 'Not found')}")

    # 修复：直接修改 data_args 中的 template 为模板对象
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    data_args.template = template_obj  # 关键修改：将模板对象设置到 data_args 中

    # trainset = get_dataset(model_args, data_args, training_args, stage, **tokenizer_module)["train_dataset"]
    trainset = get_dataset(
        template=template_obj,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage=stage,
        **tokenizer_module
    )["train_dataset"]

    # # 在这里添加调试信息
    print("=== 开始加载数据集 ===")
    # try:
    #     trainset = get_dataset(
    #         template=template_obj,
    #         model_args=model_args,
    #         data_args=data_args,
    #         training_args=training_args,
    #         stage=stage,
    #         **tokenizer_module
    #     )
    #     print("=== 数据集加载成功 ===")
    #     print(f"数据集类型: {type(trainset)}")
    #     if isinstance(trainset, dict):
    #         print(f"数据集包含的键: {list(trainset.keys())}")
    #         if "train_dataset" in trainset:
    #             trainset = trainset["train_dataset"]
    #             print(f"训练集类型: {type(trainset)}")
    #             if hasattr(trainset, '__len__'):
    #                 print(f"训练集样本数量: {len(trainset)}")
    #     else:
    #         trainset = trainset
    #         print(f"直接返回的训练集类型: {type(trainset)}")
    #
    # except Exception as e:
    #     print(f"=== 数据集加载失败 ===")
    #     print(f"错误信息: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     raise


    # print("=== 开始加载模型 ===")

    # # 修复：完全绕过 LLaMA-Factory 的模型加载，直接使用 transformers
    # try:
    #     model = AutoModel.from_pretrained(
    #         model_name_or_path,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         trust_remote_code=True,
    #         attn_implementation="eager"  # 关键：使用 eager 模式
    #     )
    #     print("=== 模型加载完成 ===")
    #
    # except Exception as e:
    #     print(f"模型加载失败: {e}")
    #     raise

    # print("=== 开始加载模型 ===")
    # model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    # print("=== 模型加载完成 ===")

    # 修复：直接使用 transformers 加载模型，绕过 LLaMA-Factory 的加载逻辑
    try:
        print("使用 transformers 直接加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,  # 使用与配置相同的类型
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # 使用 eager 模式避免可能的兼容性问题
        )
        print("=== 模型加载完成（使用 transformers） ===")

    except Exception as e:
        print(f"Transformers 直接加载失败: {e}")
        print("回退到 LLaMA-Factory 加载...")
        # 修复：确保 ALL_PARALLEL_STYLES 不为 None
        import transformers
        if not hasattr(transformers.modeling_utils,
                       'ALL_PARALLEL_STYLES') or transformers.modeling_utils.ALL_PARALLEL_STYLES is None:
            transformers.modeling_utils.ALL_PARALLEL_STYLES = []

        model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
        print("=== 模型加载完成（使用 LLaMA-Factory） ===")

    # if stage == "pt":
    #     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # elif stage == "sft":
    #     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    # elif stage == "rm":
    #     data_collator = PairwiseDataCollatorWithPadding(
    #         tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX, train_on_prompt=train_on_prompt
    #     )
    # else:
    #     raise NotImplementedError("Stage does not supported: {}.".format(stage))

    # 修改这部分代码：
    if stage == "pt":
        data_collator = FilteredDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif stage == "sft":
        data_collator = FilteredDataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    else:
        raise NotImplementedError("Stage does not supported: {}.".format(stage))

    print("=== 详细调试数据集 ===")

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

                # 2. 逐个字段分析
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
                        print(f"  前15个元素: {value[:15]}")

                        # 检查None值
                        if hasattr(value, '__iter__'):
                            none_indices = [idx for idx, item in enumerate(value) if item is None]
                            if none_indices:
                                print(f"  ⚠️  发现 {len(none_indices)} 个None值")
                                print(f"  None值位置(前10个): {none_indices[:10]}")
                            else:
                                print(f"  ✅ 没有None值")

                        # 检查数据类型
                        if len(value) > 0:
                            first_element = value[0]
                            print(f"  第一个元素类型: {type(first_element)}")
                            print(f"  第一个元素值: {first_element}")
                    else:
                        print(f"  ⚠️ 不可迭代对象: {value}")

                # 3. 字段间关系检查
                print(f"\n🔗 字段间关系检查:")
                if all(f in sample and sample[f] is not None for f in fields_to_check):
                    lens = {f: len(sample[f]) for f in fields_to_check}
                    print(f"  各字段长度: {lens}")

                    if len(set(lens.values())) != 1:
                        print(f"  ⚠️ 字段长度不一致!")
                    else:
                        print(f"  ✅ 所有字段长度一致")

                # 4. 尝试解码查看文本内容
                print(f"\n📝 文本内容预览:")
                try:
                    if 'input_ids' in sample and sample['input_ids'] is not None:
                        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
                        print(f"  input_ids解码: {input_text[:200]}...")

                    if 'labels' in sample and sample['labels'] is not None:
                        # 过滤掉IGNORE_INDEX
                        valid_labels = [label for label in sample['labels'] if label != IGNORE_INDEX]
                        if valid_labels:
                            label_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
                            print(f"  labels解码: {label_text[:200]}...")
                        else:
                            print(f"  labels: 全部为IGNORE_INDEX")
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
    dataloader = DataLoader(trainset, batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_ppl = 0
    perplexities = []
    batch: Dict[str, "torch.Tensor"]

    print("=== 开始计算困惑度 ===")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(model.device)
            outputs = model(**batch)
            shift_logits: "torch.Tensor" = outputs["logits"][..., :-1, :]
            shift_labels: "torch.Tensor" = batch["labels"][..., 1:]
            loss_mask = shift_labels != IGNORE_INDEX
            flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
            flatten_labels = shift_labels.contiguous().view(-1)
            token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels)
            token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
            sentence_logps = (token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            total_ppl += sentence_logps.exp().sum().item()
            perplexities.extend(sentence_logps.exp().tolist())

    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(perplexities, f, indent=2)

    print("Average perplexity is {:.2f}".format(total_ppl / len(perplexities)))
    print("Perplexities have been saved at {}.".format(save_name))


import traceback
def debug_data_preprocessing():
    """专门调试数据预处理过程"""

    # 使用你实际的参数
    model_name_or_path = "/root/autodl-tmp/Qwen/Qwen2.5-7B-Instruct"
    dataset_name = "beijing"

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage="sft",
            model_name_or_path=model_name_or_path,
            dataset=dataset_name,
            dataset_dir="autodl-tmp/CityGPT-main/LLAMA-Factory/data",
            template="default",
            cutoff_len=4096,
            preprocessing_num_workers=1,  # 先设置为1以便调试
            output_dir="dummy_dir",
            overwrite_cache=True,
            do_train=True,
        )
    )

    tokenizer_module = load_tokenizer(model_args)

    print("=== 详细调试信息 ===")
    print(f"data_args类型: {type(data_args)}")
    print(f"data_args所有属性: {[attr for attr in dir(data_args) if not attr.startswith('_')]}")

    # 检查数据集配置
    if hasattr(data_args, 'dataset'):
        print(f"data_args.dataset: {data_args.dataset}")
        print(f"data_args.dataset类型: {type(data_args.dataset)}")

    try:
        # 尝试获取少量样本进行调试
        dataset_result = get_dataset(
            template="default",
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            stage="sft",
            **tokenizer_module
        )

        print("数据集加载成功!")
        return dataset_result

    except Exception as e:
        print(f"错误详情: {e}")
        print(f"错误类型: {type(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # debug_data_preprocessing()
    fire.Fire(cal_ppl)


