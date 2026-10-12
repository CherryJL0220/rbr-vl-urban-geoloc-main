#!/usr/bin/env python3
# run_ppl.py
import os
import sys

# 1. 指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

# 2. 把脚本所在目录加入 Python 路径，确保能 import
sys.path.insert(0, "/root/autodl-tmp/CityGPT-main/LLAMA-Factory/scripts")

# 3. 导入函数
from cal_mutil_ppl import cal_ppl

# 4. 直接传参调用（参数名与 fire 完全一致）
cal_ppl(
    model_name_or_path="/root/autodl-tmp/llava",
    dataset="ppl_5w",
    save_name="/root/autodl-tmp/CityGPT-main/LLAMA-Factory/data/test_ppl.json",
    dataset_dir="/root/autodl-tmp/CityGPT-main/LLAMA-Factory/data",
    template="llava",
    multimodal=True,
)