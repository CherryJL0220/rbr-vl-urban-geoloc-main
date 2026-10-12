#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA-Factory 全参数微调启动脚本（Python 版）
用法:
    python run_llama3_full_sft.py
"""

import os
import subprocess
import sys

# ========================= 关键参数（按需修改） =========================
CUDA_VISIBLE_DEVICES = "0,1,2"      # 显卡编号
FORCE_TORCHRUN = 1                  # 1=启用 torchrun，0=禁用
WANDB_DISABLED = True               # True=关闭 wandb，False=开启
ACCELERATE_LOG_LEVEL = ""           # 设为空字符串即不导出；如需调试可设为 "info"
YAML_CONFIG = "/root/autodl-tmp/CityGPT-main/LLAMA-Factory/examples/train_lora/qwen2_5vl_lora_sft.yaml"
# ====================================================================

def setup_env():
    """把原来 bash 里的 export 动作搬到 Python 里"""
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    os.environ["FORCE_TORCHRUN"] = str(FORCE_TORCHRUN)
    os.environ["WANDB_DISABLED"] = str(WANDB_DISABLED).lower()

    if ACCELERATE_LOG_LEVEL:
        os.environ["ACCELERATE_LOG_LEVEL"] = ACCELERATE_LOG_LEVEL

    # 可选：打印确认
    print("[Env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[Env] FORCE_TORCHRUN       =", os.environ.get("FORCE_TORCHRUN"))
    print("[Env] WANDB_DISABLED       =", os.environ.get("WANDB_DISABLED"))
    if ACCELERATE_LOG_LEVEL:
        print("[Env] ACCELERATE_LOG_LEVEL =", os.environ.get("ACCELERATE_LOG_LEVEL"))

def run_training():
    """等价于 bash: llamafactory-cli train <yaml>"""
    cmd = ["llamafactory-cli", "train", YAML_CONFIG]
    print("\n[Run]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("\n[Error] 训练进程异常退出，返回码:", e.returncode)
        sys.exit(e.returncode)

if __name__ == "__main__":
    setup_env()
    run_training()


