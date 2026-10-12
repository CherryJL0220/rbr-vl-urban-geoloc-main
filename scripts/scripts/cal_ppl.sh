export CUDA_VISIBLE_DEVICES=0

# LLaVA1.5 模型路径
MODEL_PATH="/root/autodl-tmp/llava"
# 多模态图文数据集
DATASET="pitts"
# PPL结果保存路径
SAVE_FILE="/root/autodl-tmp/CityGPT-main/LLAMA-Factory/data/test_ppl.json"
# 数据集目录
DATASET_DIR="/root/autodl-tmp/CityGPT-main/LLAMA-Factory/data/"

# 运行多模态PPL计算（不使用LLaVA库）
python /root/autodl-tmp/CityGPT-main/LLAMA-Factory/scripts/cal_mutil_ppl.py \
  --model_name_or_path $MODEL_PATH \
  --dataset $DATASET \
  --save_name $SAVE_FILE \
  --dataset_dir $DATASET_DIR \
  --template llava \
  --multimodal True \

#  --batch_size 1  # 开始时使用小批量