# Reasoning Beyond Retrieval: A Vision-Language Framework for Fine-Grained Intra-Urban Geolocalization

Official implementation of the paper **"Reasoning Beyond Retrieval: A Vision-Language Framework for Fine-Grained Intra-Urban Geolocalization"**.

This repository provides the code, data splits, scene description generation pipeline, training scripts, and evaluation tools for database-free fine-grained urban geolocalization. The framework is built on **LLaVA-1.5-7B** and introduces three core mechanisms: **MSAW**, **TJO**, and **VFM**.

---

## 1. Overview

Traditional retrieval-based geolocalization relies on a dense and up-to-date external geo-reference database. This work reformulates intra-urban localization as an **end-to-end coordinate reasoning** problem using the internalized geospatial knowledge of a large vision-language model. At inference time, the model takes only a street-view image as input and directly outputs geographic coordinates **without any external database or feature index**.

**Core Contributions:**
- **MSAW (Multi-modal Sample Adaptive Weighting):** dynamically evaluates the geo-learning fitness of each training sample based on loss variation before and after warm-up, and assigns differentiated weights to samples.
- **TJO (Task-level Joint Optimization):** combines image-to-location and image-to-description contrastive losses to deeply align visual representations with geographic coordinates and scene semantics.
- **VFM (Visual Focusing Mechanism):** estimates geographic saliency scores using shallow cross-modal attention and softly suppresses low-information visual tokens during inference.

---

## 2. Datasets

We conduct experiments on four standard benchmark datasets and two augmented test sets.

### 2.1 Standard Benchmarks

| Dataset | Description | Usage | Source |
|---|---|---|---|
| **Pitts-250k** | Large-scale street-view dataset from Pittsburgh | Training & testing | http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/ |
| **Pitts-IAL** | Pittsburgh dataset with scene description annotations | Scene description fine-tuning, training & testing | https://github.com/xsx1001/AddressCLIP |
| **Tokyo247** | Street-view dataset from Tokyo | Training & testing | https://www.di.ens.fr/willow/research/netvlad/ |
| **SF-XL-Base** | San Francisco XL dataset | Testing | https://github.com/xsx1001/AddressCLIP |

### 2.2 Augmented Test Sets

To evaluate environmental robustness, we construct two types of augmented test sets based on **Pitts-250k** and **Tokyo247**:
- **Seasonal variation set:** spring, summer, autumn, winter
- **Weather variation set:** rain, snow, fog

These augmented sets simulate complex real-world imaging conditions and are used for robustness evaluation (CV and CRI metrics).

### 2.3 Data Splits

This repository includes:
- Training/validation/test splits for all datasets
- Scene description texts for training images
- Sparse-reference database splits for retrieval-based comparison
- Augmented test set annotations

All data and splits are released with persistent DOI identifiers. Please cite the original dataset papers if you use them.

### 2.4 Data Download

Download the images in the dataset following the instructions in Section 2.1. The training data generated in this paper is available at the link below:

https://pan.baidu.com/s/1tpMYTboeGUrcJY-3pqlZzg?pwd=r2b4 Access code: r2b4 

---

## 3. Training Process

The training pipeline consists of two stages: **scene description generation** and **geolocalization model training**.

### 3.1 Stage 1: Scene Description Generation

Scene descriptions serve as auxiliary geographic semantic anchors for the image-to-description contrastive loss.

**Steps:**
1. Fine-tune **LLaVA-1.5-7B** with LoRA on the Pitts-IAL scene description annotations.
2. Use the fine-tuned model to generate scene descriptions for other training images.
3. Refine the generated descriptions with **GPT-4.5** based on image content.

The refined descriptions are used as supervision signals for the image-to-description contrastive learning in Stage 2.

### 3.2 Stage 2: Geolocalization Model Training

**Base model:** LLaVA-1.5-7B  
**Fine-tuning method:** LoRA (rank = 8)

#### 3.2.1 Warm-up and MSAW

1. Perform one epoch of LoRA warm-up fine-tuning on the base model using the entire training set to obtain the warmed-up model.
2. For each sample, compute the loss on the base model and the warmed-up model.
3. Calculate the geo-learning fitness:

   γᵢ = Lᵢ⁽⁰⁾ − Lᵢ⁽¹⁾

4. Compute the adaptive weight:

   wᵢ = γᵢ / √( (1/N) Σⱼ (Lⱼ⁽⁰⁾)² ) + ε

5. Apply clipping: `max(0, wᵢ)`
6. Use wᵢ in the weighted cross-entropy loss:

   L_AWL = −(1/N) Σᵢ [ wᵢ · Σₜ log P_M(y_{i,t} | I_i, y_{i,t<t}) ]

#### 3.2.2 Task-level Joint Optimization (TJO)

Construct a unified multi-modal embedding space with a visual encoder and a shared text encoder for both coordinate labels and scene descriptions.

**Image-to-location contrastive loss:**

L_loc = −(1/2N) Σᵢ [ log exp(sᵢᵢˡ) / Σⱼ exp(sᵢⱼˡ) + log exp(sᵢᵢˡ) / Σⱼ exp(sⱼᵢˡ) ]

**Image-to-description contrastive loss:**

L_cap = −(1/2N) Σᵢ [ log exp(sᵢᵢᶜ) / Σⱼ exp(sᵢⱼᶜ) + log exp(sᵢᵢᶜ) / Σⱼ exp(sⱼᵢᶜ) ]

**Total loss:**

L_total = L_AWL + λ_loc · L_loc + λ_cap · L_cap

**Loss weights:** λ_loc = 0.8, λ_cap = 0.2

---

## 4. Inference

At inference time, the model takes only a street-view image as input and directly predicts geographic coordinates **without any external reference database**.

### 4.1 Visual Focusing Mechanism (VFM)

**Geographic saliency score:**

sᵢ = (1/H) Σₕ Σⱼ Aₕ⁽ᴷ⁾[j, i]

**Focusing threshold:**

τ = s_(⌊ρ · N⌋)

**Soft suppression:**

αᵢ = 1 if sᵢ ≥ τ; otherwise αᵢ = γ  
ν̃ᵢ = αᵢ · νᵢ

**Inference configuration:**
- Attention layer index K = 8
- Retention ratio ρ = 0.6
- Suppression coefficient γ = 0.1
- No external reference database

---

## 5. Evaluation Metrics

- **Localization Accuracy:** proportion of test images whose Haversine distance error is below threshold τ.
- **Coefficient of Variation (CV):** measures the fluctuation of median errors across different scenarios.
- **Comprehensive Robustness Index (CRI):** CRI = mean median error × (1 + CV). Lower values indicate better performance.

---

## 6. Repository Structure

```
rbr-vl-urban-geoloc/
├── data/                     # Dataset splits and annotations
├── save_checkpoint/          # Model checkpoint save path
├── scripts/                  # Training-related files
├── models/                   # Models
├── transformers/             # Inference-related files
└── README.md
```

---

## 7. Quick Start

### 7.1 Sample Adaptive-weight Computation

```bash
python /root/autodl-tmp/CityGPT-main/LLAMA-Factory/scripts/cal_mutil_ppl.py \
  --model_name_or_path $MODEL_PATH \
  --dataset $DATASET \
  --save_name $SAVE_FILE \
  --dataset_dir $DATASET_DIR \
  --template llava \
  --multimodal True \
```

### 7.2 MSAW + TJO Joint Training

```bash
llamafactory-cli train scripts/configs/llava1_5_lora_sft.yaml"
```

### 7.3 VFM Inference

```bash
python transformers/demo-hf.py
```

---

## 8. Data and Code Availability

All experimental data, derived datasets and splits, generated scene descriptions, source code, and model weights used in this study are publicly available through this repository.


## 9. Contact

**Corresponding author:** Xiaofei Hu  
**Email:** huxiaofeicn@163.com  
**Affiliation:** Institute of Geospatial Information, PLA Cyberspace Information Engineering University, Zhengzhou, People's Republic of China

---

## 10. Acknowledgements

This work was supported by the Joint Fund of Collaborative Innovation Center of Geo-Information Technology for Smart Central Plains, Henan Province and Key Laboratory of Spatiotemporal Perception and Intelligent Processing, Ministry of Natural Resources under Grant 232109.

---

## 11. License

This project is released under the [MIT License](LICENSE). Please also comply with the licenses of the original datasets (Pitts-250k, Pitts-IAL, Tokyo247, SF-XL-Base).
