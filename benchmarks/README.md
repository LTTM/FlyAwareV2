# 📊 FlyAwareV2 Benchmarks

Comprehensive evaluation suite for the FlyAwareV2 dataset, featuring model pretraining, evaluation metrics, and domain adaptation techniques for aerial scene understanding.

## 📝 Description

This benchmarks suite provides all the necessary tools and scripts to evaluate computer vision models on the FlyAwareV2 dataset. It supports various model architectures, training configurations, and evaluation protocols for semantic segmentation tasks in aerial imagery.

> [!NOTE]
> The benchmark suite supports both synthetic and real data evaluation, with specialized tools for domain adaptation and cross-domain generalization.

---

## 🏁 Quick Start

### 📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### 1️⃣ Synthetic Pretraining

Train models on FlyAwareV2 synthetic data with various configurations:

#### 🏙️ RGB Town10 Only

```bash
python synthetic_pretrain.py \
    --root_path "<path_to_dataset>" \
    --logdir "logs/mnet_allh_t10_allw_rgb" \
    --batch_size 16 \
    --dloader_workers 6 \
    --town Town10_Opt_120
```

#### 📏 RGB 20m Height Only

```bash
python synthetic_pretrain.py \
    --root_path "<path_to_dataset>" \
    --logdir "logs/mnet_h20_allt_allw_rgb" \
    --batch_size 16 \
    --dloader_workers 6 \
    --height height20m
```

#### 🌙 RGB Night Only

```bash
python synthetic_pretrain.py \
    --root_path "<path_to_dataset>" \
    --logdir "logs/mnet_allh_allt_night_rgb" \
    --batch_size 16 \
    --dloader_workers 6 \
    --weather night
```

#### 🔗 Early-Fusion MultiModal

```bash
python synthetic_pretrain.py \
    --root_path "<path_to_dataset>" \
    --logdir "logs/mnet_allh_allt_allw_mmearly" \
    --batch_size 16 \
    --dloader_workers 6 \
    --model mmearly \
    --modality all
```

### 2️⃣ Model Architectures

Choose from the following model configurations:

| Model | Architecture | Input Channels | Description |
|-------|-------------|----------------|-------------|
| `mobilenet` | MobileNetV3 + DeepLabV3 | RGB: 3ch, Depth: 1ch | Lightweight backbone |
| `resnet50` | ResNet50 + DeepLabV3 | RGB: 3ch, Depth: 1ch | Standard backbone |
| `mmearly` | MultiModal MobileNetV3 + DeepLabV3 | 4ch (RGB+Depth) | Early fusion |
| `mmlate` | 2x MobileNetV3 + DeepLabV3 | Dual encoder | Late fusion |

### 3️⃣ Model Evaluation

After training, the checkpoint will be saved as `<logdir>/latest.pth`. Evaluate on different data splits:

#### ☀️ Daytime Real Data Evaluation

```bash
python evaluate.py \
    --root_path "<path_to_dataset>" \
    --evaldir "evals/day" \
    --class_set coarse \
    --pretrained_ckpt "<logdir>/latest.pth" \
    --weather day \
    --variant real \
    --resize 3840 \
    --override_logs
```

#### 🎯 Fine-tuned Model Evaluation

```bash
python evaluate.py \
    --root_path "<path_to_dataset>" \
    --evaldir "evals/day" \
    --class_set coarse \
    --pretrained_ckpt "<logdir_of_uda>/latest.pth" \
    --finetuned \
    --weather day \
    --variant real \
    --resize 3840 \
    --override_logs
```

### 4️⃣ Domain Adaptation

Fine-tune pretrained models on real data using Unsupervised Domain Adaptation:

```bash
python UDA_finetune.py \
    --root_path "<path_to_dataset>" \
    --logdir "<path_to_save_dir>" \
    --class_set coarse \
    --pretrained_ckpt "<path_to_pretrained_ckpt>"
```

> [!TIP]
> Use domain adaptation to improve model performance when transferring from synthetic to real data.

---

## 📁 Available Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `synthetic_pretrain.py` | Pre-training on synthetic data | Multiple configurations, model architectures |
| `evaluate.py` | Model evaluation | Comprehensive metrics, flexible evaluation |
| `UDA_finetune.py` | Domain adaptation | Unsupervised fine-tuning on real data |
| `flops.py` | Model complexity analysis | FLOPS and parameter counting |

---

## 🙏 Credits

This benchmark suite was developed by:

- **Francesco Barbato** - Primary developer
- **Matteo Caligiuri** - Co-developer

Dipartimento di Ingegneria dell'Informazione (DEI) - UniPD

---

> [!IMPORTANT]
> Make sure to adjust batch sizes and worker numbers according to your hardware capabilities. For optimal performance, use GPU acceleration when available.
