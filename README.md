# COL780 Assignment 3 — Land Use Classification from Satellite Imagery

## Environment Setup

```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda install pytorch torchvision -c pytorch
conda install -c conda-forge opencv timm scikit-learn pandas
```

---

## Dataset Structure

```
A3_Dataset/
├── train.csv
├── validation.csv
├── test.csv
└── <image files>
```

---

## Subtask 1 — ResNet-18 Baseline

**Training (CrossEntropy):**
```bash
python train_resnet.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --model_path best_resnet_model.pth
```

**Training (Focal Loss ablation):**
```bash
python train_resnet.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --use_focal --focal_alpha 1.0 --focal_gamma 2.0 \
  --model_path best_resnet_focal.pth
```

**Testing:**
```bash
python test_resnet.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --model_path best_resnet_model.pth
```

---

## Subtask 2 — ResNet-18 + Squeeze-and-Excitation (SnE)

**Training (CrossEntropy):**
```bash
python train_sne.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --model_path best_sne_model.pth
```

**Training (Focal Loss ablation):**
```bash
python train_sne.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --use_focal --focal_alpha 0.5 --focal_gamma 1.0 \
  --model_path best_sne_focal.pth
```

**Testing:**
```bash
python test_sne.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --model_path best_sne_model.pth
```

---

## Subtask 3 — DeiT-3 Small (Vision Transformer)

**Training (CrossEntropy):**
```bash
python train_vit.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --run_name deit3_baseline_ce
```

**Training (Focal Loss ablation):**
```bash
python train_vit.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --use_focal --focal_alpha 1.0 --focal_gamma 2.0 \
  --run_name deit3_focal_a1.0_g2.0
```

**Testing:**
```bash
python test_vit.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --model_path best_deit3_baseline_ce.pth
```

---

## Subtask 4 — DeiT-3 with Dynamic Tanh / No LayerNorm (Task 2.2)

**Training (DyT, CrossEntropy):**
```bash
python train_nonormdeit.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --run_name deit3_dyt
```

**Training (standard LayerNorm, for comparison):**
```bash
python train_nonormdeit.py \
  --img_dir A3_Dataset \
  --train_csv A3_Dataset/train.csv \
  --val_csv A3_Dataset/validation.csv \
  --test_csv A3_Dataset/test.csv \
  --epochs 10 --lr 1e-4 --batch_size 32 \
  --no_dyt --run_name deit3_layernorm
```

**Testing:**
```bash
python test_nonormdeit.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --model_path best_deit3_dyt.pth
```

---

## Bonus 3.1 — Grad-CAM Visualisations (ResNet-18 and ResNet-18+SE)

**ResNet-18 Grad-CAM:**
```bash
python resnet_bonus.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --label_map A3_Dataset/label_map.json \
  --checkpoint best_resnet_model.pth \
  --output_dir bonus_outputs/resnet_gradcam
```

**ResNet-18 + SE Grad-CAM:**
```bash
python sne_bonus.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --label_map A3_Dataset/label_map.json \
  --checkpoint best_sne_model.pth \
  --output_dir bonus_outputs/sne_gradcam
```

---

## Bonus 3.2 — Attention Maps (DeiT-3)

```bash
python deit_bonus.py \
  --img_dir A3_Dataset \
  --test_csv A3_Dataset/test.csv \
  --label_map A3_Dataset/label_map.json \
  --checkpoint best_deit3_baseline_ce.pth \
  --output_dir bonus_outputs/deit_attention
```

Output images are saved one per class to the specified `--output_dir`. Each image shows the original, heatmap, and overlay side by side.

---

## Model Weights

All weights exceed the 25 MB ZIP limit and are hosted externally.

| Model | File | Link |
|---|---|---|
| ResNet-18 (best) | best_resnet_model.pth | _add link_ |
| ResNet-18 + SE (best) | best_sne_model.pth | _add link_ |
| DeiT-3 Baseline CE | best_deit3_baseline_ce.pth | _add link_ |
| DeiT-3 Focal a1.0 g2.0 | best_deit3_focal_a1.0_g2.0.pth | _add link_ |

---

## File Overview

| File | Description |
|---|---|
| `load_data.py` | Dataset class (`CropData`) shared by all models |
| `Resnet.py` | `ResNetClassifier` class |
| `SnE.py` | `ResNetSEClassifier` class + `SqueezeExcitationBlock` |
| `ViT.py` | `DieTClassifier` class (DeiT-3 Small) |
| `train_resnet.py` | Training script for Subtask 1 |
| `test_resnet.py` | Testing script for Subtask 1 |
| `train_sne.py` | Training script for Subtask 2 |
| `test_sne.py` | Testing script for Subtask 2 |
| `train_vit.py` | Training script for Subtask 3 |
| `test_vit.py` | Testing script for Subtask 3 |
| `nonormdeit.py` | `DieTClassifier` class with DyT replacing LayerNorm |
| `train_nonormdeit.py` | Training script for Subtask 4 |
| `test_nonormdeit.py` | Testing script for Subtask 4 |
| `resnet_bonus.py` | Grad-CAM visualisation for ResNet-18 (Bonus 3.1) |
| `sne_bonus.py` | Grad-CAM visualisation for ResNet-18+SE (Bonus 3.1) |
| `deit_bonus.py` | Attention map visualisation for DeiT-3 (Bonus 3.2) |
