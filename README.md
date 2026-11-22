# AvianRaptorNet: Bio-Inspired Vision Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CIFAR100](https://img.shields.io/badge/CIFAR100-84.77%25-brightgreen.svg)]()
[![Params](https://img.shields.io/badge/Params-3.23M-yellow.svg)]()

---

## 1. Overview

**AvianRaptorNet** is a convolutional neural network inspired by the dual-fovea vision system of raptors (birds of prey). It processes fine details through a central foveal path and context through a dilated peripheral path in parallel.

The fast variant reaches **84.77% Top-1 accuracy** on CIFAR-100 with **3.23M parameters**. It was trained with knowledge distillation from a ViT-B/16 teacher (see [§5](#5-knowledge-distillation-cifar-100)). Without distillation it reaches 81.28%.

### Key Features

- **Lightweight:** 3.23M parameters (~12 MB).
- **Dual-Flow Architecture:** Parallel detail (3×3) and context (dilated) streams.
- **BioMish Activation:** JIT-compiled activation function.
- **Hardware Tuned:** Uses TF32 and channels-last memory format on NVIDIA Ampere GPUs.

---

## 2. Main Results (Fast Model)

| Model                   | Params   | Top-1 Acc | Training Strategy                          |
|-------------------------|----------|-----------|--------------------------------------------|
| AvianRaptorNet-Fast     | 3.23M    | **84.77%**| + KD → DKD from ViT-B/16 (this repo)       |
| AvianRaptorNet-Fast     | 3.23M    | 83.76%    | + KD from ViT-B/16 (stage 1 only)          |
| AvianRaptorNet-Fast     | 3.23M    | 81.28%    | Mixup + AutoAugment + Safe Refinement      |
| AvianRaptorNet-Fast     | 3.23M    | 80.42%    | Mixup + AutoAugment                        |
| DenseNet-BC (ref)       | 2.8M     | 80–82%    | AutoAugment                                |
| MobileNetV2 (ref)       | ~3.4M    | ~73–74%   | Standard                                   |
| GhostNet (ref)          | ~5.2M    | ~74–77%   | Standard                                   |

> **Notes:**
> - Distilled numbers are measured on the final epoch's weights, not on a checkpoint picked by test accuracy.
> - The 81.28% result comes from an ultra-low learning-rate refinement stage run after the main training.
> - Reference rows are approximate values reported in the literature, not reproduced here.

---

## 3. Architecture

1. **Retina (Stem):** Initial convolution and normalization.
2. **Raptor Eye (Dual-Flow):**
   - Foveal path → 3×3 convolution for local detail.
   - Peripheral path → dilated 3×3 convolution for context.
   - Fusion via 1×1 convolution and batch normalization.
3. **Body:** Feather Blocks, an inverted-bottleneck residual block with depthwise separable convolution and channel attention.
4. **Head:** Global average pooling followed by a 1×1 convolution, activation, dropout, flatten, and linear classifier.

Class diagrams for all model variants are available in [`uml/`](uml/).

---

## 4. Experimental Models

- **AvianRaptorNet-Medium:** A ResNet-50-scale experimental variant built with `RaptorBottleneck` blocks.

---

## 5. Knowledge Distillation (CIFAR-100)

A 3.23M-parameter AvianRaptorNet-Fast student is distilled from an 86M-parameter ViT-B/16 teacher. The student architecture stays the same, so inference cost does not change.

| Stage | Loss | Epochs | Init | Top-1 | Checkpoint |
|-------|------|--------|------|-------|------------|
| Start | — | — | — | 81.27% | `avian_raptor_best.pth` |
| 1. KD | 0.9 · KL(T=4) + 0.1 · CE | 120 | start ckpt | 83.76% | — |
| 2. DKD | 0.9 · DKD(α=1, β=8, T=4) + 0.1 · CE | 300 | stage 1 | **84.77%** | `avian_raptor_fast_distilled.pth` |
| Teacher | — | — | — | 92.64% | ViT-B/16 (not in repo) |

- **Teacher:** [`edadaltocg/vit_base_patch16_224_in21k_ft_cifar100`](https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100), a ViT-B/16 pretrained on ImageNet-21k and fine-tuned on CIFAR-100. Measured accuracy here is 92.64%, with 32×32 images upsampled to 224×224 (bicubic). The script downloads it into `teachers/` on first run.
- **Consistent teaching:** distillation is online. On every step the teacher and student see exactly the same augmented image (crop, flip, AutoAugment, then Mixup or CutMix), following Beyer et al., *"Knowledge distillation: A good teacher is patient and consistent"* (CVPR 2022).
- **DKD:** Decoupled KD (Zhao et al., CVPR 2022) splits the KD loss into target-class and non-target-class terms. For mixed images it is weighted by the mixing ratio λ across both labels.
- **Optimization:** SGD (Nesterov, momentum 0.9, wd 1e-4 with no decay on BN or bias), cosine LR schedule, AMP, EMA of the weights, batch size 128.
- **Hardware:** one RTX 3070 (8 GB). Stage 1 takes about 2.7 h and stage 2 about 6.5 h.
- **Accuracy over time:** test accuracy drops by 2–4 points early in each stage while the LR is high. It recovers and passes the starting point as the LR decays.

```bash
pip install timm

# Stage 1: vanilla KD from the CIFAR-100 checkpoint
python -m distill.distill_cifar100 --epochs 120

# Stage 2: DKD, continuing from stage 1
python -m distill.distill_cifar100 --loss dkd --epochs 300 --lr 0.005 --warmup-epochs 1 \
    --student-weights avian_raptor_distilled_last.pth --out avian_raptor_dkd
```

Load the distilled weights:

```python
import torch
from core.avian_model import AvianRaptorNet_Fast

model = AvianRaptorNet_Fast(num_classes=100)
model.load_state_dict(torch.load("avian_raptor_fast_distilled.pth", map_location="cpu"))
model.eval()
```

Inputs are 32×32 RGB images normalized with the CIFAR-100 mean `(0.5071, 0.4867, 0.4408)` and std `(0.2675, 0.2565, 0.2761)`.

---

## 6. Teacup Classification (mobrew Project)

A transfer-learning experiment that fine-tunes AvianRaptorNet-Fast on teacup and tea-set images.

### Dataset Pipeline

1. **Scraping:** `experiments/scrape_teacup.py`
2. **Filtering:** `experiments/filter_non_teacup.py`
3. **Fine-tuning:** `experiments/fine_tune_teacup.py`
4. **Clustering:** `experiments/teacup_auto_cluster.py`

### Usage

```bash
python3 experiments/scrape_teacup.py
python3 experiments/filter_non_teacup.py
python3 experiments/fine_tune_teacup.py
python3 experiments/teacup_inference.py <image_path>
```

---

## 7. Installation & Usage

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training & Refinement

```bash
python -m cifar100.cifar100
python cifar100/finetune_cifar100.py --weights=path_to_weight.pth
```

---

## 8. COCO Object-Crop Fine-Tuning

This project includes a transfer-learning pipeline that fine-tunes the CIFAR-100 pretrained **AvianRaptorNet-Fast** on cropped COCO objects for the 80 COCO thing categories.

### How it works

- **Source checkpoint:** `avian_raptor_fast_best.pth` (CIFAR-100, 100 classes)
- **Target task:** classify cropped COCO objects into the 80 COCO categories
- **Pipeline:**
  1. Load the pretrained backbone and replace the classifier head from 100 → 80 classes.
  2. **Primary stage:** SGD with linear warmup + cosine decay, mixup, and AutoAugment for 80 epochs → `avian_coco_primary.pth`
  3. **Refinement stage:** AdamW with an ultra-low learning rate (1e-5) for 25 epochs → `avian_coco_refined.pth`
  4. **Micro stage:** AdamW with a micro learning rate (1e-6) for up to 30 epochs, with early stopping → `avian_coco_micro.pth`

### Hardware / tuning notes

These defaults are tuned for an **RTX 3070 8 GB** card. Larger resolutions or batch sizes run out of memory on this GPU.

| Setting | Value |
|---------|-------|
| Input resolution | 128 × 128 |
| Batch size | 12 |
| `num_workers` | 0 |
| Primary optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=1e-4 |
| Refinement optimizer | AdamW, lr=1e-5, weight_decay=1e-4 |
| Micro optimizer | AdamW, lr=1e-6, weight_decay=1e-6 |
| Primary epochs | 80 |
| Refinement epochs | 25 |
| Micro epochs | 30 (early stopping patience = 10) |

### Usage

```bash
# Full training (requires ./data/coco_subset_20k)
python -m coco.coco

# Micro refinement from the best refined checkpoint
python -m coco.coco_micro_refinement
```

### Outputs

| Stage | Val Top-1 Accuracy | Checkpoint |
|-------|-------------------|------------|
| Primary (EMA) | 60.57% | `avian_coco_primary_ema.pth` |
| Refined (EMA) | 61.79% | `avian_coco_refined_ema.pth` |
| Micro (raw) | 63.63% | `avian_coco_micro.pth` |
| Micro (EMA) | 63.22% | `avian_coco_micro_ema.pth` |

All checkpoints can be loaded with `AvianRaptorNet_Fast(num_classes=80)`.

---

## 9. Citation

```bibtex
@misc{AvianRaptorNet2025,
  author       = {gg582},
  title        = {AvianRaptorNet: Bio-Inspired Lightweight Vision Model with Raptor Dual-Flow Architecture},
  year         = {2025},
  publisher    = {GitHub},
  note         = {84.77% on CIFAR-100 with 3.23M parameters (distilled from ViT-B/16)},
  howpublished = {\url{https://github.com/gg582/avianraptornet}}
}
```

---

## 10. License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
