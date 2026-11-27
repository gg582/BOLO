# BOLO: Birds Only Live Once

**Motion-robust lightweight detection for video and CCTV, built on AvianRaptorNet blocks.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Detector](https://img.shields.io/badge/BOLO--11n-2.88M_params-yellow.svg)]()
[![FLOPs](https://img.shields.io/badge/GFLOPs-8.0%20%40640-orange.svg)]()

---

## 1. Overview

**BOLO** (*Birds Only Live Once*, a parody of YOLO) targets motion-blurred
video frames on embedded devices. It takes the raptor-inspired feature blocks
of [AvianRaptorNet](#appendix-a-avianraptornet-backbone) and adds a
dual-stride motion-correction layer, `MotionStrideAttention`.

| Component | What it is | Section |
|---|---|---|
| **BOLO-11n** | YOLO11n detector with RaptorFoveal / FeatherBlock / MotionStrideAttention | [§2](#2-bolo-11n-detector)–[§4](#4-training--export) |
| **AvianHeadDetector + tracker** | Overhead CCTV head detector on an AvianRaptorNet backbone, with a Kalman multi-object tracker and an interactive demo | [§5](#5-avianheaddetector--cctv-tracking) |
| **Benchmarks** | Latency, FLOPs, detection mAP, blur robustness, limitations | [§6](#6-benchmarks--evaluation-scope) |
| **BOLO-CIFAR100** | AvianRaptorNet-Fast classifier with identity-initialized motion layers | [§7](#7-bolo-cifar100-classifier) |

The AvianRaptorNet classification work that BOLO builds on (CIFAR-100,
distillation, COCO crops, teacup) is kept in the appendices.

---

## 2. BOLO-11n Detector

- **Base:** YOLO11n backbone/neck/head (`bolo/bolo11n.yaml`), 2.88M params, 8.0 GFLOPs at 640.
- **RaptorFoveal** after the stem: parallel foveal (3x3) + peripheral
  (dilated 3x3) streams from AvianRaptorNet.
- **FeatherBlock** in the P3/P4 backbone stages: inverted-bottleneck
  depthwise block with channel attention.
- **MotionStrideAttention** on P5 and P3: captures motion blur / momentum
  at two spatial strides — a large-stride (4x) branch for coarse momentum
  and a small-stride (2x) branch for fine momentum. The two energy maps are
  fused into a sigmoid attention map that gates a depthwise residual
  correction (inverse-estimation / deblur) branch:
  `out = x + attn * corr(x)`.

All custom layers use standard convolution, pooling, interpolation and
element-wise ops, so they export cleanly to ONNX. BOLO does not modify the
Ultralytics installation; its modules are registered at runtime
(`bolo/model.py`). Module-level details are in [`bolo/README.md`](bolo/README.md).

---

## 3. Datasets

| Dataset | Classes | Build | Config |
|---|---|---|---|
| COCO 20k subset | 80 | `python -m bolo.prepare_data [--tiny]` | `bolo/bolo_coco20k.yaml` (`bolo_tiny.yaml`) |
| Neatable desk-tidy | 17 | `python -m bolo.prepare_neatable` | `bolo/bolo_neatable.yaml` |
| Everyday objects | 60 | `python -m bolo.prepare_oid && python -m bolo.prepare_everyday` | `bolo/bolo_everyday.yaml` |

The everyday set merges TACO, an office-supplies set, the COCO 20k subset and
OpenImages (stationery and household classes).

---

## 4. Training & Export

Synthetic motion-blur augmentation (random direction, 5–31 px kernels,
probability 0.3 by default) teaches the network to detect under blur; no
video dataset is required.

```bash
# Smoke test (tiny split, validates the full pipeline)
python -m bolo.train_bolo --data bolo_tiny.yaml --epochs 2 --name bolo_smoke

# Everyday objects (~5 h on an RTX 3070 at 70 epochs)
python -m bolo.train_bolo --data bolo_everyday.yaml --epochs 70 --batch 16 --name bolo_everyday

# Full COCO 20k
python -m bolo.train_bolo --data bolo_coco20k.yaml --epochs 100 --batch 16 --name bolo_full
```

Export (fused `.pth` + embedded ONNX):

```bash
python -m bolo.export_bolo --weights runs/bolo/bolo_everyday/weights/best.pt --prefix bolo_everyday
```

| File | Description |
|------|-------------|
| `<prefix>.pth` | fused model state dict |
| `<prefix>_fp32.onnx` | slimmed FP32 graph (reference) |
| `<prefix>_fp16.onnx` | embedded-optimized FP16 model (fixed shape, onnxslim, parity-checked with onnxruntime) |

The prefix defaults to `bolo11n_avian`. FP16 ONNX uses a fixed input size
(640 by default); export at the target device's resolution, e.g. `--imgsz 320`.

---

## 5. AvianHeadDetector & CCTV Tracking

A real-time person-head detector for overhead / booth CCTV
(`bolo/head_detector.py`):

- **Backbone:** AvianRaptorNet-Fast, initialized from `avian_coco_micro.pth`
  ([Appendix C](#appendix-c-coco-object-crop-fine-tuning)); outputs P3/P4/P5 (strides 8/16/32).
- **Neck:** lightweight PANet with `MotionStrideAttention`.
- **Head:** anchor-free decoupled head (classification + box regression); ONNX / TensorRT exportable.
- **Tracker:** constant-velocity Kalman box tracker with Hungarian matching
  (`bolo/tracker.py`) for counting and false-positive suppression.

```bash
# SCUT-HEAD / WEPDTOF -> single-class YOLO format
python scripts/prepare_head_dataset.py --scut-dir <SCUT-HEAD> --wepdtof-dir <WEPDTOF>

# Two-stage training (frozen backbone, then fine-tune), AMP + EMA, RTX 3070 defaults
python -m bolo.train_head_detector --data data/head_dataset/head.yaml --pretrained avian_coco_micro.pth

# ONNX export, video inference
python -m bolo.export_head_detector --weights runs/head_detector/<run>/best.pth
python -m bolo.infer_video --weights runs/head_detector/<run>/best.pth --source video.mp4
```

Accuracy on SCUT-HEAD val: AP50 0.788 / AP50-95 0.323
(see [§6](#avianheaddetector-scut-head-val)).

**Interactive demo** (`demo.py`): raw detections vs. Kalman-filtered tracks
with a live HUD (count, latency, FPS). With no `--source` it plays
`data/cctv_samples/cctv_default_demo.mp4`, a clip of the open Oxford Town
Centre CCTV dataset. Keys: `Space` pause, `T` raw overlay, `C` confidence,
`S` snapshot, `Q` quit.

```bash
python demo.py                       # default CCTV clip
python demo.py --source 0            # webcam
python demo.py --source video.mp4 --save-video out.mp4
```

---

## 6. Benchmarks & Evaluation Scope

Detection, robustness and deployment measurements for BOLO, and what has
**not** been measured yet. Classifier latency for the AvianRaptorNet backbone
is in [Appendix E](#appendix-e-avianraptornet-classifier-latency).

Hardware: NVIDIA RTX 3070 (8 GB), AMD Ryzen 5 5600X (6C/12T), PyTorch 2.9.1,
onnxruntime 1.32 (CPU). Raw outputs are written to `runs/` by the scripts below.

### Complexity & latency

```bash
python scripts/benchmark_latency.py   # BOLO ONNX on CPU (+ classifiers, Appendix E)
python -m bolo.benchmark_cctv         # BOLO PyTorch latency on CCTV footage
```

Median latency at batch 1. GFLOPs = 2 x MACs at the listed input size.

| Model | Input | Params | GFLOPs | GPU FP32 | GPU FP16 | CPU FP32 | GPU FP16 throughput |
|---|---|---|---|---|---|---|---|
| BOLO-11n (everyday, 60 cls) | 640 x 640 | 2.88M | 7.99 | 6.51 ms | 7.03 ms | 131.86 ms | 574 img/s (b16) |

| BOLO ONNX export | Size | onnxruntime CPU (640 x 640) |
|---|---|---|
| `bolo_everyday_fp32.onnx` | 11.2 MB | 42.91 ms |
| `bolo_everyday_fp16.onnx` | 5.7 MB | 45.66 ms |

End-to-end BOLO on real CCTV frames (Oxford Town Centre clip
`data/cctv_samples/cctv_default_demo.mp4`, 1280 x 720, pre-process + forward
+ NMS, GPU FP16): **7.86 ms/frame (~127 FPS)**.

Notes: at batch 1 these small models are kernel-launch bound on the GPU, so
FP16 gives no latency gain there. Batched, FP16 pays off: BOLO-11n at batch 16
runs at 574 img/s in FP16 vs 324 img/s in FP32 (1.77x). The FP16 ONNX graph is
half the size but no faster on an x86 CPU without native FP16 arithmetic (a
second run gave 45.6 ms FP32 vs 44.4 ms FP16, i.e. equal within run-to-run
noise); its latency benefit needs an FP16-capable NPU/GPU target.

### BOLO detection mAP

| Eval set | Images | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| everyday val (all sources, logged at end of training) | — | 0.364 | 0.282 | 0.506 | 0.365 |
| everyday val, COCO part only (re-evaluated) | 800 | 0.252 | 0.171 | 0.421 | 0.242 |

The full everyday val set needs the TACO / Office / OpenImages raw data; the
COCO part is rebuilt deterministically from `data/coco_subset_20k` by
`bolo/eval_blur_coco.py` (same seed and selection as `prepare_everyday.py`,
disjoint from training images).

### BOLO motion-blur robustness

```bash
python -m bolo.eval_blur_coco
```

Every image of the 800-image COCO everyday val set (3,841 boxes, 31 classes)
is blurred with a linear motion kernel of the given length (seeded random
direction per image); evaluated at 640.

| Blur kernel | mAP50 | mAP50-95 | Retained mAP50-95 | P | R |
|---|---|---|---|---|---|
| none | 0.252 | 0.171 | 100% | 0.421 | 0.242 |
| 5 px | 0.244 | 0.165 | 96% | 0.382 | 0.245 |
| 11 px | 0.219 | 0.146 | 85% | 0.379 | 0.224 |
| 21 px | 0.185 | 0.118 | 69% | 0.394 | 0.190 |
| 31 px | 0.136 | 0.085 | 49% | 0.358 | 0.151 |

### MotionStrideAttention test-time bypass (indirect ablation)

```bash
python -m bolo.eval_blur_coco --no-msa
```

Without retraining, both `MotionStrideAttention` layers of the same checkpoint
are replaced by identity (`out = x`) and the blur sweep is repeated. If the
layer specifically corrected blur, removing it should hurt more as blur grows.

| Blur kernel | mAP50-95 (MSA on) | mAP50-95 (MSA bypassed) | Drop from bypass |
|---|---|---|---|
| none | 0.171 | 0.127 | -26% |
| 5 px | 0.165 | 0.124 | -25% |
| 11 px | 0.146 | 0.108 | -26% |
| 21 px | 0.118 | 0.092 | -22% |
| 31 px | 0.085 | 0.070 | -18% |

**Result: no blur-specific effect.** The network depends on the MSA branch by
a roughly constant ~25% on clean and mildly blurred images, and *less* under
heavy blur, so the layer currently acts as a general feature-refinement block,
not as a blur corrector. This is a post-hoc test (the network was trained with
the layer present), so it cannot replace a retrained YOLO11n / no-MSA
baseline, but it gives no evidence for the motion-correction claim so far.

### AvianHeadDetector (SCUT-HEAD val)

```bash
python -m bolo.eval_head_detector
```

SCUT-HEAD val split of `data/head_dataset` (646 images, Part A 289 / Part B
357, 15,466 head boxes), 640 x 640, NMS IoU 0.6, AP over IoU 0.50:0.95 with
the ultralytics AP routine; P/R at the max-F1 confidence.

| Checkpoint | Epoch | AP50 | AP50-95 | P | R | GPU FP32 forward + NMS |
|---|---|---|---|---|---|---|
| `avian_head_scut_real` | 100 | 0.778 | 0.305 | 0.775 | 0.766 | 16.2 ms |
| `avian_head_finetune_night` | 22 | **0.788** | **0.323** | 0.783 | 0.774 | 16.2 ms |

`best.pth` was selected by validation loss on this same split, so these
numbers are slightly optimistic. The low AP50-95 relative to AP50 reflects
the very small SCUT heads (median ~22 px, 77% under 32 px in original
resolution), where box localization at strict IoU is hard.

### Limitations / not yet measured

- **Motion-correction claim not supported yet.** No plain YOLO11n / no-MSA
  model has been retrained with the same recipe, and the test-time bypass
  above shows no blur-specific benefit from `MotionStrideAttention`.
- **CCTV domain gap.** The everyday BOLO checkpoint is not a CCTV model
  (the CCTV model is AvianHeadDetector, §5). On the Oxford Town Centre
  footage it detects almost nothing (max confidence 0.10 at 640, 0.32 at
  1280, mostly false positives): wide-angle street views with small
  bags/backpacks are far from its training data. `bolo/benchmark_cctv.py`
  also reports a label-free clean-vs-blurred consistency check, but it is not
  meaningful for this checkpoint on this footage.
- **AvianHeadDetector on real CCTV is unlabeled.** Head AP is reported on
  SCUT-HEAD only (classroom and web images); WEPDTOF (fisheye) and labeled CCTV
  footage (e.g. Town Centre head annotations) have not been evaluated.
- **No high-resolution classification benchmark** for the AvianRaptorNet
  backbone (e.g. ImageNet-1k); no ImageNet data is available here.
- **No on-device measurement.** Latency was measured on a desktop GPU/CPU
  only; no embedded board (Jetson, mobile NPU, etc.) and no TensorRT/INT8.

---

## 7. BOLO-CIFAR100 Classifier

`BoloCIFAR100` reforms the AvianRaptorNet-Fast model ([Appendix A](#appendix-a-avianraptornet-backbone)) for
native 32 x 32 CIFAR-100 images. It retains the checkpoint-compatible
backbone/classifier and inserts identity-initialized dual-stride motion layers
at its 256- and 512-channel feature stages.

```bash
# CIFAR-100 downloads to data/ automatically on the first run.
python -m bolo.train_cifar100 --data data --epochs 40 --batch-size 256 \
  --init-weights avian_raptor_fast_best.pth
python -m bolo.export_cifar100 --weights bolo_cifar100_best.pth
```

This creates `bolo_cifar100.pth`, `bolo_cifar100_fp32.onnx`, and
`bolo_cifar100_fp16.onnx`. The ONNX input is `[batch, 3, 32, 32]`.

---

## 8. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 9. Citation

```bibtex
@misc{BOLO2026,
  author       = {gg582},
  title        = {BOLO: Birds Only Live Once -- Motion-Robust Lightweight Detection},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/gg582/BOLO}}
}

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

---

# Appendices: AvianRaptorNet

## Appendix A. AvianRaptorNet Backbone

**AvianRaptorNet** is a convolutional neural network inspired by the dual-fovea vision system of raptors (birds of prey). It processes fine details through a central foveal path and context through a dilated peripheral path in parallel.

The fast variant reaches **84.77% Top-1 accuracy** on CIFAR-100 with **3.23M parameters**. It was trained with knowledge distillation from a ViT-B/16 teacher (see [Appendix B](#appendix-b-knowledge-distillation-cifar-100)). Without distillation it reaches 81.28%.

### Key Features

- **Lightweight:** 3.23M parameters (~12 MB).
- **Dual-Flow Architecture:** Parallel detail (3×3) and context (dilated) streams.
- **BioMish Activation:** JIT-compiled activation function.
- **Hardware Tuned:** Uses TF32 and channels-last memory format on NVIDIA Ampere GPUs.

### Main Results (Fast Model)

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

### Architecture

1. **Retina (Stem):** Initial convolution and normalization.
2. **Raptor Eye (Dual-Flow):**
   - Foveal path → 3×3 convolution for local detail.
   - Peripheral path → dilated 3×3 convolution for context.
   - Fusion via 1×1 convolution and batch normalization.
3. **Body:** Feather Blocks, an inverted-bottleneck residual block with depthwise separable convolution and channel attention.
4. **Head:** Global average pooling followed by a 1×1 convolution, activation, dropout, flatten, and linear classifier.

Class diagrams for all model variants are available in [`uml/`](uml/).

### Experimental Models

- **AvianRaptorNet-Medium:** A ResNet-50-scale experimental variant built with `RaptorBottleneck` blocks.

### Training & Refinement (CIFAR-100)

```bash
python -m cifar100.cifar100
python cifar100/finetune_cifar100.py --weights=path_to_weight.pth
```

---

## Appendix B. Knowledge Distillation (CIFAR-100)

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

## Appendix C. COCO Object-Crop Fine-Tuning

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

## Appendix D. Teacup Classification (mobrew Project)

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

## Appendix E. AvianRaptorNet Classifier Latency

Measured with `python scripts/benchmark_latency.py` on the same hardware as
[§6](#6-benchmarks--evaluation-scope) (random weights; timing depends only on the architecture).

| Model | Input | Params | GFLOPs | GPU FP32 | GPU FP16 | CPU FP32 | GPU FP16 throughput |
|---|---|---|---|---|---|---|---|
| AvianRaptorNet-Fast (CIFAR-100) | 32 x 32 | 3.23M | 0.32 | 3.17 ms | 3.11 ms | 8.98 ms | 13,041 img/s (b256) |
| AvianRaptorNet-Fast (COCO crops) | 128 x 128 | 3.21M | 5.19 | 2.80 ms | 3.20 ms | 44.97 ms | 890 img/s (b32) |
