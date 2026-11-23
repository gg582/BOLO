"""Training Pipeline for AvianHeadDetector.

Optimized for RTX 3070 (8GB VRAM) ceiling/booth CCTV head detection:
- Two-stage training: Phase 1 (Backbone Freeze) -> Phase 2 (Unfreeze & Fine-tune)
- Mixed Precision (AMP FP16) + Channels-Last memory format
- Exponential Moving Average (EMA) of weights
- Cosine Annealing Learning Rate Scheduler with Warm-up
- RAM caching option for maximum throughput
"""

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from bolo.dataset import HeadDetectionDataset, collate_fn
from bolo.head_detector import AvianHeadDetector
from bolo.loss import HeadDetectionLoss

ROOT = Path(__file__).resolve().parent.parent


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1.0 - self.decay) * msd[k].detach()


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load dataset config
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_cfg.get("path", ROOT / "data" / "head_dataset"))
    train_img_dir = data_root / data_cfg["train"]
    train_lbl_dir = data_root / "labels" / "train"
    val_img_dir = data_root / data_cfg["val"]
    val_lbl_dir = data_root / "labels" / "val"

    train_dataset = HeadDetectionDataset(
        img_dir=train_img_dir,
        lbl_dir=train_lbl_dir,
        imgsz=args.imgsz,
        augment=True,
        cache_ram=args.cache_ram,
    )
    val_dataset = HeadDetectionDataset(
        img_dir=val_img_dir,
        lbl_dir=val_lbl_dir,
        imgsz=args.imgsz,
        augment=False,
        cache_ram=args.cache_ram,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.workers // 2),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"[Dataset] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Build model
    pretrained_weights = args.pretrained if Path(args.pretrained).exists() else None
    model = AvianHeadDetector(pretrained_backbone=pretrained_weights, num_classes=1)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    if args.resume_weights and Path(args.resume_weights).exists():
        print(f"[Fine-tune] Loading full model weights from {args.resume_weights}...")
        ckpt = torch.load(args.resume_weights, map_location=device)
        sd = {k.replace("module.", ""): v for k, v in (ckpt.get("ema") or ckpt.get("model") or ckpt).items()}
        model.load_state_dict(sd)
        if "val_loss" in ckpt:
            best_val_loss = float(ckpt["val_loss"])
            print(f"[Fine-tune] Inherited best val_loss: {best_val_loss:.4f}")

    # Initially freeze backbone if freeze_epochs > 0
    if args.freeze_epochs > 0:
        model.freeze_backbone()

    criterion = HeadDetectionLoss()
    ema = ModelEMA(model) if args.use_ema else None

    # Optimizer (decay on weights, no decay on BN/bias)
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim <= 1 or "bias" in n or "bn" in n:
                nodecay_params.append(p)
            else:
                decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "lr": args.lr, "initial_lr": args.lr, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "lr": args.lr, "initial_lr": args.lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.save_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, int(len(train_loader) * 2))
    global_step = 0

    print(f"[Training] Starting {args.epochs} epochs (Freeze: {args.freeze_epochs} epochs)...")
    start_time = time.time()

    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Disable mosaic / augmentations in final epochs
        if args.close_mosaic > 0 and epoch == (args.epochs - args.close_mosaic + 1):
            print(f"[Training] Epoch {epoch}: Closing mosaic and affine augmentations for final convergence clean-up.")
            train_dataset.augment = False

        # Phase switch: Unfreeze backbone
        if epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()
            # Rebuild parameter groups with lower LR for unfrozen backbone
            backbone_params = list(model.backbone.parameters())
            head_params = list(model.neck.parameters()) + list(model.head.parameters())
            bb_lr = args.lr * 0.1
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": bb_lr, "initial_lr": bb_lr, "weight_decay": args.weight_decay},
                    {"params": head_params, "lr": args.lr, "initial_lr": args.lr, "weight_decay": args.weight_decay},
                ],
                betas=(0.9, 0.999),
            )

        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_box = 0.0

        for batch_i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            targets = [t.to(device, non_blocking=True) for t in targets]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(images)
            # Loss computed in FP32 outside autocast to prevent geometric squaring overflow
            loss, cls_loss, box_loss = criterion(preds, targets, img_size=(args.imgsz, args.imgsz))
            loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_i + 1) % args.grad_accum == 0 or (batch_i + 1) == len(train_loader):
                global_step += 1
                # Cosine LR scheduling with warmup
                if global_step < warmup_steps:
                    lr_factor = float(global_step) / float(max(1, warmup_steps))
                else:
                    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    lr_factor = max(lr_factor, 0.05)

                for g in optimizer.param_groups:
                    g["lr"] = g.get("initial_lr", args.lr) * lr_factor

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema:
                    ema.update(model)

            running_loss += (loss.item() * args.grad_accum)
            running_cls += cls_loss.item()
            running_box += box_loss.item()

        n_batches = max(len(train_loader), 1)
        epoch_loss = running_loss / n_batches
        epoch_cls = running_cls / n_batches
        epoch_box = running_box / n_batches
        epoch_duration = time.time() - epoch_start

        # Validation loop
        is_eval_epoch = (epoch % args.eval_interval == 0) or (epoch == args.epochs)
        val_loss_str = ""

        if is_eval_epoch and len(val_dataset) > 0:
            eval_model = ema.ema if ema else model
            eval_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_imgs, val_tgts in val_loader:
                    val_imgs = val_imgs.to(device, non_blocking=True)
                    if device.type == "cuda":
                        val_imgs = val_imgs.contiguous(memory_format=torch.channels_last)
                    val_tgts = [t.to(device, non_blocking=True) for t in val_tgts]
                    # Compute eval loss by calling head directly
                    p3, p4, p5 = eval_model.backbone(val_imgs)
                    n3, n4, n5 = eval_model.neck(p3, p4, p5)
                    raw_preds = eval_model.head((n3, n4, n5))
                    v_loss, _, _ = criterion(raw_preds, val_tgts, img_size=(args.imgsz, args.imgsz))
                    val_loss += v_loss.item()

            val_loss /= max(len(val_loader), 1)
            val_loss_str = f" | val_loss {val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(
                    {"model": (ema.ema if ema else model).state_dict(), "epoch": epoch, "val_loss": val_loss},
                    out_dir / "best.pth",
                )
                val_loss_str += " (*best)"
            else:
                epochs_no_improve += 1
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    print(f"[Early Stopping] No improvement in val_loss for {epochs_no_improve} eval intervals ({epochs_no_improve * args.eval_interval} epochs). Stopping early to prevent over-fitting.")
                    break

        # Save last checkpoint
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "loss": epoch_loss},
            out_dir / "last.pth",
        )

        elapsed_total = (time.time() - start_time) / 60.0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] ({epoch_duration:.1f}s) | "
            f"Loss: {epoch_loss:.4f} (cls: {epoch_cls:.4f}, box: {epoch_box:.4f})"
            f"{val_loss_str} | Elapsed: {elapsed_total:.1f}m",
            flush=True,
        )

    print(f"\n[Training Complete] Best weights saved to {out_dir / 'best.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train AvianHeadDetector on overhead CCTV.")
    parser.add_argument("--data", type=str, default="data/head_dataset/head.yaml", help="Path to head.yaml")
    parser.add_argument("--pretrained", type=str, default="avian_coco_micro.pth", help="Pretrained backbone")
    parser.add_argument("--resume-weights", type=str, default=None, help="Path to full model checkpoint (best.pth) to resume/finetune from")
    parser.add_argument("--close-mosaic", type=int, default=15, help="Number of final epochs to disable mosaic & heavy augmentations")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val_loss")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--freeze-epochs", type=int, default=10, help="Epochs to freeze backbone")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass (default 4)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--cache-ram", action="store_true", help="Cache dataset images in RAM for 0-latency I/O")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use Exponential Moving Average")
    parser.add_argument("--eval-interval", type=int, default=5, help="Epoch interval between evaluations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    parser.add_argument("--name", type=str, default="avian_head_run", help="Experiment name")
    parser.add_argument("--save-dir", type=str, default="runs/head_detector", help="Output directory")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
