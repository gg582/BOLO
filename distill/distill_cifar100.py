#!/usr/bin/env python3
"""
Knowledge distillation: ViT-B/16 (IN-21k -> CIFAR-100, ~92.6%) teacher -> AvianRaptorNet-Fast student.

The student is initialised from the best CIFAR-100 checkpoint (avian_raptor_best.pth, 81.27%).
Online, "consistent" distillation (Beyer et al., 2022): teacher and student see exactly the same
augmented + mixed image every step; the teacher gets it upsampled to 224x224.

Usage:
    python -m distill.distill_cifar100 --epochs 120                       # stage 1: vanilla KD
    python -m distill.distill_cifar100 --loss dkd --epochs 300 --lr 0.005 \
        --student-weights avian_raptor_distilled_last.pth --out avian_raptor_dkd   # stage 2: DKD
"""
import argparse
import copy
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from core.avian_model import AvianRaptorNet_Fast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CIFAR_MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
TEACHER_URL = ("https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/"
               "resolve/main/pytorch_model.bin")


def build_teacher(path, device):
    teacher = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.hub.download_url_to_file(TEACHER_URL, path)
    teacher.load_state_dict(torch.load(path, map_location="cpu"))
    teacher = teacher.to(device).eval().half()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def to_teacher_input(x_student, mean, std):
    """Student-normalised 32x32 batch -> teacher-normalised (mean=std=0.5) 224x224 batch."""
    x = x_student * std + mean                      # back to [0,1]
    x = (x - 0.5) / 0.5
    return F.interpolate(x, size=224, mode="bicubic", align_corners=False).half()


def mix_batch(x, y, mixup_alpha, cutmix_alpha, cutmix_prob):
    """Mixup or CutMix. Returns mixed x, (y_a, y_b, lam)."""
    idx = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < cutmix_prob:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
        x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], float(lam)


def dkd_loss(s_logits, t_logits, target, alpha, beta, T):
    """Decoupled KD (Zhao et al., CVPR 2022): alpha * TCKD + beta * NCKD."""
    gt = F.one_hot(target, s_logits.size(1)).bool()
    p_s = F.softmax(s_logits / T, 1)
    p_t = F.softmax(t_logits / T, 1)
    # Target-class KD: binary (target vs. rest) distributions.
    b_s = torch.stack([(p_s * gt).sum(1), (p_s * ~gt).sum(1)], 1).clamp_min(1e-8)
    b_t = torch.stack([(p_t * gt).sum(1), (p_t * ~gt).sum(1)], 1).clamp_min(1e-8)
    tckd = (b_t * (b_t.log() - b_s.log())).sum(1).mean() * T * T
    # Non-target-class KD: distributions over the non-target classes only.
    log_q_s = F.log_softmax(s_logits / T - 1000.0 * gt, 1)
    q_t = F.softmax(t_logits / T - 1000.0 * gt, 1)
    nckd = F.kl_div(log_q_s, q_t, reduction="batchmean") * T * T
    return alpha * tckd + beta * nckd


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def ema_update(ema, model, decay):
    msd = model.state_dict()
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
        else:
            v.copy_(msd[k])


def main():
    ap = argparse.ArgumentParser(description="Distill ViT-B/16 into AvianRaptorNet-Fast on CIFAR-100")
    ap.add_argument("--student-weights", default="avian_raptor_best.pth")
    ap.add_argument("--teacher-weights", default="teachers/vit_b16_in21k_ft_cifar100.bin")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.9, help="weight of KD loss (1-alpha for CE)")
    ap.add_argument("--loss", choices=["kd", "dkd"], default="kd")
    ap.add_argument("--dkd-alpha", type=float, default=1.0, help="DKD: TCKD weight")
    ap.add_argument("--dkd-beta", type=float, default=8.0, help="DKD: NCKD weight")
    ap.add_argument("--dkd-warmup-epochs", type=int, default=5, help="DKD: linear ramp of KD weight")
    ap.add_argument("--mixup-alpha", type=float, default=0.8)
    ap.add_argument("--cutmix-alpha", type=float, default=1.0)
    ap.add_argument("--cutmix-prob", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--out", default="avian_raptor_distilled")
    ap.add_argument("--limit-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    args = ap.parse_args()

    device = torch.device("cuda")
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)
    stats = (CIFAR_MEAN.flatten().tolist(), CIFAR_STD.flatten().tolist())

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(*stats),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(*stats)])
    trainset = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=train_tf)
    testset = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)

    student = AvianRaptorNet_Fast(num_classes=100).to(device)
    student.load_state_dict(torch.load(args.student_weights, map_location=device), strict=True)
    student = student.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    teacher = build_teacher(args.teacher_weights, device)

    base_acc = evaluate(student, test_loader, device)
    print(f"Student baseline ({args.student_weights}): {base_acc:.2f}%", flush=True)

    # No weight decay on BN / bias parameters.
    decay, no_decay = [], []
    for n, p in student.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.SGD([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, momentum=0.9, nesterov=True)

    steps_per_epoch = args.limit_steps or len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    scaler = torch.amp.GradScaler("cuda")
    Tk = args.temperature
    best = {"raw": base_acc, "ema": base_acc}
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        student.train()
        run_loss = run_kd = 0.0
        agree = n_seen = 0
        for i, (x, y) in enumerate(train_loader):
            if args.limit_steps and i >= args.limit_steps:
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, ya, yb, lam = mix_batch(x, y, args.mixup_alpha, args.cutmix_alpha, args.cutmix_prob)

            with torch.no_grad():
                t_logits = teacher(to_teacher_input(x, mean, std)).float()

            with torch.autocast("cuda", dtype=torch.float16):
                s_logits = student(x.contiguous(memory_format=torch.channels_last))
            s_logits = s_logits.float()

            if args.loss == "dkd":
                ramp = min(1.0, (step + 1) / max(1, args.dkd_warmup_epochs * steps_per_epoch))
                kd = ramp * (lam * dkd_loss(s_logits, t_logits, ya, args.dkd_alpha, args.dkd_beta, Tk)
                             + (1 - lam) * dkd_loss(s_logits, t_logits, yb, args.dkd_alpha, args.dkd_beta, Tk))
            else:
                kd = F.kl_div(F.log_softmax(s_logits / Tk, 1), F.softmax(t_logits / Tk, 1),
                              reduction="batchmean") * Tk * Tk
            ce = lam * F.cross_entropy(s_logits, ya) + (1 - lam) * F.cross_entropy(s_logits, yb)
            loss = args.alpha * kd + (1 - args.alpha) * ce

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            ema_update(ema, student, args.ema_decay if step > warmup_steps else 0.0)
            step += 1

            run_loss += loss.item()
            run_kd += kd.item()
            agree += s_logits.argmax(1).eq(t_logits.argmax(1)).sum().item()
            n_seen += y.size(0)

        acc = evaluate(student, test_loader, device)
        ema_acc = evaluate(ema, test_loader, device)
        tag = ""
        if acc > best["raw"]:
            best["raw"] = acc
            torch.save(student.state_dict(), f"{args.out}.pth")
            tag += " *raw"
        if ema_acc > best["ema"]:
            best["ema"] = ema_acc
            torch.save(ema.state_dict(), f"{args.out}_ema.pth")
            tag += " *ema"
        torch.save(student.state_dict(), f"{args.out}_last.pth")
        nb = min(i + 1, steps_per_epoch)
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | lr {lr_at(step - 1):.5f} | loss {run_loss / nb:.4f} "
              f"| kd {run_kd / nb:.4f} | T-agree {100 * agree / n_seen:.1f}% | test {acc:.2f}% "
              f"| ema {ema_acc:.2f}% | best raw {best['raw']:.2f} ema {best['ema']:.2f} "
              f"| {(time.time() - t0) / 60:.1f} min{tag}", flush=True)

    torch.save(student.state_dict(), f"{args.out}_last.pth")
    print(f"Done. Baseline {base_acc:.2f}% -> best raw {best['raw']:.2f}%, best EMA {best['ema']:.2f}%")


if __name__ == "__main__":
    main()
