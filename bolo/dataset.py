"""PyTorch Dataset & Overhead Data Augmentation Pipeline for Head Detection.

Optimized for overhead/ceiling fisheye and booth CCTV imagery:
- Mosaic augmentation (4 images combined)
- 360-degree all-angle continuous rotation (vital for top-down overhead cameras)
- Affine/perspective scaling and translation
- HSV color jitter
- RAM caching option (--cache-ram) for zero-latency I/O on 30GB+ systems
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def augment_hsv(img: np.ndarray, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4):
    """HSV color space jitter."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_affine(
    img: np.ndarray,
    targets: np.ndarray,
    degrees: float = 360.0,
    translate: float = 0.1,
    scale: float = 0.2,
    shear: float = 2.0,
    border: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous 360-degree rotation and affine transformation for overhead CCTV."""
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined matrix
    M = T @ R @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform bounding boxes
    n = len(targets)
    if n:
        # targets: [N, 4] (x1, y1, x2, y2)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        # New enclosing boxes (AABB)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_boxes = np.column_stack((x.min(1), y.min(1), x.max(1), y.max(1)))

        # Clip inside image
        new_boxes[:, 0] = np.clip(new_boxes[:, 0], 0, width)
        new_boxes[:, 1] = np.clip(new_boxes[:, 1], 0, height)
        new_boxes[:, 2] = np.clip(new_boxes[:, 2], 0, width)
        new_boxes[:, 3] = np.clip(new_boxes[:, 3], 0, height)

        # Filter valid boxes
        w = new_boxes[:, 2] - new_boxes[:, 0]
        h = new_boxes[:, 3] - new_boxes[:, 1]
        valid = (w > 4) & (h > 4)
        targets = new_boxes[valid]
    return img, targets


class HeadDetectionDataset(Dataset):
    """Head Detection Dataset with RAM caching and overhead augmentations."""

    def __init__(
        self,
        img_dir: Path,
        lbl_dir: Path,
        imgsz: int = 640,
        augment: bool = True,
        cache_ram: bool = False,
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.imgsz = imgsz
        self.augment = augment

        self.img_files = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")) + list(self.img_dir.glob("*.jpeg"))
        )

        self.labels = []
        for img_p in self.img_files:
            lbl_p = self.lbl_dir / f"{img_p.stem}.txt"
            boxes = []
            if lbl_p.exists():
                with open(lbl_p) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            boxes.append([cls_id, cx, cy, w, h])
            self.labels.append(np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32))

        # RAM Caching
        self.cached_images = {}
        if cache_ram and len(self.img_files) > 0:
            print(f"[Dataset] Caching {len(self.img_files)} images in RAM...")
            for idx in range(len(self.img_files)):
                im = cv2.imread(str(self.img_files[idx]))
                if im is not None:
                    # Pre-resize to target resolution to save RAM and CPU during epoch loops
                    im = cv2.resize(im, (imgsz, imgsz))
                    self.cached_images[idx] = im
            print(f"[Dataset] Cached {len(self.cached_images)} images in RAM.")

    def __len__(self) -> int:
        return len(self.img_files)

    def load_image(self, idx: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        if idx in self.cached_images:
            return self.cached_images[idx].copy(), (self.imgsz, self.imgsz)
        img = cv2.imread(str(self.img_files[idx]))
        if img is None:
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        orig_shape = img.shape[:2]
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        return img, orig_shape

    def load_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """4-image mosaic for overhead crowded scenes."""
        s = self.imgsz
        yc, xc = (int(random.uniform(s // 2, 3 * s // 2)) for _ in range(2))
        indices = [idx] + [random.randint(0, len(self.img_files) - 1) for _ in range(3)]

        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_boxes = []

        for i, index in enumerate(indices):
            img, _ = self.load_image(index)
            h, w = img.shape[:2]

            # Place in mosaic
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            lbl = self.labels[index]
            if len(lbl):
                # Normalized [cx, cy, w, h] -> pixel [x1, y1, x2, y2]
                bx1 = (lbl[:, 1] - lbl[:, 3] / 2) * w + padw
                by1 = (lbl[:, 2] - lbl[:, 4] / 2) * h + padh
                bx2 = (lbl[:, 1] + lbl[:, 3] / 2) * w + padw
                by2 = (lbl[:, 2] + lbl[:, 4] / 2) * h + padh
                boxes = np.column_stack([lbl[:, 0], bx1, by1, bx2, by2])
                mosaic_boxes.append(boxes)

        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            np.clip(mosaic_boxes[:, 1:], 0, 2 * s, out=mosaic_boxes[:, 1:])

        # Random crop back to [s, s]
        mosaic_img, boxes_xy = random_affine(
            mosaic_img,
            mosaic_boxes[:, 1:] if len(mosaic_boxes) else np.zeros((0, 4)),
            degrees=180.0,
            translate=0.2,
            scale=0.2,
            border=(-s // 2, -s // 2),
        )
        if len(mosaic_boxes) and len(boxes_xy):
            mosaic_boxes = np.column_stack([np.zeros(len(boxes_xy)), boxes_xy])
        else:
            mosaic_boxes = np.zeros((0, 5), dtype=np.float32)

        return mosaic_img, mosaic_boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.augment and random.random() < 0.5:
            img, boxes = self.load_mosaic(idx)
        else:
            img, _ = self.load_image(idx)
            lbl = self.labels[idx]
            if len(lbl):
                # Pixel space [x1, y1, x2, y2]
                bx1 = (lbl[:, 1] - lbl[:, 3] / 2) * self.imgsz
                by1 = (lbl[:, 2] - lbl[:, 4] / 2) * self.imgsz
                bx2 = (lbl[:, 1] + lbl[:, 3] / 2) * self.imgsz
                by2 = (lbl[:, 2] + lbl[:, 4] / 2) * self.imgsz
                boxes = np.column_stack([lbl[:, 0], bx1, by1, bx2, by2])
            else:
                boxes = np.zeros((0, 5), dtype=np.float32)

            if self.augment:
                # Overhead 360-degree rotation and HSV jitter
                augment_hsv(img)
                b_xy = boxes[:, 1:] if len(boxes) else np.zeros((0, 4))
                img, b_xy = random_affine(img, b_xy, degrees=360.0, translate=0.1, scale=0.1)
                if len(boxes) and len(b_xy):
                    boxes = np.column_stack([np.zeros(len(b_xy)), b_xy])
                else:
                    boxes = np.zeros((0, 5), dtype=np.float32)

        # Normalize image to [0, 1] RGB
        img_rgb = img[:, :, ::-1].transpose(2, 0, 1)  # HWC BGR -> CHW RGB
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).float() / 255.0
        target_tensor = torch.from_numpy(boxes.astype(np.float32))
        return img_tensor, target_tensor


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
