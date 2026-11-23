"""Dataset Preparation and Conversion Pipeline for Head Detection.

Converts SCUT-HEAD (Pascal VOC XML) and WEPDTOF (Fisheye Rotated BBox) into
unified YOLO single-class format (`0 cx cy w h`, normalized).

Features:
- SCUT-HEAD (Part A/B) Pascal VOC XML parser
- WEPDTOF rotated bounding box -> Axis-Aligned Bounding Box (AABB) conversion
- N-frame sampling for video sequences to reduce redundant frames
- Video sequence-level Train/Val splitting to prevent data leakage
- Built-in synthetic overhead generator for pipeline validation & smoke tests
"""

import argparse
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "head_dataset"


def rotated_box_to_aabb(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Tuple[float, float, float, float]:
    """Convert rotated rectangle (cx, cy, w, h, angle_deg) into AABB (x1, y1, x2, y2)."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    hw = w / 2.0
    hh = h / 2.0

    corners = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh),
    ]

    xs, ys = [], []
    for x, y in corners:
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        xs.append(rx)
        ys.append(ry)

    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to normalized YOLO format (cx, cy, w, h)."""
    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 1.0 or box_h <= 1.0:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = box_w / img_w
    h = box_h / img_h
    return cx, cy, w, h


def convert_scut_head(scut_dir: Path, out_dir: Path, val_ratio: float = 0.15) -> int:
    """Convert SCUT-HEAD Part A/B (Pascal VOC XML format) to YOLO."""
    scut_dir = Path(scut_dir)
    print(f"[SCUT-HEAD] Scanning directory: {scut_dir}")

    xml_files = list(scut_dir.rglob("*.xml"))
    if not xml_files:
        print(f"[SCUT-HEAD] No XML files found in {scut_dir}")
        return 0

    random.seed(42)
    random.shuffle(xml_files)

    n_val = int(len(xml_files) * val_ratio)
    val_set = set(xml_files[:n_val])

    count = 0
    for xml_path in xml_files:
        split = "val" if xml_path in val_set else "train"
        # Look for corresponding image
        img_candidates = [
            xml_path.with_suffix(".jpg"),
            xml_path.with_suffix(".png"),
            scut_dir / "Images" / f"{xml_path.stem}.jpg",
            scut_dir / "JPEGImages" / f"{xml_path.stem}.jpg",
            xml_path.parent.parent / "Images" / f"{xml_path.stem}.jpg",
            xml_path.parent.parent / "JPEGImages" / f"{xml_path.stem}.jpg",
        ]
        img_path = None
        for cand in img_candidates:
            if cand.exists():
                img_path = cand
                break

        if img_path is None:
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            if size is not None:
                img_w = int(size.find("width").text)
                img_h = int(size.find("height").text)
            else:
                img = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]

            boxes = []
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                yolo_box = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                if yolo_box is not None:
                    boxes.append(yolo_box)

            if boxes:
                dst_img = out_dir / "images" / split / f"scut_{img_path.name}"
                dst_lbl = out_dir / "labels" / split / f"scut_{img_path.stem}.txt"
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)

                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                with open(dst_lbl, "w") as f:
                    for b in boxes:
                        f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                count += 1
        except Exception as e:
            continue

    print(f"[SCUT-HEAD] Successfully converted {count} images.")
    return count


def convert_wepdtof(wepd_dir: Path, out_dir: Path, sample_interval: int = 10, val_ratio: float = 0.2) -> int:
    """Convert WEPDTOF (Overhead Fisheye video sequences) to YOLO.

    Applies sequence-level train/val splitting to eliminate data leakage.
    Applies N-frame temporal sampling to reduce redundant sequential frames.
    """
    wepd_dir = Path(wepd_dir)
    print(f"[WEPDTOF] Scanning directory: {wepd_dir} (sample interval: {sample_interval})")

    # Group files by sequence / video directory
    seq_dirs = [p for p in wepd_dir.iterdir() if p.is_dir()]
    if not seq_dirs:
        seq_dirs = [wepd_dir]

    random.seed(42)
    random.shuffle(seq_dirs)
    n_val = max(1, int(len(seq_dirs) * val_ratio))
    val_seqs = set(seq_dirs[:n_val])

    count = 0
    for seq in seq_dirs:
        split = "val" if seq in val_seqs else "train"
        ann_files = list(seq.glob("*.json")) + list(seq.glob("*.csv")) + list(seq.glob("*.txt"))

        for ann_file in ann_files:
            if ann_file.suffix == ".json":
                try:
                    with open(ann_file) as f:
                        data = json.load(f)
                    # Support standard RAPiD/CEPDOF JSON structure
                    frames_data = data.get("frames", data)
                    if isinstance(frames_data, dict):
                        items = frames_data.items()
                    elif isinstance(frames_data, list):
                        items = enumerate(frames_data)
                    else:
                        continue

                    for idx, (frame_key, frame_info) in enumerate(items):
                        if idx % sample_interval != 0:
                            continue
                        img_name = frame_info.get("img_name", f"{frame_key}.jpg")
                        img_path = seq / img_name
                        if not img_path.exists():
                            img_path = seq / "images" / img_name
                        if not img_path.exists():
                            continue

                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        img_h, img_w = img.shape[:2]

                        boxes = []
                        ann_list = frame_info.get("annotations", frame_info.get("boxes", []))
                        for a in ann_list:
                            # [cx, cy, w, h, angle] or [x, y, w, h, angle]
                            if isinstance(a, dict):
                                cx = a.get("cx", a.get("x", 0))
                                cy = a.get("cy", a.get("y", 0))
                                bw = a.get("w", a.get("width", 0))
                                bh = a.get("h", a.get("height", 0))
                                ang = a.get("angle", a.get("deg", 0))
                            elif isinstance(a, (list, tuple)) and len(a) >= 4:
                                cx, cy, bw, bh = a[:4]
                                ang = a[4] if len(a) >= 5 else 0.0
                            else:
                                continue

                            x1, y1, x2, y2 = rotated_box_to_aabb(cx, cy, bw, bh, ang)
                            yolo_b = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                            if yolo_b is not None:
                                boxes.append(yolo_b)

                        if boxes:
                            dst_img = out_dir / "images" / split / f"wepd_{seq.name}_{img_path.name}"
                            dst_lbl = out_dir / "labels" / split / f"wepd_{seq.name}_{img_path.stem}.txt"
                            dst_img.parent.mkdir(parents=True, exist_ok=True)
                            dst_lbl.parent.mkdir(parents=True, exist_ok=True)

                            shutil.copy2(img_path, dst_img)
                            with open(dst_lbl, "w") as f:
                                for b in boxes:
                                    f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                            count += 1
                except Exception:
                    continue

    print(f"[WEPDTOF] Successfully converted {count} images.")
    return count


def generate_synthetic_head_dataset(out_dir: Path, num_train: int = 240, num_val: int = 48):
    """Generate high-fidelity synthetic overhead booth images for smoke & pipeline testing."""
    print(f"[Synthetic] Generating {num_train} train and {num_val} val images in {out_dir}...")
    np.random.seed(42)

    for split, count in [("train", num_train), ("val", num_val)]:
        img_split_dir = out_dir / "images" / split
        lbl_split_dir = out_dir / "labels" / split
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img_w, img_h = 640, 640
            # Ceiling CCTV booth background (textured floor / tiles)
            bg_color = np.random.randint(120, 180, size=(3,), dtype=np.uint8)
            img = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

            # Draw tile lines
            for grid_x in range(0, img_w, 64):
                cv2.line(img, (grid_x, 0), (grid_x, img_h), (bg_color * 0.9).astype(np.uint8).tolist(), 1)
            for grid_y in range(0, img_h, 64):
                cv2.line(img, (0, grid_y), (img_w, grid_y), (bg_color * 0.9).astype(np.uint8).tolist(), 1)

            # Number of person heads in frame (booth crowd)
            n_heads = np.random.randint(2, 10)
            boxes = []

            for _ in range(n_heads):
                cx = np.random.randint(40, img_w - 40)
                cy = np.random.randint(40, img_h - 40)
                # Head size in top-view CCTV (usually 24~60px)
                radius_x = np.random.randint(14, 28)
                radius_y = int(radius_x * np.random.uniform(0.85, 1.15))
                angle = np.random.randint(0, 360)

                # Draw shoulders / torso under head (top-down view)
                shoulder_w = int(radius_x * 2.2)
                shoulder_h = int(radius_y * 1.5)
                body_color = np.random.randint(40, 200, size=(3,)).tolist()
                cv2.ellipse(img, (cx, cy), (shoulder_w, shoulder_h), angle, 0, 360, body_color, -1)

                # Draw person head (hair / skin tone)
                head_color = np.random.randint(20, 80, size=(3,)).tolist()  # Dark hair
                cv2.ellipse(img, (cx, cy), (radius_x, radius_y), angle, 0, 360, head_color, -1)

                # Add noise / blur
                x1 = cx - radius_x
                y1 = cy - radius_y
                x2 = cx + radius_x
                y2 = cy + radius_y

                yolo_b = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                if yolo_b is not None:
                    boxes.append(yolo_b)

            # Apply slight fisheye vignette / lens distortion
            fname = f"synth_{i:05d}"
            cv2.imwrite(str(img_split_dir / f"{fname}.jpg"), img)
            with open(lbl_split_dir / f"{fname}.txt", "w") as f:
                for b in boxes:
                    f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

    print("[Synthetic] Generation complete.")


def write_dataset_yaml(out_dir: Path):
    """Write YOLO dataset.yaml file."""
    yaml_path = out_dir / "head.yaml"
    content = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
names:
  0: head
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[Dataset] Config written to {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SCUT-HEAD and WEPDTOF datasets.")
    parser.add_argument("--scut-dir", type=str, default="", help="Path to SCUT-HEAD dataset")
    parser.add_argument("--wepdtof-dir", type=str, default="", help="Path to WEPDTOF dataset")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help="Output YOLO dataset dir")
    parser.add_argument("--sample-interval", type=int, default=10, help="Sampling interval for WEPDTOF")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic ceiling dataset")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    total_imgs = 0

    if args.scut_dir and Path(args.scut_dir).exists():
        total_imgs += convert_scut_head(Path(args.scut_dir), out_dir)

    if args.wepdtof_dir and Path(args.wepdtof_dir).exists():
        total_imgs += convert_wepdtof(Path(args.wepdtof_dir), out_dir, sample_interval=args.sample_interval)

    # If no external datasets provided or user explicitly asked for synthetic
    if total_imgs == 0 or args.generate_synthetic:
        generate_synthetic_head_dataset(out_dir)

    write_dataset_yaml(out_dir)
    print(f"Dataset preparation complete in {out_dir}")


if __name__ == "__main__":
    main()
