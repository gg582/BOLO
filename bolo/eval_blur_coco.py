"""Labeled motion-blur robustness of BOLO on the COCO part of the everyday val set.

``bolo/prepare_everyday.py`` takes its COCO validation images as the first
``COCO_VAL`` (800) labeled images of a ``random.Random(0)`` shuffle of
``data/coco_subset_20k``. Those images are disjoint from the COCO training
images, and the split only needs the COCO annotations, so it can be rebuilt
on any machine that has the 20k subset (TACO / Office / OpenImages sources
are not needed).

For each motion-blur kernel length the whole set is blurred with a linear
kernel (seeded random direction per image) and evaluated with the standard
ultralytics detection metrics (mAP50, mAP50-95, P, R).

With ``--no-msa`` every ``MotionStrideAttention`` layer is bypassed at test
time (``out = x``) without retraining. Comparing the two runs shows how much
the trained network relies on the motion-correction branch at each blur
level. It is an indirect, post-hoc ablation, not a retrained baseline.

Usage:
    python -m bolo.eval_blur_coco
    python -m bolo.eval_blur_coco --no-msa
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bolo.benchmark_cctv import motion_kernel
from bolo.model import register_bolo_modules
from bolo.prepare_data import load_coco_index
from bolo.prepare_everyday import COCO_MAP, COCO_VAL, NAMES

ROOT = Path(__file__).resolve().parent.parent
COCO = ROOT / "data" / "coco_subset_20k"
OUT = ROOT / "data" / "everyday_coco_val_blur"
DEFAULT_WEIGHTS = ROOT / "runs" / "bolo" / "bolo_everyday2" / "weights" / "best.pt"
KERNELS = (0, 5, 11, 21, 31)


def coco_yolo_labels():
    """Same conversion as bolo/prepare_data.py, kept in memory."""
    images, anns, cat_ids, _ = load_coco_index()
    cat_map = {cid: i for i, cid in enumerate(cat_ids)}
    labels = {}
    for p in (COCO / "images").glob("*.jpg"):
        img_id = int(p.stem)
        if img_id not in images:
            continue
        _, w, h = images[img_id]
        lines = []
        for a in anns.get(img_id, []):
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx, cy, nw, nh = (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                continue
            lines.append(f"{cat_map[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        labels[p.name] = lines
    return labels


def everyday_coco_val():
    """Reproduce prepare_everyday's COCO val selection (first rng use = this shuffle)."""
    labels = coco_yolo_labels()
    rng = random.Random(0)
    files = sorted((COCO / "images").iterdir())
    rng.shuffle(files)
    out = []
    for img in files:
        if img.name not in labels:
            continue
        remapped = []
        for line in labels[img.name]:
            cls, rest = line.split(" ", 1)
            if int(cls) in COCO_MAP:
                remapped.append(f"{COCO_MAP[int(cls)]} {rest}")
        if remapped:
            out.append((img, "\n".join(remapped) + "\n"))
        if len(out) >= COCO_VAL:
            break
    return out


def build_split(entries, k, angles):
    split = OUT / f"k{k:02d}"
    img_dir, lab_dir = split / "images", split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for (img, text), ang in zip(entries, angles):
        dst = img_dir / img.name
        if not dst.exists():
            im = cv2.imread(str(img))
            if k:
                im = cv2.filter2D(im, -1, motion_kernel(k, ang))
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lab_dir / f"{img.stem}.txt").write_text(text)
    yaml = split / "data.yaml"
    yaml.write_text(
        f"path: {split}\ntrain: images\nval: images\nnc: {len(NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    return yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-msa", action="store_true",
                    help="bypass every MotionStrideAttention layer (out = x) at test time")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tag = "_nomsa" if args.no_msa else ""
    args.out = args.out or str(ROOT / "runs" / "bolo" / f"eval_blur_coco{tag}.json")

    entries = everyday_coco_val()
    n_boxes = sum(t.count("\n") for _, t in entries)
    per_cls = defaultdict(int)
    for _, t in entries:
        for line in t.splitlines():
            per_cls[NAMES[int(line.split()[0])]] += 1
    print(f"{len(entries)} images, {n_boxes} boxes, {len(per_cls)} classes")
    angles = np.random.default_rng(0).uniform(0.0, 180.0, size=len(entries))

    register_bolo_modules()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    if args.no_msa:
        from bolo.modules import MotionStrideAttention

        n = 0
        for mod in model.model.modules():
            if isinstance(mod, MotionStrideAttention):
                mod.forward = lambda x: x
                n += 1
        print(f"bypassed {n} MotionStrideAttention layers")
    rows = []
    for k in KERNELS:
        yaml = build_split(entries, k, angles)
        m = model.val(data=str(yaml), imgsz=args.imgsz, batch=args.batch, device=0,
                      plots=False, verbose=False, project=str(OUT / "val"), name=f"k{k:02d}{tag}",
                      exist_ok=True)
        row = {
            "kernel_px": k,
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
        }
        rows.append(row)
        print(row)

    base = rows[0]["mAP50_95"]
    for r in rows:
        r["mAP50_95_retained"] = round(r["mAP50_95"] / base, 3) if base else None
    report = {
        "weights": Path(args.weights).name,
        "msa_bypassed": args.no_msa,
        "images": len(entries),
        "boxes": n_boxes,
        "classes_present": len(per_cls),
        "imgsz": args.imgsz,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
