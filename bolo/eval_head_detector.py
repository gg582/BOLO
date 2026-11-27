"""Head-detection accuracy of AvianHeadDetector on the SCUT-HEAD val split.

Uses the same pre-processing as ``demo.py`` / ``bolo/infer_video.py`` (plain
square resize, RGB, /255), torchvision NMS, and the standard ultralytics AP
computation (``ap_per_class``) over IoU 0.50:0.95. Boxes are compared in the
original image coordinates.

Usage:
    python -m bolo.eval_head_detector
    python -m bolo.eval_head_detector --weights runs/head_detector/avian_head_scut_real/best.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as ops

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bolo.head_detector import AvianHeadDetector  # noqa: E402

DATA = ROOT / "data" / "head_dataset"
RUNS = ("avian_head_scut_real", "avian_head_finetune_night")
IOUV = np.linspace(0.5, 0.95, 10)


def load_labels(path, w, h):
    if not path.exists():
        return np.zeros((0, 4))
    rows = [list(map(float, l.split()[1:5])) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows)
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def match(pred, gt):
    """tp[N, 10]: prediction i is a TP at IoU threshold j (greedy by score)."""
    tp = np.zeros((len(pred), len(IOUV)), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    iou = ops.box_iou(torch.as_tensor(pred[:, :4]), torch.as_tensor(gt)).numpy()
    order = np.argsort(-pred[:, 4])
    for j, thr in enumerate(IOUV):
        used = np.zeros(len(gt), dtype=bool)
        for i in order:
            cand = np.where((iou[i] >= thr) & ~used)[0]
            if len(cand):
                g = cand[np.argmax(iou[i, cand])]
                used[g] = True
                tp[i, j] = True
    return tp


@torch.no_grad()
def evaluate(weights, imgsz, conf, iou, device):
    from ultralytics.utils.metrics import ap_per_class

    model = AvianHeadDetector(num_classes=1)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in (ckpt.get("ema") or ckpt.get("model") or ckpt).items()}
    model.load_state_dict(state)
    model.to(device).eval()

    imgs = sorted((DATA / "images" / "val").glob("*.jpg"))
    tps, confs, n_gt, n_pred, times = [], [], 0, 0, []
    for p in imgs:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        x = cv2.cvtColor(cv2.resize(im, (imgsz, imgsz)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255.0)[None].to(device)
        t0 = time.perf_counter()
        out = model(x)[0]
        boxes, scores = out[:, :4], out[:, 4]
        keep = scores > conf
        boxes, scores = boxes[keep], scores[keep]
        k = ops.nms(boxes, scores, iou)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        pred = torch.cat([boxes[k], scores[k, None]], 1).cpu().numpy()
        pred[:, [0, 2]] *= w / imgsz
        pred[:, [1, 3]] *= h / imgsz
        gt = load_labels(DATA / "labels" / "val" / f"{p.stem}.txt", w, h)
        tps.append(match(pred, gt))
        confs.append(pred[:, 4])
        n_gt += len(gt)
        n_pred += len(pred)

    tp = np.concatenate(tps)
    cf = np.concatenate(confs)
    res = ap_per_class(tp, cf, np.zeros(len(cf)), np.zeros(n_gt), plot=False)
    p_, r_, ap = res[2], res[3], res[5]  # precision, recall, AP[nc, 10]
    return {
        "weights": str(Path(weights).relative_to(ROOT)) if str(weights).startswith(str(ROOT)) else str(weights),
        "epoch": ckpt.get("epoch"),
        "images": len(imgs),
        "gt_boxes": int(n_gt),
        "pred_boxes": int(n_pred),
        "AP50": round(float(ap[0, 0]), 4),
        "AP50_95": round(float(ap[0].mean()), 4),
        "precision": round(float(p_[0]), 4),
        "recall": round(float(r_[0]), 4),
        "forward_nms_ms_median": round(float(np.median(times[10:])), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", nargs="*", default=[str(ROOT / "runs" / "head_detector" / r / "best.pth") for r in RUNS])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="low threshold for AP; P/R reported at max-F1")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "runs" / "head_detector" / "eval_scut_val.json"))
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for w in args.weights:
        if not Path(w).exists():
            continue
        row = evaluate(w, args.imgsz, args.conf, args.iou, device)
        rows.append(row)
        print(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"imgsz": args.imgsz, "conf": args.conf, "nms_iou": args.iou,
                                          "rows": rows}, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
