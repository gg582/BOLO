"""BOLO benchmark on real CCTV footage: complexity, latency, and blur robustness.

Measures, on a single open CCTV clip (default: the Oxford Town Centre excerpt
used by ``demo.py``):

1. Parameters and GFLOPs at the inference resolution.
2. Latency of the model forward pass (batch 1) for FP32 / FP16 on GPU and
   FP32 on CPU, plus end-to-end per-frame latency (pre-process + forward +
   NMS) through the ultralytics predictor on real CCTV frames.
3. Motion-blur robustness. The footage has no box labels for the BOLO
   classes, so the detections on the clean frame are used as pseudo ground
   truth. Each sampled frame is then blurred with a linear motion kernel of a
   fixed length (random but seeded direction per frame), and we report how
   many clean detections survive (pseudo-recall), how many blurred detections
   agree with a clean one (pseudo-precision), and mean confidence. This is a
   *consistency* measure, not mAP against human labels.

Usage:
    python -m bolo.benchmark_cctv
    python -m bolo.benchmark_cctv --source data/cctv_samples/TownCentreXVID.mp4 --stride 25
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bolo.model import register_bolo_modules

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "cctv_samples" / "cctv_default_demo.mp4"
DEFAULT_WEIGHTS = ROOT / "runs" / "bolo" / "bolo_everyday2" / "weights" / "best.pt"
BLUR_KERNELS = (0, 5, 11, 21, 31)


def motion_kernel(k, angle):
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, rot, (k, k))
    return kern / kern.sum()


def read_frames(source, stride, limit):
    cap = cv2.VideoCapture(str(source))
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            if limit and len(frames) >= limit:
                break
        idx += 1
    cap.release()
    return frames


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(ref, pred, iou_thr=0.5):
    """Greedy class-aware matching. Returns number of matched pairs."""
    (rb, rc), (pb, pc, ps) = ref, pred
    if len(rb) == 0 or len(pb) == 0:
        return 0
    iou = box_iou(pb, rb)
    iou[pc[:, None] != rc[None, :]] = 0.0
    used, tp = set(), 0
    for i in np.argsort(-ps):
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= iou_thr and j not in used:
            used.add(j)
            tp += 1
    return tp


def to_np(result):
    b = result.boxes
    return b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()


@torch.no_grad()
def time_forward(net, device, half, imgsz, warmup, iters):
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    m = net.to(device)
    if half:
        m, x = m.half(), x.half()
    for _ in range(warmup):
        m(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    m.float()
    return float(np.median(ts)), float(np.percentile(ts, 90))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=5, help="use every N-th frame")
    ap.add_argument("--limit", type=int, default=0, help="max frames (0 = all sampled)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cpu-iters", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "runs" / "bolo" / "benchmark_cctv.json"))
    args = ap.parse_args()

    register_bolo_modules()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    model = YOLO(args.weights)
    net = model.model.fuse().eval()
    n_params = sum(p.numel() for p in net.parameters())
    gflops = get_flops(net, args.imgsz)
    print(f"params: {n_params / 1e6:.2f}M  GFLOPs@{args.imgsz}: {gflops:.2f}")

    gpu = torch.device("cuda:0") if torch.cuda.is_available() else None
    report = {
        "weights": str(Path(args.weights).relative_to(ROOT)) if args.weights.startswith(str(ROOT)) else args.weights,
        "source": Path(args.source).name,
        "imgsz": args.imgsz,
        "params_M": round(n_params / 1e6, 3),
        "gflops": round(gflops, 2),
        "env": {
            "gpu": torch.cuda.get_device_name(0) if gpu else None,
            "cpu": platform.processor() or platform.machine(),
            "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__,
        },
        "latency_forward_ms": {},
    }

    # --- forward-pass latency --------------------------------------------
    if gpu:
        for half in (False, True):
            med, p90 = time_forward(net, gpu, half, args.imgsz, 30, args.iters)
            key = "gpu_fp16" if half else "gpu_fp32"
            report["latency_forward_ms"][key] = {"median": round(med, 2), "p90": round(p90, 2)}
            print(f"{key}: median {med:.2f} ms  p90 {p90:.2f} ms")
    med, p90 = time_forward(net, torch.device("cpu"), False, args.imgsz, 5, args.cpu_iters)
    report["latency_forward_ms"]["cpu_fp32"] = {"median": round(med, 2), "p90": round(p90, 2)}
    print(f"cpu_fp32: median {med:.2f} ms  p90 {p90:.2f} ms")

    # --- end-to-end on real CCTV frames + blur sweep -----------------------
    frames = read_frames(args.source, args.stride, args.limit)
    print(f"{len(frames)} frames from {Path(args.source).name}")
    device = 0 if gpu else "cpu"
    pred_kw = dict(imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)

    def predict(img, half=False):
        return model.predict(img, half=half, **pred_kw)[0]

    for _ in range(10):
        predict(frames[0], half=True)
    e2e = []
    for f in frames:
        r = predict(f, half=True)
        e2e.append(sum(r.speed.values()))
    report["latency_end_to_end_ms_gpu_fp16"] = {
        "median": round(float(np.median(e2e)), 2),
        "fps": round(1000.0 / float(np.median(e2e)), 1),
    }
    print(f"end-to-end gpu_fp16: median {np.median(e2e):.2f} ms/frame")

    rng = np.random.default_rng(0)
    angles = rng.uniform(0.0, 180.0, size=len(frames))
    clean = [to_np(predict(f)) for f in frames]
    n_ref = sum(len(c[0]) for c in clean)
    sweep = []
    for k in BLUR_KERNELS:
        tp = n_pred = 0
        confs = []
        for f, ref, ang in zip(frames, clean, angles):
            img = f if k == 0 else cv2.filter2D(f, -1, motion_kernel(k, ang))
            pb, pc, ps = to_np(predict(img))
            tp += match((ref[0], ref[1]), (pb, pc, ps))
            n_pred += len(pb)
            confs.extend(ps.tolist())
        row = {
            "kernel_px": k,
            "dets_per_frame": round(n_pred / len(frames), 2),
            "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 3),
            "pseudo_recall": round(tp / max(n_ref, 1), 3),
            "pseudo_precision": round(tp / max(n_pred, 1), 3),
        }
        sweep.append(row)
        print(row)
    report["blur_sweep"] = {"frames": len(frames), "conf": args.conf, "iou": 0.5, "rows": sweep}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
