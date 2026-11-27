"""Complexity and latency of the AvianRaptorNet classifiers and the BOLO ONNX exports.

- AvianRaptorNet_Fast @ 32x32 (CIFAR-100, 100 classes) and @ 128x128
  (COCO object crops, 80 classes): params, GFLOPs, PyTorch latency on GPU
  (FP32 / FP16, batch-1 latency and batched throughput) and CPU (FP32, batch 1).
  Timing depends only on the architecture, so random weights are used.
- BOLO ONNX exports (FP32 / FP16, 640x640) on onnxruntime CPU, the closest
  available stand-in for an embedded CPU target.

Usage:
    python scripts/benchmark_latency.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.avian_model import AvianRaptorNet_Fast  # noqa: E402

ONNX_MODELS = ("bolo_everyday_fp32.onnx", "bolo_everyday_fp16.onnx")


def timed(fn, warmup, iters, sync):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


@torch.no_grad()
def bench_classifier(num_classes, size):
    from ultralytics.utils.torch_utils import get_flops

    net = AvianRaptorNet_Fast(num_classes=num_classes).eval()
    params = sum(p.numel() for p in net.parameters())
    gflops = get_flops(net, size)  # ultralytics reports GFLOPs (2 x MACs)
    row = {"input": f"{size}x{size}", "classes": num_classes,
           "params_M": round(params / 1e6, 3), "gflops": round(gflops, 3)}

    cpu_x = torch.randn(1, 3, size, size)
    row["cpu_fp32_b1_ms"] = round(timed(lambda: net(cpu_x), 5, 50, lambda: None), 2)

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        sync = torch.cuda.synchronize
        bs = 256 if size <= 32 else 32  # 8 GB GPU
        row["throughput_batch"] = bs
        for half in (False, True):
            m = net.to(dev).half() if half else net.to(dev).float()
            dt = torch.float16 if half else torch.float32
            tag = "fp16" if half else "fp32"
            x1 = torch.randn(1, 3, size, size, device=dev, dtype=dt)
            row[f"gpu_{tag}_b1_ms"] = round(timed(lambda: m(x1), 20, 200, sync), 2)
            xb = torch.randn(bs, 3, size, size, device=dev, dtype=dt)
            ms = timed(lambda: m(xb), 5, 30, sync)
            row[f"gpu_{tag}_img_per_s"] = int(bs / ms * 1e3)
            del xb
            torch.cuda.empty_cache()
        net.float().cpu()
    return row


def bench_onnx():
    import onnxruntime as ort

    rows = []
    for name in ONNX_MODELS:
        path = ROOT / name
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dtype = np.float16 if "float16" in inp.type else np.float32
        x = np.random.rand(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(dtype)
        ms = timed(lambda: sess.run(None, {inp.name: x}), 5, 30, lambda: None)
        rows.append({"model": name, "provider": "CPUExecutionProvider",
                     "size_MB": round(path.stat().st_size / 2**20, 1), "median_ms": round(ms, 2)})
        print(rows[-1])
    return rows


def main():
    report = {
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor() or platform.machine(),
            "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__,
        },
        "classifiers": [],
    }
    for nc, size in ((100, 32), (80, 128)):
        row = bench_classifier(nc, size)
        report["classifiers"].append(row)
        print(row)
    report["bolo_onnx_cpu"] = bench_onnx()
    out = ROOT / "runs" / "benchmark_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
