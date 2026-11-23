"""Export AvianHeadDetector to ONNX (FP32 & FP16) and benchmark inference FPS.

Validates parity between PyTorch and ONNX Runtime and reports latency / FPS
on the active hardware (e.g. RTX 3070).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from bolo.head_detector import AvianHeadDetector


def export_onnx(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Export] Benchmarking on device: {device}")

    # Build model & load weights
    model = AvianHeadDetector(num_classes=1)
    if args.weights and Path(args.weights).exists():
        print(f"[Export] Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print("[Export] No weights specified or file not found; exporting initialized model structure.")

    model.to(device).eval()

    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # Output paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    fp32_onnx_path = out_dir / f"{prefix}_fp32.onnx"
    fp16_onnx_path = out_dir / f"{prefix}_fp16.onnx"

    input_names = ["images"]
    output_names = ["detections"]  # [B, N, 5] (x1, y1, x2, y2, score)
    dynamic_axes = {"images": {0: "batch"}, "detections": {0: "batch"}} if args.dynamic else None

    print(f"[Export] Exporting FP32 ONNX -> {fp32_onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[Export] FP32 ONNX successfully saved ({fp32_onnx_path.stat().st_size / 1e6:.2f} MB)")

    # PyTorch latency benchmark
    with torch.no_grad():
        for _ in range(20):  # warmup
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_iters = 100
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pt_latency = (time.perf_counter() - t0) / n_iters * 1000.0
        pt_fps = 1000.0 / pt_latency

    print(f"[Benchmark] PyTorch ({device.type.upper()}): {pt_latency:.2f} ms ({pt_fps:.1f} FPS)")

    # Test with ONNX Runtime if installed
    try:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(fp32_onnx_path), providers=providers)
        ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}

        # Warmup
        for _ in range(10):
            _ = session.run(None, ort_inputs)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            ort_out = session.run(None, ort_inputs)
        ort_latency = (time.perf_counter() - t0) / n_iters * 1000.0
        ort_fps = 1000.0 / ort_latency
        print(f"[Benchmark] ONNX Runtime ({session.get_providers()[0]}): {ort_latency:.2f} ms ({ort_fps:.1f} FPS)")

        # Verify numerical parity
        with torch.no_grad():
            pt_out = model(dummy_input).cpu().numpy()
        max_diff = np.max(np.abs(pt_out - ort_out[0]))
        print(f"[Validation] Max absolute difference (PyTorch vs ONNX): {max_diff:.6f}")

    except ImportError:
        print("[Export] onnxruntime not installed, skipping ORT benchmark.")

    # FP16 conversion via onnxslim or onnx if available
    try:
        import onnx
        from onnxconverter_common import float16

        model_onnx = onnx.load(str(fp32_onnx_path))
        model_fp16 = float16.convert_float_to_float16(model_onnx)
        onnx.save(model_fp16, str(fp16_onnx_path))
        print(f"[Export] FP16 ONNX successfully saved -> {fp16_onnx_path} ({fp16_onnx_path.stat().st_size / 1e6:.2f} MB)")
    except Exception as e:
        print(f"[Export] FP16 conversion note: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export AvianHeadDetector to ONNX.")
    parser.add_argument("--weights", type=str, default="", help="Path to trained checkpoint (.pth)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Benchmark device")
    parser.add_argument("--prefix", type=str, default="avian_head", help="Output file prefix")
    parser.add_argument("--out-dir", type=str, default="runs/export", help="Output directory")
    args = parser.parse_args()

    export_onnx(args)


if __name__ == "__main__":
    main()
