"""Real-time overhead video & camera inference for AvianHeadDetector.

Features:
- NMS (Non-Maximum Suppression) box filtering
- Real-time FPS monitoring overlay
- Video file input or live webcam capture
- Video output writing (.mp4 / .avi)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torchvision.ops as ops

from bolo.head_detector import AvianHeadDetector


def run_nms(predictions: torch.Tensor, conf_thresh: float = 0.3, iou_thresh: float = 0.45) -> np.ndarray:
    """Apply NMS to predictions [N, 5] (x1, y1, x2, y2, score)."""
    boxes = predictions[:, :4]
    scores = predictions[:, 4]

    mask = scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    keep_idx = ops.nms(boxes, scores, iou_thresh)
    keep_boxes = boxes[keep_idx].cpu().numpy()
    keep_scores = scores[keep_idx].cpu().numpy()

    return np.column_stack([keep_boxes, keep_scores])


def infer_video(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    # Initialize model
    model = AvianHeadDetector(num_classes=1)
    if args.weights and Path(args.weights).exists():
        print(f"[Inference] Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    model.to(device).eval()

    # Video Capture
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Inference] Error: Could not open video source {args.source}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps_input, (orig_w, orig_h))
        print(f"[Inference] Writing output to {out_p}")

    scale_x = orig_w / float(args.imgsz)
    scale_y = orig_h / float(args.imgsz)

    frame_count = 0
    t_start = time.time()

    print("[Inference] Starting detection loop (press 'q' to stop if running GUI)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.perf_counter()

        # Preprocessing
        resized = cv2.resize(frame, (args.imgsz, args.imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        # Forward
        with torch.no_grad():
            preds = model(tensor)[0]  # [N_proposals, 5]

        # NMS
        detections = run_nms(preds, conf_thresh=args.conf, iou_thresh=args.iou)
        inference_time = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(inference_time, 1e-5)

        # Draw bboxes
        for det in detections:
            x1 = int(det[0] * scale_x)
            y1 = int(det[1] * scale_y)
            x2 = int(det[2] * scale_x)
            y2 = int(det[3] * scale_y)
            score = det[4]

            # Bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 64), 2)
            label = f"Head: {score:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 64), 1)

        # HUD overlay
        cv2.putText(
            frame,
            f"AvianHead FPS: {fps:.1f} | Latency: {inference_time:.1f}ms | Heads: {len(detections)}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if writer:
            writer.write(frame)

        if args.show:
            cv2.imshow("AvianHead Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-5)
    print(f"[Inference Done] Processed {frame_count} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Run head detection on video/webcam.")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index")
    parser.add_argument("--weights", type=str, default="runs/head_detector/avian_head_run/best.pth", help="Checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--output", type=str, default="", help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Display video stream window")
    args = parser.parse_args()

    infer_video(args)


if __name__ == "__main__":
    main()
