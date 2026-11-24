#!/usr/bin/env python3
"""Interactive Verification Demo for AvianHeadDetector & Kalman Tracker.

Visualizes overhead CCTV person head detection & Kalman tracking in real-time
to inspect true detections, false alarms (FP), and missed targets (FN).

Features:
- Side-by-side or Toggle Mode: Raw Detector (Yellow) vs Kalman Filtered (Green with Track ID).
- Real-time HUD: Active Count, Total Count, Latency, FPS.
- Supports:
  1. Video file (--source video.mp4)
  2. Live Webcam (--source 0)
  3. Real Validation Sequence Slideshow (--slideshow)
- Hotkeys:
  [SPACE] Pause / Resume
  [T]     Toggle Raw Detections overlay
  [C]     Adjust Confidence Threshold (+0.05 / -0.05)
  [S]     Save snapshot with HUD to disk
  [Q/ESC] Quit demo
"""

import argparse
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torchvision.ops as ops

from bolo.head_detector import AvianHeadDetector
from bolo.infer_video import run_nms


def generate_demo_clip_if_needed(output_path: Path, num_frames: int = 150):
    """Generate a smooth CCTV simulation video from real validation images if no video is supplied."""
    val_dir = ROOT / "data" / "head_dataset" / "images" / "val"
    val_imgs = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))
    if not val_imgs:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    print(f"[Demo] Generating sample sequence clip from {len(val_imgs)} real CCTV images...")
    # Select a dense CCTV classroom/booth scene
    sample_img = cv2.imread(str(val_imgs[0]))
    h, w = sample_img.shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))

    # Create panning/zooming CCTV realistic sequence across 5 scenes
    for scene_idx in range(min(5, len(val_imgs))):
        base_frame = cv2.imread(str(val_imgs[scene_idx]))
        for f in range(30):
            # Gentle camera jitter / slow pan simulating CCTV pole sway
            dx = int(math.sin(f * 0.2) * 4)
            dy = int(math.cos(f * 0.2) * 3)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame_jitter = cv2.warpAffine(base_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            writer.write(frame_jitter)

    writer.release()
    print(f"[Demo] Sample demo clip saved to {output_path}")
    return output_path


def main():
    import math

    parser = argparse.ArgumentParser(description="AvianHeadDetector Visual Verification Demo")
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Path to video file (.mp4) or webcam index (0). If empty, runs real validation sequence.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "runs/head_detector/avian_head_finetune_night/best.pth"),
        help="Path to trained checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-video", type=str, default="", help="Path to save annotated video output")
    parser.add_argument("--min-hits", type=int, default=3, help="Consecutive detections required to confirm person")
    args = parser.parse_args()

    # Fallback to 1st run best.pth if night finetune is not found
    weights_path = Path(args.weights)
    if not weights_path.exists():
        fallback = ROOT / "runs/head_detector/avian_head_scut_real/best.pth"
        if fallback.exists():
            weights_path = fallback

    print(f"[Demo] Loading AvianHeadDetector with weights: {weights_path}")
    device = torch.device(args.device)
    model = AvianHeadDetector(num_classes=1).to(device)
    model.tracker.min_hits = args.min_hits

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in (ckpt.get("ema") or ckpt.get("model") or ckpt).items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Determine input source
    is_slideshow = False
    slideshow_files = []
    slideshow_idx = 0

    if not args.source:
        # Prioritize real CCTV footage from open dataset
        cctv_sample = ROOT / "data" / "cctv_samples" / "cctv_default_demo.mp4"
        demo_video = ROOT / "runs" / "demo_sample_cctv.mp4"
        if cctv_sample.exists():
            source = str(cctv_sample)
            print(f"[Demo] Using default open CCTV video footage: {cctv_sample}")
        elif demo_video.exists():
            source = str(demo_video)
            print(f"[Demo] Using synthetic CCTV sequence: {demo_video}")
        else:
            generate_demo_clip_if_needed(demo_video)
            source = str(demo_video) if demo_video.exists() else "0"
    else:
        source = args.source

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif Path(source).exists():
        cap = cv2.VideoCapture(source)
    else:
        # Fallback to validation image directory
        is_slideshow = True
        val_dir = ROOT / "data" / "head_dataset" / "images" / "val"
        slideshow_files = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")))
        print(f"[Demo] Source video not found. Playing {len(slideshow_files)} real CCTV validation images.")

    writer = None
    show_raw_detections = True
    paused = False
    conf_thresh = args.conf

    print("\n" + "=" * 65)
    print(" AvianHeadDetector Real-Time Visual Inspection Demo")
    print("=" * 65)
    print(" Controls:")
    print("  [SPACE]  : Pause / Resume playback")
    print("  [T]      : Toggle Raw Detection proposals (Yellow)")
    print("  [UP/DOWN]: Adjust confidence threshold (+/- 0.05)")
    print("  [S]      : Save snapshot to disk (demo_snapshot.jpg)")
    print("  [Q/ESC]  : Quit")
    print("=" * 65 + "\n")

    fps_buffer = []

    while True:
        if not paused:
            if not is_slideshow:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    ret, frame = cap.read()
                    if not ret:
                        break
            else:
                frame = cv2.imread(str(slideshow_files[slideshow_idx]))
                slideshow_idx = (slideshow_idx + 1) % len(slideshow_files)
                time.sleep(0.08)  # ~12 FPS slideshow pace
        else:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                paused = not paused
            elif key == ord("q") or key == 27:
                break
            continue

        orig_h, orig_w = frame.shape[:2]

        # Preprocess frame
        img_resized = cv2.resize(frame, (args.imgsz, args.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        # Inference: Raw detections vs Kalman tracked
        t0 = time.time()
        with torch.no_grad():
            proposals = model(tensor, export_onnx=True)  # [1, N, 5]
            raw_dets = run_nms(proposals[0], conf_thresh=conf_thresh, iou_thresh=args.iou)
            # Kalman multi-object tracker update
            tracks = model.tracker.update(raw_dets)  # [M, 6] (x1, y1, x2, y2, id, score)
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000.0
        fps_buffer.append(1.0 / max(t1 - t0, 1e-5))
        if len(fps_buffer) > 20:
            fps_buffer.pop(0)
        current_fps = float(np.mean(fps_buffer))

        scale_x = orig_w / float(args.imgsz)
        scale_y = orig_h / float(args.imgsz)

        # Draw raw detection candidates (Yellow dotted/thin) for comparison
        if show_raw_detections:
            for det in raw_dets:
                rx1, ry1 = int(det[0] * scale_x), int(det[1] * scale_y)
                rx2, ry2 = int(det[2] * scale_x), int(det[3] * scale_y)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 220, 255), 1)

        # Draw Kalman confirmed tracks (Bright green box with ID & velocity trace)
        for trk in tracks:
            x1, y1 = int(trk[0] * scale_x), int(trk[1] * scale_y)
            x2, y2 = int(trk[2] * scale_x), int(trk[3] * scale_y)
            trk_id = int(trk[4])
            score = trk[5]

            # High-visibility bounding box
            color = (0, 255, 64)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Center target point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            # Label banner
            label = f"ID #{trk_id} ({score:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(y1 - 18, 0)), (x1 + lw + 4, max(y1, 18)), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, max(y1 - 4, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # HUD Top Banner
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (orig_w, 64), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

        active_count = len(tracks)
        raw_count = len(raw_dets)
        total_unique = model.tracker.total_unique_count

        cv2.putText(
            frame,
            f"AvianHead CCTV Tracker | FPS: {current_fps:.1f} ({latency_ms:.1f}ms) | Conf: {conf_thresh:.2f}",
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Active Heads: {active_count} | Raw Proposals: {raw_count} | Total Unique: {total_unique} | [T] Raw BBox: {'ON' if show_raw_detections else 'OFF'}",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )

        # Output video writer init
        if args.save_video and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (orig_w, orig_h))

        if writer:
            writer.write(frame)

        cv2.imshow("AvianHead - CCTV Visual Inspection Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("t") or key == ord("T"):
            show_raw_detections = not show_raw_detections
            print(f"[Demo] Raw detection proposals overlay: {'ON' if show_raw_detections else 'OFF'}")
        elif key == ord("s") or key == ord("S"):
            cv2.imwrite("demo_snapshot.jpg", frame)
            print("[Demo] Snapshot saved to demo_snapshot.jpg")
        elif key == 82 or key == ord("+"):  # Up arrow
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"[Demo] Confidence threshold set to: {conf_thresh:.2f}")
        elif key == 84 or key == ord("-"):  # Down arrow
            conf_thresh = max(0.10, conf_thresh - 0.05)
            print(f"[Demo] Confidence threshold set to: {conf_thresh:.2f}")

    if not is_slideshow:
        cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[Demo] Visual inspection ended.")


if __name__ == "__main__":
    main()
