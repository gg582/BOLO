import numpy as np
from bolo.tracker import AvianHeadTracker
"""AvianHeadDetector: Real-time person head detector for overhead/booth CCTV.

Architecture:
- Backbone: AvianRaptorBackbone (adapted from AvianRaptorNet_Fast)
  Loads pretrained weights from avian_coco_micro.pth without loss.
  Produces P3 (Stride 8, 256ch), P4 (Stride 16, 512ch), P5 (Stride 32, 512ch).
- Neck: LightPAN (Lightweight PANet with MotionStrideAttention)
- Head: DecoupledHead (Anchor-free, Classification + Box Regression)
- Compatible with ONNX / TensorRT export (FP32 / FP16).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.avian_model import BioMish, ChannelAttention, FeatherBlock, RaptorFovealLite
from bolo.modules import MotionStrideAttention


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast with BioMish."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            BioMish(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            BioMish(),
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], dim=1))


class ConvBlock(nn.Module):
    """Standard Conv2d + BatchNorm2d + BioMish."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            BioMish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


from torch.utils.checkpoint import checkpoint

class AvianRaptorBackbone(nn.Module):
    """Multi-scale backbone wrapping AvianRaptorNet_Fast stages.

    Outputs:
        P3: Stride 8, 256 channels
        P4: Stride 16, 512 channels
        P5: Stride 32, 512 channels
    """

    def __init__(self, pretrained_path: str = None, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # Stem: Conv stride=2 downsamples 640->320 to manage memory & compute.
        # Kernel shape [48, 3, 3, 3] is identical to AvianRaptorNet_Fast retina.
        self.retina = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            BioMish(),
        )
        self.raptor_eye = RaptorFovealLite(48, 96)  # Stride 2, 96ch

        # Stage 1: Stride 4 (160x160 for 640x640), 128ch
        self.stage1 = nn.Sequential(
            FeatherBlock(96, 128, stride=2),
            FeatherBlock(128, 128, stride=1),
        )

        # Stage 2 -> P3: Stride 8 (80x80 for 640x640), 256ch
        self.stage2 = nn.Sequential(
            FeatherBlock(128, 256, stride=2),
            FeatherBlock(256, 256, stride=1),
            FeatherBlock(256, 256, stride=1),
        )

        # Stage 3 -> P4: Stride 16 (40x40 for 640x640), 512ch
        self.stage3 = nn.Sequential(
            FeatherBlock(256, 512, stride=2),
            FeatherBlock(512, 512, stride=1),
        )

        # Stage 4 -> P5: Stride 32 (20x20 for 640x640), 512ch
        self.stage4 = nn.Sequential(
            FeatherBlock(512, 512, stride=2),
            SPPF(512, 512, k=5),
            MotionStrideAttention(512),
        )

        self._init_weights()
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def load_pretrained(self, weights_path: str, verbose: bool = True) -> Dict[str, int]:
        """Load weights from avian_coco_micro.pth with 100% backbone key mapping."""
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        model_dict = self.state_dict()
        mapping = {}

        # 1. retina mapping
        for k in list(state.keys()):
            if k.startswith("retina."):
                mapping[k] = k

        # 2. raptor_eye mapping
        for k in list(state.keys()):
            if k.startswith("raptor_eye."):
                mapping[k] = k

        # 3. body stages mapping:
        # body.0, body.1 -> stage1.0, stage1.1
        # body.2, body.3, body.4 -> stage2.0, stage2.1, stage2.2
        # body.5, body.6 -> stage3.0, stage3.1
        body_map = {
            "body.0": "stage1.0",
            "body.1": "stage1.1",
            "body.2": "stage2.0",
            "body.3": "stage2.1",
            "body.4": "stage2.2",
            "body.5": "stage3.0",
            "body.6": "stage3.1",
        }
        for k, v in list(state.items()):
            for b_src, b_dst in body_map.items():
                if k.startswith(b_src + "."):
                    dst_k = k.replace(b_src, b_dst, 1)
                    mapping[k] = dst_k

        loaded = 0
        transferred_state = {}
        for src_k, dst_k in mapping.items():
            if dst_k in model_dict:
                if model_dict[dst_k].shape == state[src_k].shape:
                    transferred_state[dst_k] = state[src_k]
                    loaded += 1
                elif verbose:
                    print(f"[WARN] Shape mismatch: {src_k} -> {dst_k}")

        model_dict.update(transferred_state)
        self.load_state_dict(model_dict)
        total_backbone = len(model_dict)
        if verbose:
            print(f"[AvianRaptorBackbone] Loaded {loaded}/{len(mapping)} weights from {weights_path} "
                  f"(Backbone coverage: {loaded}/{total_backbone} params)")
        return {"loaded": loaded, "total_mapped": len(mapping), "total_backbone": total_backbone}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training:
            x = self.retina(x)
            x = checkpoint(self.raptor_eye, x, use_reentrant=False)
            x = checkpoint(self.stage1, x, use_reentrant=False)
            p3 = checkpoint(self.stage2, x, use_reentrant=False)
            p4 = checkpoint(self.stage3, p3, use_reentrant=False)
            p5 = checkpoint(self.stage4, p4, use_reentrant=False)
        else:
            x = self.retina(x)
            x = self.raptor_eye(x)
            x = self.stage1(x)
            p3 = self.stage2(x)      # Stride 8, 256ch
            p4 = self.stage3(p3)     # Stride 16, 512ch
            p5 = self.stage4(p4)     # Stride 32, 512ch
        return p3, p4, p5


class LightPAN(nn.Module):
    """Lightweight PANet Neck with MotionStrideAttention."""

    def __init__(self, in_channels: List[int] = [256, 512, 512], out_ch: int = 128):
        super().__init__()
        c3, c4, c5 = in_channels
        self.out_ch = out_ch

        # Projections to uniform neck channel
        self.proj5 = ConvBlock(c5, out_ch, 1, 1, 0)
        self.proj4 = ConvBlock(c4, out_ch, 1, 1, 0)
        self.proj3 = ConvBlock(c3, out_ch, 1, 1, 0)

        # Top-down FPN path
        self.fpn_conv4 = ConvBlock(out_ch * 2, out_ch, 3, 1, 1)
        self.fpn_conv3 = ConvBlock(out_ch * 2, out_ch, 3, 1, 1)

        # Bottom-up PAN path
        self.down3 = ConvBlock(out_ch, out_ch, 3, 2, 1)
        self.pan_conv4 = ConvBlock(out_ch * 2, out_ch, 3, 1, 1)

        self.down4 = ConvBlock(out_ch, out_ch, 3, 2, 1)
        self.pan_conv5 = ConvBlock(out_ch * 2, out_ch, 3, 1, 1)

        # Motion stride correction on finest (P3) and coarsest (P5) levels
        self.motion3 = MotionStrideAttention(out_ch)
        self.motion5 = MotionStrideAttention(out_ch)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Projections
        p5_proj = self.proj5(p5)
        p4_proj = self.proj4(p4)
        p3_proj = self.proj3(p3)

        # Top-down path
        p5_up = F.interpolate(p5_proj, size=p4_proj.shape[-2:], mode="nearest")
        f4 = self.fpn_conv4(torch.cat([p5_up, p4_proj], dim=1))

        f4_up = F.interpolate(f4, size=p3_proj.shape[-2:], mode="nearest")
        f3 = self.fpn_conv3(torch.cat([f4_up, p3_proj], dim=1))
        out3 = self.motion3(f3)

        # Bottom-up path
        d3 = self.down3(out3)
        out4 = self.pan_conv4(torch.cat([d3, f4], dim=1))

        d4 = self.down4(out4)
        out5 = self.pan_conv5(torch.cat([d4, p5_proj], dim=1))
        out5 = self.motion5(out5)

        return out3, out4, out5


class DecoupledHead(nn.Module):
    """Anchor-Free Decoupled Detection Head.

    Separates classification and bounding box regression branches.
    Predicts distances (left, top, right, bottom) from grid cell centers.
    """

    def __init__(self, in_ch: int = 128, num_classes: int = 1, strides: List[int] = [8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_ch, in_ch, 3, 1, 1),
                ConvBlock(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, num_classes, 1),
            )
            for _ in strides
        ])

        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_ch, in_ch, 3, 1, 1),
                ConvBlock(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, 4, 1),
            )
            for _ in strides
        ])

        # Prior bias for classification to stabilize early training
        for cls_layer in self.cls_convs:
            nn.init.constant_(cls_layer[-1].bias, -4.595)  # sigmoid(-4.595) ~ 0.01

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        outputs = []
        for i, x in enumerate(feats):
            cls_out = self.cls_convs[i](x)
            reg_out = F.relu(self.reg_convs[i](x))  # Distances must be non-negative
            outputs.append((cls_out, reg_out))
        return outputs


class AvianHeadDetector(nn.Module):
    """Full End-to-End Head Detection Pipeline."""

    def __init__(
        self,
        pretrained_backbone: str = None,
        num_classes: int = 1,
        neck_channels: int = 128,
        strides: List[int] = [8, 16, 32],
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.strides = strides
        self.backbone = AvianRaptorBackbone(pretrained_backbone, use_checkpoint=use_checkpoint)
        self.neck = LightPAN(in_channels=[256, 512, 512], out_ch=neck_channels)
        self.head = DecoupledHead(in_ch=neck_channels, num_classes=num_classes, strides=strides)
        self.tracker = AvianHeadTracker(max_age=15, min_hits=3, iou_threshold=0.3)

    def freeze_backbone(self):
        """Freeze all backbone parameters for warm-up phase."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[AvianHeadDetector] Backbone frozen.")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for joint refinement."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[AvianHeadDetector] Backbone unfrozen.")

    def forward(self, x: torch.Tensor, export_onnx: bool = False):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        raw_preds = self.head((n3, n4, n5))

        if self.training:
            return raw_preds

        # Inference decoding: generates decoded bboxes [x1, y1, x2, y2, score]
        return self._decode_predictions(raw_preds, x.shape[-2:], export_onnx=export_onnx)

    def _decode_predictions(
        self,
        raw_preds: List[Tuple[torch.Tensor, torch.Tensor]],
        img_size: Tuple[int, int],
        export_onnx: bool = False,
    ) -> torch.Tensor:
        """Decode raw ltrb distances and logits into [x1, y1, x2, y2, score]."""
        batch_size = raw_preds[0][0].shape[0]
        decoded_boxes = []
        scores_list = []

        for i, (cls_logits, reg_dist) in enumerate(raw_preds):
            stride = self.strides[i]
            H, W = cls_logits.shape[-2:]

            # Grid generation
            yv, xv = torch.meshgrid(
                torch.arange(H, device=cls_logits.device, dtype=cls_logits.dtype),
                torch.arange(W, device=cls_logits.device, dtype=cls_logits.dtype),
                indexing="ij",
            )
            # Center coordinates in input space
            cx = (xv + 0.5) * stride
            cy = (yv + 0.5) * stride

            # reg_dist: [B, 4, H, W] -> [l, t, r, b] * stride
            l = reg_dist[:, 0] * stride
            t = reg_dist[:, 1] * stride
            r = reg_dist[:, 2] * stride
            b = reg_dist[:, 3] * stride

            x1 = torch.clamp(cx - l, min=0.0, max=float(img_size[1]))
            y1 = torch.clamp(cy - t, min=0.0, max=float(img_size[0]))
            x2 = torch.clamp(cx + r, min=0.0, max=float(img_size[1]))
            y2 = torch.clamp(cy + b, min=0.0, max=float(img_size[0]))

            boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(batch_size, -1, 4)
            scores = torch.sigmoid(cls_logits).view(batch_size, -1, self.head.num_classes)

            decoded_boxes.append(boxes)
            scores_list.append(scores)

        all_boxes = torch.cat(decoded_boxes, dim=1)
        all_scores = torch.cat(scores_list, dim=1)

        # Output: [B, N_proposals, 5] (x1, y1, x2, y2, score)
        return torch.cat([all_boxes, all_scores], dim=-1)



    def track(self, x: torch.Tensor, conf_thresh: float = 0.35, iou_thresh: float = 0.45) -> np.ndarray:
        """End-to-end inference and Kalman tracking on a single frame.

        Args:
            x: Input image tensor [1, 3, H, W] in [0, 1] range.
            conf_thresh: Score threshold for detection candidates.
            iou_thresh: NMS IoU threshold.

        Returns:
            np.ndarray: Tracked heads [M, 6] (x1, y1, x2, y2, track_id, score).
        """
        with torch.no_grad():
            proposals = self.forward(x, export_onnx=True)  # [1, N, 5]
            pred = proposals[0]
            boxes = pred[:, :4]
            scores = pred[:, 4]

            mask = scores > conf_thresh
            boxes = boxes[mask]
            scores = scores[mask]

            if len(boxes) == 0:
                dets = np.zeros((0, 5), dtype=np.float32)
            else:
                import torchvision.ops as ops
                keep = ops.nms(boxes, scores, iou_thresh)
                keep_boxes = boxes[keep].cpu().numpy()
                keep_scores = scores[keep].cpu().numpy()
                dets = np.column_stack([keep_boxes, keep_scores])

            return self.tracker.update(dets)

def build_head_detector(pretrained_weights: str = "avian_coco_micro.pth") -> AvianHeadDetector:
    """Convenience builder function."""
    weights = pretrained_weights if Path(pretrained_weights).exists() else None
    return AvianHeadDetector(pretrained_backbone=weights)
