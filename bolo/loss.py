"""Loss calculation for Anchor-free Head Detection.

Combines BCEWithLogits for classification and CIoU for bounding box regression.
Assigns positive anchor points based on spatial center containment across P3, P4, P5.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate Complete IoU (CIoU) in full FP32 to prevent FP16 overflow."""
    box1 = box1.float()
    box2 = box2.float()
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    w1, h1 = (b1_x2 - b1_x1).clamp(min=0.0), (b1_y2 - b1_y1).clamp(min=0.0)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=0.0), (b2_y2 - b2_y1).clamp(min=0.0)
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    iou = inter_area / union_area

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Center distance
    b1_cx = (b1_x1 + b1_x2) / 2.0
    b1_cy = (b1_y1 + b1_y2) / 2.0
    b2_cx = (b2_x1 + b2_x2) / 2.0
    b2_cy = (b2_y1 + b2_y2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio consistency
    v = (4.0 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou.clamp(min=-1.0, max=1.0)


class HeadDetectionLoss(nn.Module):
    """Anchor-Free Detection Loss with Center Assignment."""

    def __init__(self, strides: List[int] = [8, 16, 32], cls_weight: float = 1.0, box_weight: float = 2.5):
        super().__init__()
        self.strides = strides
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        preds: List[Tuple[torch.Tensor, torch.Tensor]],
        targets: List[torch.Tensor],
        img_size: Tuple[int, int] = (640, 640),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss in FP32 precision to prevent numerical overflow."""
        # Convert all prediction tensors to FP32
        preds = [(c.float(), r.float().clamp(max=float(max(img_size)))) for c, r in preds]
        targets = [t.float() for t in targets]

        device = preds[0][0].device
        batch_size = preds[0][0].shape[0]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_pos_count = 0

        for b in range(batch_size):
            gt = targets[b]  # [N_gt, 5]
            has_gt = len(gt) > 0

            for lvl, (cls_logits, reg_dist) in enumerate(preds):
                stride = self.strides[lvl]
                H, W = cls_logits.shape[-2:]

                yv, xv = torch.meshgrid(
                    torch.arange(H, device=device, dtype=cls_logits.dtype),
                    torch.arange(W, device=device, dtype=cls_logits.dtype),
                    indexing="ij",
                )
                grid_cx = (xv + 0.5) * stride
                grid_cy = (yv + 0.5) * stride

                pred_cls = cls_logits[b, 0]  # [H, W]
                pred_reg = reg_dist[b]       # [4, H, W]

                cls_target = torch.zeros_like(pred_cls)

                if has_gt:
                    gt_boxes = gt[:, 1:5]  # [N_gt, 4] (x1, y1, x2, y2)
                    gx1, gy1, gx2, gy2 = gt_boxes.unbind(-1)

                    # Center assignment: grid center falls within ground truth box
                    # expand dims: grid_cx [H, W, 1], gx1 [1, 1, N_gt]
                    in_l = grid_cx.unsqueeze(-1) - gx1.view(1, 1, -1)
                    in_t = grid_cy.unsqueeze(-1) - gy1.view(1, 1, -1)
                    in_r = gx2.view(1, 1, -1) - grid_cx.unsqueeze(-1)
                    in_b = gy2.view(1, 1, -1) - grid_cy.unsqueeze(-1)

                    is_in_box = (in_l > 0) & (in_t > 0) & (in_r > 0) & (in_b > 0)
                    pos_mask = is_in_box.any(dim=-1)  # [H, W]

                    cls_target[pos_mask] = 1.0

                    if pos_mask.sum() > 0:
                        # Match each positive grid point to closest / smallest ground-truth box
                        areas = (gx2 - gx1) * (gy2 - gy1)
                        best_gt_idx = is_in_box.float().argmax(dim=-1)[pos_mask]
                        matched_gt = gt_boxes[best_gt_idx]

                        # Predicted decoded box for positive cells
                        pos_reg = pred_reg[:, pos_mask] * stride  # [4, N_pos]
                        p_cx = grid_cx[pos_mask]
                        p_cy = grid_cy[pos_mask]

                        px1 = (p_cx - pos_reg[0]).clamp(min=0.0, max=float(img_size[1]))
                        py1 = (p_cy - pos_reg[1]).clamp(min=0.0, max=float(img_size[0]))
                        px2 = (p_cx + pos_reg[2]).clamp(min=0.0, max=float(img_size[1]))
                        py2 = (p_cy + pos_reg[3]).clamp(min=0.0, max=float(img_size[0]))

                        pred_boxes = torch.stack([px1, py1, px2, py2], dim=-1)
                        ciou = bbox_ciou(pred_boxes, matched_gt)
                        box_loss = (1.0 - ciou).sum()

                        total_box_loss = total_box_loss + box_loss
                        total_pos_count += pos_mask.sum().item()

                # Calibrated alpha-balanced Focal BCE Loss
                bce = self.bce(pred_cls, cls_target)
                p = torch.sigmoid(pred_cls)
                pos_mask_target = (cls_target == 1.0)
                neg_mask_target = ~pos_mask_target

                pos_loss = (bce[pos_mask_target] * (1.0 - p[pos_mask_target]).pow(2.0)).sum()
                neg_loss = (bce[neg_mask_target] * p[neg_mask_target].pow(2.0)).sum()

                # Alpha weighting to balance sparse head positive signals vs large background
                alpha = 0.75
                total_cls_loss = total_cls_loss + (alpha * pos_loss + (1.0 - alpha) * neg_loss)

        num_pos = max(total_pos_count, 1)
        final_cls_loss = self.cls_weight * (total_cls_loss / num_pos)
        final_box_loss = self.box_weight * (total_box_loss / num_pos)
        total_loss = final_cls_loss + final_box_loss

        return total_loss, final_cls_loss, final_box_loss
