"""
Detection Loss for object detection training.

This module implements a comprehensive loss function combining:
- Objectness loss: Binary cross-entropy for object presence
- Bounding box loss: GIoU/MSE for box coordinates
- Classification loss: Cross-entropy/Focal loss for class prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU (Intersection over Union) between two sets of boxes.

    Args:
        boxes1: Boxes in format [N, 4] (x, y, w, h) - center format
        boxes2: Boxes in format [M, 4] (x, y, w, h) - center format

    Returns:
        IoU matrix of shape [N, M]
    """
    # Convert from (x_center, y_center, w, h) to (x1, y1, x2, y2)
    boxes1_x1y1 = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_x2y2 = boxes1[..., :2] + boxes1[..., 2:] / 2

    boxes2_x1y1 = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_x2y2 = boxes2[..., :2] + boxes2[..., 2:] / 2

    # Calculate intersection area
    inter_x1y1 = torch.max(boxes1_x1y1.unsqueeze(1), boxes2_x1y1.unsqueeze(0))
    inter_x2y2 = torch.min(boxes1_x2y2.unsqueeze(1), boxes2_x2y2.unsqueeze(0))
    inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Calculate union area
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    union_area = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Generalized IoU (GIoU) between boxes.

    GIoU improves upon IoU by considering the smallest enclosing box,
    providing better gradients when boxes don't overlap.

    Args:
        boxes1: Boxes in format [N, 4] (x, y, w, h)
        boxes2: Boxes in format [N, 4] (x, y, w, h)

    Returns:
        GIoU values of shape [N]
    """
    # Convert to corner format
    boxes1_x1y1 = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_x2y2 = boxes1[..., :2] + boxes1[..., 2:] / 2

    boxes2_x1y1 = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_x2y2 = boxes2[..., :2] + boxes2[..., 2:] / 2

    # Calculate intersection
    inter_x1y1 = torch.max(boxes1_x1y1, boxes2_x1y1)
    inter_x2y2 = torch.min(boxes1_x2y2, boxes2_x2y2)
    inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Calculate union
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)

    # Calculate smallest enclosing box
    enclose_x1y1 = torch.min(boxes1_x1y1, boxes2_x1y1)
    enclose_x2y2 = torch.max(boxes1_x2y2, boxes2_x2y2)
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    # Calculate GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    return giou


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Gunakan raw logits agar tidak terjadi double sigmoid
        p = torch.sigmoid(inputs_logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs_logits, targets, reduction='none'
        )

        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss.mean()


class DetectionLoss(nn.Module):
    """
    Complete detection loss combining objectness, bbox, and classification losses.

    The loss assigns anchors to ground truth boxes based on IoU, then computes:
    1. Objectness loss: BCE for predicting if anchor contains object
    2. Bounding box loss: GIoU or MSE for box regression
    3. Classification loss: BCE or Focal loss for class prediction

    Args:
        num_classes: Number of object classes
        lambda_obj: Weight for objectness loss (positive anchors)
        lambda_noobj: Weight for no-object loss (negative anchors)
        lambda_bbox: Weight for bounding box loss
        lambda_class: Weight for classification loss
        iou_threshold_pos: IoU threshold for positive anchor assignment
        iou_threshold_neg: IoU threshold for negative anchor assignment
        bbox_loss_type: Type of bbox loss ('giou' or 'mse')
        use_focal_loss: Use focal loss for classification
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
    """

    def __init__(
        self,
        num_classes: int = 80,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_bbox: float = 5.0,
        lambda_class: float = 1.0,
        iou_threshold_pos: float = 0.5,
        iou_threshold_neg: float = 0.4,
        bbox_loss_type: str = 'giou',
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg
        self.bbox_loss_type = bbox_loss_type
        self.use_focal_loss = use_focal_loss

        # Loss functions
        self.bce_loss = nn.BCELoss(reduction='none')
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def assign_anchors_to_targets(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign anchors to ground truth targets based on IoU.

        Args:
            pred_boxes: Predicted boxes [N, 4]
            target_boxes: Ground truth boxes [M, 4]
            target_labels: Ground truth labels [M]

        Returns:
            Tuple of (objectness_target, bbox_target, class_target)
            - objectness_target: [N] 1 for positive, 0 for negative, -1 for ignore
            - bbox_target: [N, 4] target boxes for positive anchors
            - class_target: [N] class labels for positive anchors
        """
        N = pred_boxes.shape[0]
        M = target_boxes.shape[0]

        if M == 0:
            # No targets in this image
            return (
                torch.zeros(N, device=pred_boxes.device),
                torch.zeros(N, 4, device=pred_boxes.device),
                torch.zeros(N, dtype=torch.long, device=pred_boxes.device)
            )

        # Calculate IoU between all predictions and targets
        iou_matrix = box_iou(pred_boxes, target_boxes)  # [N, M]

        # Get best IoU for each anchor
        max_iou, max_iou_idx = iou_matrix.max(dim=1)  # [N]

        # Initialize targets
        objectness_target = torch.zeros(N, device=pred_boxes.device)
        bbox_target = torch.zeros(N, 4, device=pred_boxes.device)
        class_target = torch.zeros(N, dtype=torch.long, device=pred_boxes.device)

        # Positive anchors: IoU > threshold_pos
        pos_mask = max_iou >= self.iou_threshold_pos
        objectness_target[pos_mask] = 1.0
        bbox_target[pos_mask] = target_boxes[max_iou_idx[pos_mask]]
        class_target[pos_mask] = target_labels[max_iou_idx[pos_mask]]

        # Negative anchors: IoU < threshold_neg
        # (Between thresholds are ignored in loss computation)

        return objectness_target, bbox_target, class_target, pos_mask

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = predictions['objectness'].shape[0]
        device = predictions['objectness'].device

        # PERBAIKAN 1: Ekstrak RAW LOGITS untuk stabilitas kalkulasi (mencegah error AMP)
        pred_raw = predictions['predictions']  # [B, A, H, W, 5+C]
        pred_boxes = predictions['boxes']      # [B, A, H, W, 4] (Kotak yang sudah di-decode untuk hitung IoU)

        pred_obj_logits = pred_raw[..., 0]     
        pred_class_logits = pred_raw[..., 5:]  

        # PERBAIKAN 2: Inisialisasi sebagai Tensor sejak awal agar mencegah error "has no attribute 'item'"
        total_obj_loss = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_class_loss = torch.tensor(0.0, device=device)
        num_positive_anchors = 0

        for b in range(batch_size):
            obj_logits_b = pred_obj_logits[b].reshape(-1) 
            boxes_b = pred_boxes[b].reshape(-1, 4)  
            classes_logits_b = pred_class_logits[b].reshape(-1, self.num_classes) 

            target_boxes_b = targets['boxes'][b].to(device)  
            target_labels_b = targets['labels'][b].to(device)  

            obj_target, bbox_target, class_target, pos_mask = \
                self.assign_anchors_to_targets(boxes_b, target_boxes_b, target_labels_b)

            # PERBAIKAN 3: Harus pakai reduction='none' agar mask bisa dikalikan secara akurat (elemen per elemen)
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logits_b, obj_target, reduction='none'
            )

            pos_obj_loss = obj_loss * pos_mask.float() * self.lambda_obj
            neg_obj_loss = obj_loss * (~pos_mask).float() * self.lambda_noobj
            
            # Rata-rata per jumlah total anchor agar skalanya stabil (~0.5)
            num_anchors = obj_logits_b.shape[0]
            total_obj_loss += (pos_obj_loss.sum() + neg_obj_loss.sum()) / num_anchors

            if pos_mask.sum() > 0:
                num_pos = pos_mask.sum().item()
                num_positive_anchors += num_pos

                if self.bbox_loss_type == 'giou':
                    giou = generalized_box_iou(
                        boxes_b[pos_mask],
                        bbox_target[pos_mask]
                    )
                    bbox_loss = (1 - giou).sum()
                else:  
                    bbox_loss = F.mse_loss(
                        boxes_b[pos_mask],
                        bbox_target[pos_mask],
                        reduction='sum'
                    )
                total_bbox_loss += bbox_loss

                class_target_onehot = F.one_hot(
                    class_target[pos_mask],
                    num_classes=self.num_classes
                ).float()

                if self.use_focal_loss:
                    class_loss = self.focal_loss(
                        classes_logits_b[pos_mask],
                        class_target_onehot
                    ) * num_pos
                else:
                    class_loss = F.binary_cross_entropy_with_logits(
                        classes_logits_b[pos_mask],
                        class_target_onehot,
                        reduction='sum'
                    )
                total_class_loss += class_loss

        num_positive_anchors = max(num_positive_anchors, 1)

        obj_loss_avg = total_obj_loss / batch_size
        bbox_loss_avg = total_bbox_loss / num_positive_anchors * self.lambda_bbox
        class_loss_avg = total_class_loss / num_positive_anchors * self.lambda_class

        total_loss = obj_loss_avg + bbox_loss_avg + class_loss_avg

        return {
            'total_loss': total_loss,
            'obj_loss': obj_loss_avg,
            'bbox_loss': bbox_loss_avg,
            'class_loss': class_loss_avg,
            'num_pos': torch.tensor(num_positive_anchors, dtype=torch.float32, device=device)
        }


def test_loss():
    """Test function to verify loss computation."""
    print("Testing DetectionLoss...")

    # Create loss function
    loss_fn = DetectionLoss(
        num_classes=80,
        bbox_loss_type='giou',
        use_focal_loss=True
    )

    # Create dummy predictions
    batch_size = 2
    predictions = {
        'predictions': torch.randn(batch_size, 3, 20, 20, 85),
        'objectness': torch.rand(batch_size, 3, 20, 20, 1),
        'boxes': torch.rand(batch_size, 3, 20, 20, 4) * 320,
        'class_scores': torch.rand(batch_size, 3, 20, 20, 80)
    }

    # Create dummy targets
    targets = {
        'boxes': [
            torch.tensor([[100.0, 100.0, 50.0, 50.0], [200.0, 200.0, 80.0, 80.0]]),
            torch.tensor([[150.0, 150.0, 60.0, 60.0]])
        ],
        'labels': [
            torch.tensor([0, 15]),
            torch.tensor([42])
        ]
    }

    # Calculate loss
    losses = loss_fn(predictions, targets)

    print("\nLoss values:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")

    assert 'total_loss' in losses
    assert 'obj_loss' in losses
    assert 'bbox_loss' in losses
    assert 'class_loss' in losses

    print("\n✓ Loss test passed!")


if __name__ == "__main__":
    test_loss()
