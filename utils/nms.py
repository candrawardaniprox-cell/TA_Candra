"""
Non-Maximum Suppression (NMS) for object detection.

This module implements NMS to remove duplicate detections and keep only
the most confident predictions for each object.
"""

import torch
from typing import Tuple


def box_iou_nms(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU for NMS (optimized version).

    Args:
        boxes1: Boxes in format [N, 4] (x, y, w, h)
        boxes2: Boxes in format [M, 4] (x, y, w, h)

    Returns:
        IoU matrix of shape [N, M]
    """
    # Convert from center format (x, y, w, h) to corner format (x1, y1, x2, y2)
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


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45
) -> torch.Tensor:
    """
    Perform Non-Maximum Suppression on detections.

    NMS removes duplicate detections by:
    1. Sorting all boxes by confidence score (descending)
    2. Iteratively selecting the highest-scoring box
    3. Removing all boxes with IoU > threshold with the selected box
    4. Repeating until no boxes remain

    Args:
        boxes: Bounding boxes of shape [N, 4] in format (x, y, w, h)
        scores: Confidence scores of shape [N]
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Sort boxes by score in descending order
    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = []

    while sorted_indices.numel() > 0:
        # Take the box with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx.item())

        if sorted_indices.numel() == 1:
            break

        # Get current box and remaining boxes
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]

        # Calculate IoU with remaining boxes
        ious = box_iou_nms(current_box, remaining_boxes).squeeze(0)

        # Keep only boxes with IoU <= threshold
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45
) -> torch.Tensor:
    """
    Perform class-aware NMS (NMS applied per class).

    This prevents boxes of different classes from suppressing each other,
    which is important when objects of different classes overlap.

    Args:
        boxes: Bounding boxes of shape [N, 4] in format (x, y, w, h)
        scores: Confidence scores of shape [N]
        classes: Class predictions of shape [N]
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Get unique classes
    unique_classes = classes.unique()

    keep_indices = []

    # Apply NMS for each class separately
    for class_id in unique_classes:
        # Get boxes for this class
        class_mask = classes == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Get original indices for this class
        class_indices = torch.where(class_mask)[0]

        # Apply NMS
        nms_indices = non_max_suppression(
            class_boxes,
            class_scores,
            iou_threshold
        )

        # Map back to original indices
        keep_indices.extend(class_indices[nms_indices].tolist())

    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    max_detections: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform NMS and return filtered boxes, scores, and classes.

    This is a convenience function that applies class-aware NMS and
    limits the number of detections.

    Args:
        boxes: Bounding boxes of shape [N, 4] in format (x, y, w, h)
        scores: Confidence scores of shape [N]
        classes: Class predictions of shape [N]
        iou_threshold: IoU threshold for suppression
        max_detections: Maximum number of detections to return

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_classes)
    """
    if boxes.numel() == 0:
        return (
            torch.empty(0, 4, device=boxes.device),
            torch.empty(0, device=boxes.device),
            torch.empty(0, dtype=torch.long, device=boxes.device)
        )

    # Apply class-aware NMS
    keep_indices = class_aware_nms(boxes, scores, classes, iou_threshold)

    # Sort by score and limit to max_detections
    kept_scores = scores[keep_indices]
    sorted_indices = torch.argsort(kept_scores, descending=True)
    sorted_indices = sorted_indices[:max_detections]

    final_indices = keep_indices[sorted_indices]

    return boxes[final_indices], scores[final_indices], classes[final_indices]


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
    sigma: float = 0.5,
    score_threshold: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Soft-NMS (alternative to hard NMS).

    Soft-NMS decays the scores of overlapping boxes instead of removing them,
    which can improve performance in crowded scenes.

    Args:
        boxes: Bounding boxes of shape [N, 4] in format (x, y, w, h)
        scores: Confidence scores of shape [N]
        iou_threshold: IoU threshold for score decay
        sigma: Gaussian function parameter for score decay
        score_threshold: Minimum score to keep a box

    Returns:
        Tuple of (indices_to_keep, updated_scores)
    """
    if boxes.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=boxes.device),
            torch.empty(0, device=boxes.device)
        )

    # Clone scores to avoid modifying original
    scores = scores.clone()

    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = []

    while sorted_indices.numel() > 0:
        # Take box with highest score
        current_idx = sorted_indices[0]

        # Check if score is above threshold
        if scores[current_idx] < score_threshold:
            break

        keep_indices.append(current_idx.item())

        if sorted_indices.numel() == 1:
            break

        # Get current box and remaining boxes
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]

        # Calculate IoU
        ious = box_iou_nms(current_box, remaining_boxes).squeeze(0)

        # Decay scores based on IoU (Gaussian function)
        weight = torch.exp(-(ious ** 2) / sigma)
        scores[remaining_indices] *= weight

        # Re-sort by updated scores
        updated_scores = scores[remaining_indices]
        new_order = torch.argsort(updated_scores, descending=True)
        sorted_indices = remaining_indices[new_order]

    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
    return keep_indices, scores[keep_indices]


def test_nms():
    """Test function to verify NMS functionality."""
    print("Testing NMS...")

    # Create dummy detections with overlapping boxes
    boxes = torch.tensor([
        [100, 100, 50, 50],  # Box 1
        [105, 105, 50, 50],  # Box 2 (overlaps with 1)
        [200, 200, 60, 60],  # Box 3
        [110, 110, 50, 50],  # Box 4 (overlaps with 1 and 2)
        [205, 205, 60, 60],  # Box 5 (overlaps with 3)
    ], dtype=torch.float32)

    scores = torch.tensor([0.9, 0.85, 0.95, 0.75, 0.88], dtype=torch.float32)
    classes = torch.tensor([0, 0, 1, 0, 1], dtype=torch.long)

    print(f"\nInput boxes: {len(boxes)}")
    print(f"Scores: {scores}")
    print(f"Classes: {classes}")

    # Test regular NMS (class-agnostic)
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.45)
    print(f"\nRegular NMS kept {len(keep_indices)} boxes: {keep_indices}")

    # Test class-aware NMS
    keep_indices_ca = class_aware_nms(boxes, scores, classes, iou_threshold=0.45)
    print(f"Class-aware NMS kept {len(keep_indices_ca)} boxes: {keep_indices_ca}")

    # Test batched NMS
    filtered_boxes, filtered_scores, filtered_classes = batched_nms(
        boxes, scores, classes, iou_threshold=0.45, max_detections=100
    )
    print(f"\nBatched NMS results:")
    print(f"  Boxes: {filtered_boxes.shape}")
    print(f"  Scores: {filtered_scores}")
    print(f"  Classes: {filtered_classes}")

    # Test Soft-NMS
    keep_indices_soft, updated_scores = soft_nms(
        boxes[classes == 0],
        scores[classes == 0],
        iou_threshold=0.45,
        sigma=0.5
    )
    print(f"\nSoft-NMS (class 0 only):")
    print(f"  Kept indices: {keep_indices_soft}")
    print(f"  Updated scores: {updated_scores}")

    print("\n✓ NMS test passed!")


if __name__ == "__main__":
    test_nms()
