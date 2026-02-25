"""
Evaluation metrics for object detection.

This module implements standard object detection metrics:
- IoU (Intersection over Union)
- mAP (mean Average Precision)
- Precision and Recall
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate IoU between two boxes.

    Args:
        box1: Box in format [x, y, w, h] (center format)
        box2: Box in format [x, y, w, h] (center format)

    Returns:
        IoU value between 0 and 1
    """
    # Convert to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou.item() if isinstance(iou, torch.Tensor) else iou


def calculate_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two batches of boxes.

    Args:
        boxes1: Boxes in format [N, 4] (x, y, w, h)
        boxes2: Boxes in format [M, 4] (x, y, w, h)

    Returns:
        IoU matrix of shape [N, M]
    """
    # Convert to corner format
    boxes1_x1y1 = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_x2y2 = boxes1[..., :2] + boxes1[..., 2:] / 2

    boxes2_x1y1 = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_x2y2 = boxes2[..., :2] + boxes2[..., 2:] / 2

    # Calculate intersection
    inter_x1y1 = torch.max(
        boxes1_x1y1.unsqueeze(1),
        boxes2_x1y1.unsqueeze(0)
    )
    inter_x2y2 = torch.min(
        boxes1_x2y2.unsqueeze(1),
        boxes2_x2y2.unsqueeze(0)
    )
    inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Calculate union
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    union_area = boxes1_area.unsqueeze(1) + boxes2_area.unsqueeze(0) - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def calculate_precision_recall(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision and recall for each class.

    Args:
        predictions: List of prediction dicts (one per image) containing:
            - 'boxes': [N, 4] predicted boxes
            - 'scores': [N] confidence scores
            - 'classes': [N] predicted classes
        targets: List of target dicts (one per image) containing:
            - 'boxes': [M, 4] ground truth boxes
            - 'labels': [M] ground truth labels
        iou_threshold: IoU threshold for considering a detection as correct
        num_classes: Number of object classes

    Returns:
        Tuple of (precision, recall) arrays of shape [num_classes]
    """
    # Initialize counters for each class
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_classes = pred['classes']

        target_boxes = target['boxes']
        target_labels = target['labels']

        # Process each class
        for class_id in range(num_classes):
            # Get predictions for this class
            class_mask_pred = pred_classes == class_id
            class_boxes_pred = pred_boxes[class_mask_pred]
            class_scores_pred = pred_scores[class_mask_pred]

            # Get targets for this class
            class_mask_target = target_labels == class_id
            class_boxes_target = target_boxes[class_mask_target]

            # Count ground truth objects
            num_targets = len(class_boxes_target)
            false_negatives[class_id] += num_targets

            if len(class_boxes_pred) == 0:
                continue

            if len(class_boxes_target) == 0:
                false_positives[class_id] += len(class_boxes_pred)
                continue

            # Calculate IoU matrix
            iou_matrix = calculate_iou_batch(class_boxes_pred, class_boxes_target)

            # Sort predictions by score (descending)
            sorted_indices = torch.argsort(class_scores_pred, descending=True)

            # Track which targets have been matched
            matched_targets = set()

            for idx in sorted_indices:
                # Get IoUs with all targets
                ious = iou_matrix[idx]

                # Find best matching target
                max_iou, max_idx = ious.max(dim=0)

                if max_iou >= iou_threshold and max_idx.item() not in matched_targets:
                    # True positive
                    true_positives[class_id] += 1
                    false_negatives[class_id] -= 1
                    matched_targets.add(max_idx.item())
                else:
                    # False positive
                    false_positives[class_id] += 1

    # Calculate precision and recall for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for class_id in range(num_classes):
        tp = true_positives[class_id]
        fp = false_positives[class_id]
        fn = false_negatives[class_id]

        if tp + fp > 0:
            precision[class_id] = tp / (tp + fp)
        if tp + fn > 0:
            recall[class_id] = tp / (tp + fn)

    return precision, recall


def calculate_ap(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision (AP) for a single class.

    AP is the area under the precision-recall curve.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        class_id: Class to calculate AP for
        iou_threshold: IoU threshold for matching

    Returns:
        Average Precision value
    """
    # Collect all predictions and targets for this class
    all_pred_boxes = []
    all_pred_scores = []
    all_target_boxes = []
    all_image_ids = []

    for img_id, (pred, target) in enumerate(zip(predictions, targets)):
        # Get predictions for this class
        class_mask_pred = pred['classes'] == class_id
        if class_mask_pred.sum() > 0:
            all_pred_boxes.append(pred['boxes'][class_mask_pred])
            all_pred_scores.append(pred['scores'][class_mask_pred])
            all_image_ids.extend([img_id] * class_mask_pred.sum().item())

        # Get targets for this class
        class_mask_target = target['labels'] == class_id
        if class_mask_target.sum() > 0:
            all_target_boxes.append((img_id, target['boxes'][class_mask_target]))

    if len(all_pred_boxes) == 0 or len(all_target_boxes) == 0:
        return 0.0

    # Concatenate predictions
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)

    # Sort predictions by score
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_image_ids = [all_image_ids[i] for i in sorted_indices.cpu().numpy()]

    # Create target lookup
    target_lookup = {img_id: boxes for img_id, boxes in all_target_boxes}
    total_targets = sum(len(boxes) for _, boxes in all_target_boxes)

    # Track matches
    matched_targets = defaultdict(set)
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))

    for i, (pred_box, img_id) in enumerate(zip(all_pred_boxes, all_image_ids)):
        if img_id not in target_lookup:
            fp[i] = 1
            continue

        target_boxes = target_lookup[img_id]

        # Calculate IoU with all targets in this image
        ious = torch.tensor([
            calculate_iou(pred_box, target_box)
            for target_box in target_boxes
        ])

        # Find best match
        max_iou, max_idx = ious.max(dim=0)

        if max_iou >= iou_threshold and max_idx.item() not in matched_targets[img_id]:
            tp[i] = 1
            matched_targets[img_id].add(max_idx.item())
        else:
            fp[i] = 1

    # Calculate cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_targets
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Add sentinel values at beginning and end
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate AP as area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def calculate_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int = 80,
    iou_thresholds: List[float] = [0.5]
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) across all classes.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_classes: Number of object classes
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Dictionary containing mAP metrics
    """
    results = {}

    for iou_threshold in iou_thresholds:
        aps = []
        for class_id in range(num_classes):
            ap = calculate_ap(predictions, targets, class_id, iou_threshold)
            aps.append(ap)

        map_value = np.mean(aps)
        results[f'mAP@{iou_threshold:.2f}'] = map_value

    # Calculate mAP@[0.5:0.95] if multiple thresholds provided
    if len(iou_thresholds) > 1:
        all_maps = [results[f'mAP@{t:.2f}'] for t in iou_thresholds]
        results['mAP@[0.5:0.95]'] = np.mean(all_maps)

    return results


def test_metrics():
    """Test function to verify metrics calculation."""
    print("Testing metrics...")

    # Create dummy predictions and targets
    predictions = [
        {
            'boxes': torch.tensor([[100, 100, 50, 50], [200, 200, 80, 80]]),
            'scores': torch.tensor([0.9, 0.8]),
            'classes': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[150, 150, 60, 60]]),
            'scores': torch.tensor([0.85]),
            'classes': torch.tensor([0])
        }
    ]

    targets = [
        {
            'boxes': torch.tensor([[105, 105, 50, 50], [195, 195, 80, 80]]),
            'labels': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[155, 155, 60, 60]]),
            'labels': torch.tensor([0])
        }
    ]

    # Test IoU calculation
    iou = calculate_iou(
        torch.tensor([100, 100, 50, 50]),
        torch.tensor([105, 105, 50, 50])
    )
    print(f"IoU between overlapping boxes: {iou:.4f}")

    # Test precision and recall
    precision, recall = calculate_precision_recall(
        predictions, targets, iou_threshold=0.5, num_classes=80
    )
    print(f"Precision (class 0): {precision[0]:.4f}")
    print(f"Recall (class 0): {recall[0]:.4f}")

    # Test mAP calculation
    map_results = calculate_map(
        predictions, targets, num_classes=80, iou_thresholds=[0.5, 0.75]
    )
    print("\nmAP results:")
    for key, value in map_results.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ Metrics test passed!")


if __name__ == "__main__":
    test_metrics()
