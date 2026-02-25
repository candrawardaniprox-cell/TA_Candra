"""
Utilities module for object detection.

This module contains utility functions and classes for:
- loss: Detection loss computation
- metrics: Evaluation metrics (mAP, IoU, etc.)
- nms: Non-Maximum Suppression
- visualization: Bounding box drawing and visualization
"""

from .loss import DetectionLoss
from .metrics import calculate_iou, calculate_map
from .nms import non_max_suppression, batched_nms, class_aware_nms
from .visualization import draw_bounding_boxes, visualize_detections

__all__ = [
    'DetectionLoss',
    'calculate_iou',
    'calculate_map',
    'non_max_suppression',
    'batched_nms',
    'class_aware_nms',
    'draw_bounding_boxes',
    'visualize_detections',
]
