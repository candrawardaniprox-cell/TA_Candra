"""
Hybrid CNN-Transformer Object Detection Model.

This module assembles all components (CNN backbone, Transformer encoder,
Detection head) into a complete end-to-end object detection system.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .backbone import CNNBackbone
from .transformer import TransformerEncoder
from .detection_head import DetectionHead


class HybridDetector(nn.Module):
    """
    Complete Hybrid CNN-Transformer Object Detection Model.

    Architecture Flow:
        1. Input Image [B, 3, 320, 320]
        2. CNN Backbone → Feature Maps [B, 256, 20, 20]
        3. Flatten to Sequence [B, 400, 256]
        4. Transformer Encoder → Contextualized Features [B, 400, 256]
        5. Detection Head → Predictions [B, 3, 20, 20, 85]

    The model combines the local feature extraction of CNNs with the global
    context modeling of Transformers, providing both fine-grained details
    and long-range dependencies for accurate object detection.

    Args:
        num_classes: Number of object classes to detect
        image_size: Input image size (square images assumed)
        backbone_channels: List of channel sizes for CNN backbone
        transformer_dim: Dimension of transformer (must match last backbone channel)
        transformer_heads: Number of attention heads
        transformer_layers: Number of transformer encoder layers
        transformer_ff_dim: Feed-forward network dimension
        num_anchors: Number of anchor boxes per grid cell
        anchors: List of (width, height) tuples for anchor boxes
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 80,
        image_size: int = 320,
        backbone_channels: List[int] = [3, 32, 64, 128, 256],
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 1024,
        num_anchors: int = 3,
        anchors: List[Tuple[float, float]] = [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.num_anchors = num_anchors
        self.anchors = anchors

        # Calculate grid size (input reduced by 16x through 4 MaxPools)
        self.grid_size = image_size // 16

        # CNN Backbone for local feature extraction
        self.backbone = CNNBackbone(
            channels=backbone_channels,
            kernel_size=3,
            padding=1
        )

        # Transformer Encoder for global context modeling
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            d_ff=transformer_ff_dim,
            height=self.grid_size,
            width=self.grid_size,
            dropout=dropout
        )

        # Detection Head for predicting objects
        self.detection_head = DetectionHead(
            in_channels=transformer_dim,
            num_anchors=num_anchors,
            num_classes=num_classes,
            hidden_dim=transformer_dim
        )

        # Store for easy access
        self.predictions_per_anchor = 5 + num_classes

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            x: Input images of shape [B, 3, H, W]
            return_features: If True, return intermediate features

        Returns:
            Dictionary containing:
            - 'predictions': Raw predictions [B, num_anchors, H, W, 5+num_classes]
            - 'objectness': Decoded objectness scores [B, num_anchors, H, W, 1]
            - 'boxes': Decoded bounding boxes [B, num_anchors, H, W, 4]
            - 'class_scores': Decoded class probabilities [B, num_anchors, H, W, num_classes]
            - 'features' (optional): Intermediate features if return_features=True
        """
        # Extract local features with CNN
        cnn_features = self.backbone(x)  # [B, 256, 20, 20]

        # Process through transformer for global context
        transformer_features = self.transformer(cnn_features)  # [B, 400, 256]

        # Generate detection predictions
        predictions = self.detection_head(
            transformer_features,
            height=self.grid_size,
            width=self.grid_size
        )  # [B, 3, 20, 20, 85]

        # Decode predictions to interpretable format
        objectness, boxes, class_scores = self.detection_head.decode_predictions(
            predictions,
            anchors=self.anchors,
            grid_size=self.grid_size,
            image_size=self.image_size
        )

        # Prepare output dictionary
        output = {
            'predictions': predictions,
            'objectness': objectness,
            'boxes': boxes,
            'class_scores': class_scores,
        }

        if return_features:
            output['features'] = {
                'cnn': cnn_features,
                'transformer': transformer_features
            }

        return output

    def get_detections(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get final detections with confidence filtering and NMS.

        Args:
            images: Input images [B, 3, H, W]
            conf_threshold: Minimum confidence threshold
            nms_iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image

        Returns:
            List of dictionaries (one per image) containing:
            - 'boxes': [N, 4] bounding boxes (x, y, w, h)
            - 'scores': [N] confidence scores
            - 'classes': [N] class predictions
        """
        # Get predictions
        with torch.no_grad():
            outputs = self.forward(images)

        objectness = outputs['objectness']  # [B, A, H, W, 1]
        boxes = outputs['boxes']  # [B, A, H, W, 4]
        class_scores = outputs['class_scores']  # [B, A, H, W, C]

        batch_size = images.shape[0]
        detections = []

        for b in range(batch_size):
            # Flatten spatial dimensions and anchors
            obj_b = objectness[b].reshape(-1, 1)  # [A*H*W, 1]
            boxes_b = boxes[b].reshape(-1, 4)  # [A*H*W, 4]
            class_scores_b = class_scores[b].reshape(-1, self.num_classes)  # [A*H*W, C]

            # Get class predictions and scores
            class_probs, class_preds = class_scores_b.max(dim=1)  # [A*H*W]

            # Combine objectness and class probability
            scores = (obj_b.squeeze() * class_probs).cpu()  # [A*H*W]

            # Filter by confidence threshold
            mask = scores > conf_threshold
            if mask.sum() == 0:
                detections.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'classes': torch.empty(0, dtype=torch.long)
                })
                continue

            filtered_boxes = boxes_b[mask].cpu()  # [N, 4]
            filtered_scores = scores[mask]  # [N]
            filtered_classes = class_preds[mask].cpu()  # [N]

            # Apply NMS (will be implemented in utils/nms.py)
            # For now, just sort by score and take top-k
            sorted_indices = torch.argsort(filtered_scores, descending=True)
            sorted_indices = sorted_indices[:max_detections]

            detections.append({
                'boxes': filtered_boxes[sorted_indices],
                'scores': filtered_scores[sorted_indices],
                'classes': filtered_classes[sorted_indices]
            })

        return detections

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        head_params = sum(p.numel() for p in self.detection_head.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'backbone': backbone_params,
            'transformer': transformer_params,
            'detection_head': head_params,
            'total': total_params
        }

    def print_model_summary(self):
        """Print a summary of the model architecture."""
        params = self.count_parameters()

        print("=" * 70)
        print("Hybrid CNN-Transformer Object Detection Model")
        print("=" * 70)
        print(f"Input Size: {self.image_size}x{self.image_size}")
        print(f"Grid Size: {self.grid_size}x{self.grid_size}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Number of Anchors: {self.num_anchors}")
        print("-" * 70)
        print("Component Parameters:")
        print(f"  CNN Backbone:      {params['backbone']:>12,}")
        print(f"  Transformer:       {params['transformer']:>12,}")
        print(f"  Detection Head:    {params['detection_head']:>12,}")
        print("-" * 70)
        print(f"  Total Parameters:  {params['total']:>12,} ({params['total']/1e6:.2f}M)")
        print("=" * 70)


def test_hybrid_model():
    """Test function to verify complete model functionality."""
    print("Testing HybridDetector...")

    # Create model with default COCO configuration
    model = HybridDetector(
        num_classes=80,
        image_size=320,
        backbone_channels=[3, 32, 64, 128, 256],
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=2,
        transformer_ff_dim=1024,
        num_anchors=3
    )

    # Print model summary
    model.print_model_summary()

    # Create dummy input
    batch_size = 2
    input_images = torch.randn(batch_size, 3, 320, 320)

    print(f"\nInput shape: {input_images.shape}")

    # Forward pass
    outputs = model(input_images, return_features=True)

    print("\nOutput shapes:")
    print(f"  Predictions: {outputs['predictions'].shape}")
    print(f"  Objectness: {outputs['objectness'].shape}")
    print(f"  Boxes: {outputs['boxes'].shape}")
    print(f"  Class scores: {outputs['class_scores'].shape}")

    # Test detection extraction
    detections = model.get_detections(input_images, conf_threshold=0.5)
    print(f"\nDetections for {len(detections)} images:")
    for i, det in enumerate(detections):
        print(f"  Image {i}: {len(det['boxes'])} detections")

    # Verify shapes
    assert outputs['predictions'].shape == (batch_size, 3, 20, 20, 85), \
        f"Unexpected predictions shape: {outputs['predictions'].shape}"
    assert outputs['objectness'].shape == (batch_size, 3, 20, 20, 1), \
        f"Unexpected objectness shape: {outputs['objectness'].shape}"
    assert outputs['boxes'].shape == (batch_size, 3, 20, 20, 4), \
        f"Unexpected boxes shape: {outputs['boxes'].shape}"
    assert outputs['class_scores'].shape == (batch_size, 3, 20, 20, 80), \
        f"Unexpected class_scores shape: {outputs['class_scores'].shape}"

    print("\n✓ Hybrid model test passed!")


if __name__ == "__main__":
    test_hybrid_model()
