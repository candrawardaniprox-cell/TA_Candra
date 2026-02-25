"""
YOLO-style Detection Head for object detection.

This module implements the detection head that converts transformer features
into object detection predictions. It predicts objectness, bounding boxes,
and class probabilities for each anchor box at each grid cell.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class DetectionHead(nn.Module):
    """
    YOLO-style detection head for object detection.

    The detection head takes transformer features and predicts detection outputs
    for each grid cell. For each of the 3 anchor boxes per cell, it predicts:
    - Objectness score (1 value): probability that an object exists
    - Bounding box coordinates (4 values): x, y, w, h
    - Class probabilities (num_classes values): probability for each class

    Architecture:
        Input: [B, H*W, C] from transformer
        Reshape: [B, C, H, W] back to spatial
        Conv layers: Process features
        Output: [B, num_anchors * (5 + num_classes), H, W]

    The output format for each anchor:
        [objectness, x_offset, y_offset, width, height, class_1, ..., class_N]

    Args:
        in_channels: Number of input channels from transformer
        num_anchors: Number of anchor boxes per grid cell
        num_classes: Number of object classes
        hidden_dim: Hidden dimension for intermediate conv layers
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.predictions_per_anchor = 5 + num_classes  # obj + bbox(4) + classes

        # Intermediate convolutional layers for feature refinement
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Final prediction layer
        # Outputs: [B, num_anchors * (5 + num_classes), H, W]
        self.pred_conv = nn.Conv2d(
            hidden_dim,
            num_anchors * self.predictions_per_anchor,
            kernel_size=1  # 1x1 conv for predictions
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with special attention to prediction layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize prediction layer with small weights for stable training
        nn.init.normal_(self.pred_conv.weight, mean=0, std=0.01)
        if self.pred_conv.bias is not None:
            nn.init.constant_(self.pred_conv.bias, 0)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Forward pass through detection head.

        Args:
            x: Input features from transformer [B, H*W, C]
            height: Height of the feature map grid
            width: Width of the feature map grid

        Returns:
            Detection predictions of shape [B, num_anchors, H, W, 5+num_classes]
            where the last dimension contains [obj, x, y, w, h, class_scores...]
        """
        B, N, C = x.shape

        # Reshape from sequence to spatial: [B, H*W, C] → [B, C, H, W]
        x = x.permute(0, 2, 1).reshape(B, C, height, width)

        # Process through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Get predictions
        predictions = self.pred_conv(x)  # [B, num_anchors*(5+C), H, W]

        # Reshape to separate anchors and predictions
        # [B, num_anchors*(5+C), H, W] → [B, num_anchors, 5+C, H, W]
        predictions = predictions.view(
            B, self.num_anchors, self.predictions_per_anchor, height, width
        )

        # Permute to [B, num_anchors, H, W, 5+C] for easier processing
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        return predictions

    def decode_predictions(
        self,
        predictions: torch.Tensor,
        anchors: List[Tuple[float, float]],
        grid_size: int,
        image_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode raw predictions into actual bounding boxes.

        This converts the network outputs into interpretable predictions:
        - Apply sigmoid to objectness and class scores
        - Apply sigmoid to x,y offsets and add grid coordinates
        - Apply exponential to w,h and multiply by anchor dimensions

        Args:
            predictions: Raw predictions [B, num_anchors, H, W, 5+num_classes]
            anchors: List of (width, height) tuples for each anchor box
            grid_size: Size of the grid (H or W, assumed square)
            image_size: Original image size for scaling

        Returns:
            Tuple of (objectness, boxes, class_scores)
            - objectness: [B, num_anchors, H, W, 1]
            - boxes: [B, num_anchors, H, W, 4] in format (x, y, w, h)
            - class_scores: [B, num_anchors, H, W, num_classes]
        """
        B, num_anchors, H, W, _ = predictions.shape
        device = predictions.device

        # Split predictions
        obj_pred = predictions[..., 0:1]  # [B, A, H, W, 1]
        bbox_pred = predictions[..., 1:5]  # [B, A, H, W, 4]
        class_pred = predictions[..., 5:]  # [B, A, H, W, num_classes]

        # Apply sigmoid to objectness
        objectness = torch.sigmoid(obj_pred)

        # Create grid coordinates for center point decoding
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        # Decode bounding boxes
        # x, y: apply sigmoid and add grid offset, then scale to image size
        bbox_xy = torch.sigmoid(bbox_pred[..., :2])  # [B, A, H, W, 2]
        bbox_xy[..., 0] += grid_x.unsqueeze(0).unsqueeze(0)  # Add grid_x
        bbox_xy[..., 1] += grid_y.unsqueeze(0).unsqueeze(0)  # Add grid_y
        bbox_xy = bbox_xy / grid_size * image_size  # Scale to image coordinates

        # w, h: apply exponential and multiply by anchor dimensions, then scale
        anchor_tensor = torch.tensor(anchors, device=device).view(1, num_anchors, 1, 1, 2)
        bbox_wh = torch.exp(bbox_pred[..., 2:4]) * anchor_tensor  # [B, A, H, W, 2]
        bbox_wh = bbox_wh * image_size  # Scale to image size

        # Concatenate to form complete boxes
        boxes = torch.cat([bbox_xy, bbox_wh], dim=-1)  # [B, A, H, W, 4]

        # Apply sigmoid to class scores
        class_scores = torch.sigmoid(class_pred)

        return objectness, boxes, class_scores


def test_detection_head():
    """Test function to verify detection head functionality."""
    print("Testing DetectionHead...")

    # Configuration
    batch_size = 2
    in_channels = 256
    num_anchors = 3
    num_classes = 80
    grid_size = 20

    # Create model
    head = DetectionHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    # Create dummy input (output from transformer)
    input_tensor = torch.randn(batch_size, grid_size * grid_size, in_channels)

    # Forward pass
    predictions = head(input_tensor, height=grid_size, width=grid_size)

    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Expected shape: ({batch_size}, {num_anchors}, {grid_size}, {grid_size}, {5+num_classes})")

    # Test decoding
    anchors = [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)]
    objectness, boxes, class_scores = head.decode_predictions(
        predictions, anchors, grid_size=grid_size, image_size=320
    )

    print(f"\nDecoded predictions:")
    print(f"Objectness shape: {objectness.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Class scores shape: {class_scores.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify shapes
    assert predictions.shape == (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes), \
        f"Unexpected prediction shape: {predictions.shape}"
    assert objectness.shape == (batch_size, num_anchors, grid_size, grid_size, 1), \
        f"Unexpected objectness shape: {objectness.shape}"
    assert boxes.shape == (batch_size, num_anchors, grid_size, grid_size, 4), \
        f"Unexpected boxes shape: {boxes.shape}"
    assert class_scores.shape == (batch_size, num_anchors, grid_size, grid_size, num_classes), \
        f"Unexpected class_scores shape: {class_scores.shape}"

    print("✓ Detection head test passed!")


if __name__ == "__main__":
    test_detection_head()
