"""
CNN Backbone for feature extraction.

This module implements a lightweight CNN backbone that progressively extracts
features from input images. The backbone reduces spatial dimensions while
increasing channel depth, providing rich feature representations for the
transformer encoder.
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2d, BatchNorm, and ReLU activation.

    This is a standard building block for CNNs, combining convolution,
    normalization, and activation in a single module.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride of convolution
        padding: Padding added to input
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # Bias not needed before BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNNBackbone(nn.Module):
    """
    Lightweight CNN backbone for feature extraction.

    Architecture:
        Input: [B, 3, 320, 320]
        Conv1: [B, 32, 320, 320]  → MaxPool → [B, 32, 160, 160]
        Conv2: [B, 64, 160, 160]  → MaxPool → [B, 64, 80, 80]
        Conv3: [B, 128, 80, 80]   → MaxPool → [B, 128, 40, 40]
        Conv4: [B, 256, 40, 40]   → MaxPool → [B, 256, 20, 20]
        Output: [B, 256, 20, 20] (1/16 of input resolution)

    The progressive downsampling (4x MaxPool) gives us a 16x reduction,
    transforming 320×320 images to 20×20 feature maps.

    Args:
        channels: List of channel sizes [in, c1, c2, c3, out]
        kernel_size: Kernel size for all convolutions
        padding: Padding for all convolutions
    """

    def __init__(
        self,
        channels: List[int] = [3, 32, 64, 128, 256],
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        assert len(channels) >= 2, "Need at least input and output channels"

        # Build convolutional layers
        layers = []
        for i in range(len(channels) - 1):
            # Add convolutional block
            layers.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            # Add MaxPool after each conv block for spatial downsampling
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.backbone = nn.Sequential(*layers)
        self.out_channels = channels[-1]

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN backbone.

        Args:
            x: Input tensor of shape [B, 3, H, W]

        Returns:
            Feature maps of shape [B, C, H/16, W/16]
        """
        return self.backbone(x)

    def get_output_shape(self, input_size: int) -> tuple:
        """
        Calculate output spatial dimensions.

        Args:
            input_size: Input image size (assuming square images)

        Returns:
            Tuple of (channels, height, width)
        """
        # Each MaxPool reduces by 2x, we have 4 MaxPools
        output_size = input_size // (2 ** (len(self.backbone) // 2))
        return (self.out_channels, output_size, output_size)


def test_backbone():
    """Test function to verify backbone functionality."""
    print("Testing CNNBackbone...")

    # Create model
    backbone = CNNBackbone(channels=[3, 32, 64, 128, 256])

    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 320, 320)

    # Forward pass
    output = backbone(input_tensor)

    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {backbone.get_output_shape(320)}")

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify output shape
    assert output.shape == (batch_size, 256, 20, 20), \
        f"Expected shape (2, 256, 20, 20), got {output.shape}"
    print("✓ Backbone test passed!")


if __name__ == "__main__":
    test_backbone()
