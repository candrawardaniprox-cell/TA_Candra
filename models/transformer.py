"""
Transformer Encoder for global context modeling.

This module implements a transformer encoder that processes CNN feature maps
as sequences of tokens. The transformer captures long-range dependencies and
global context, which is crucial for object detection.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial feature maps.

    Since transformers are permutation-invariant, we need to inject positional
    information. This module creates learnable 2D positional embeddings that
    are added to the flattened feature map tokens.

    Args:
        d_model: Dimension of the model (feature dimension)
        height: Height of the feature map grid
        width: Width of the feature map grid
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, height: int, width: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create learnable positional embeddings for each spatial position
        # Shape: [1, height*width, d_model]
        self.pos_embedding = nn.Parameter(
            torch.randn(1, height * width, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tokens.

        Args:
            x: Input tensor of shape [B, N, D] where N = H*W

        Returns:
            Tensor with positional encoding added, shape [B, N, D]
        """
        x = x + self.pos_embedding
        return self.dropout(x)


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Sinusoidal 2D Positional Encoding (alternative to learnable).

    Uses sine and cosine functions of different frequencies for each dimension,
    similar to the original Transformer paper but adapted for 2D spatial positions.

    Args:
        d_model: Dimension of the model
        height: Height of the feature map grid
        width: Width of the feature map grid
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, height: int, width: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(height * width, d_model)
        position = torch.arange(0, height * width, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, H*W, D]

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding."""
        x = x + self.pe
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Implements the standard transformer encoder layer with:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Layer normalization
    4. Residual connections

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # Input shape: [B, N, D]
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU activation (commonly used in transformers)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder layer.

        Args:
            x: Input tensor of shape [B, N, D]
            mask: Optional attention mask

        Returns:
            Output tensor of shape [B, N, D]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing CNN feature maps.

    This module converts spatial feature maps into a sequence of tokens,
    adds positional encoding, and processes them through transformer layers
    to capture global context and long-range dependencies.

    Architecture:
        Input: [B, C, H, W] feature maps from CNN
        Flatten: [B, H*W, C] sequence of tokens
        Position: [B, H*W, C] with positional encoding
        Transformer: [B, H*W, C] after N encoder layers
        Output: [B, H*W, C] contextualized features

    Args:
        d_model: Dimension of the model (must match CNN output channels)
        n_heads: Number of attention heads
        n_layers: Number of transformer encoder layers
        d_ff: Dimension of feed-forward network
        height: Height of input feature map
        width: Width of input feature map
        dropout: Dropout probability
        use_sinusoidal_pe: Use sinusoidal instead of learnable positional encoding
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        height: int,
        width: int,
        dropout: float = 0.1,
        use_sinusoidal_pe: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.height = height
        self.width = width

        # Positional encoding
        if use_sinusoidal_pe:
            self.pos_encoding = SinusoidalPositionalEncoding2D(
                d_model, height, width, dropout
            )
        else:
            self.pos_encoding = PositionalEncoding2D(
                d_model, height, width, dropout
            )

        # Stack of transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder.

        Args:
            x: Input feature maps of shape [B, C, H, W]

        Returns:
            Contextualized features of shape [B, H*W, C]
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions: [B, C, H, W] → [B, H*W, C]
        # This converts the 2D feature map into a sequence of tokens
        x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer normalization
        x = self.norm(x)

        return x

    def reshape_to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape sequence back to spatial dimensions.

        Args:
            x: Sequence tensor of shape [B, H*W, C]

        Returns:
            Spatial tensor of shape [B, C, H, W]
        """
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.height, self.width)
        return x


def test_transformer():
    """Test function to verify transformer functionality."""
    print("Testing TransformerEncoder...")

    # Create model
    transformer = TransformerEncoder(
        d_model=256,
        n_heads=8,
        n_layers=2,
        d_ff=1024,
        height=20,
        width=20,
        dropout=0.1
    )

    # Create dummy input (output from CNN backbone)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 256, 20, 20)

    # Forward pass
    output = transformer(input_tensor)

    # Print shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape (sequence): {output.shape}")

    # Reshape back to spatial
    output_spatial = transformer.reshape_to_spatial(output)
    print(f"Output shape (spatial): {output_spatial.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify output shapes
    assert output.shape == (batch_size, 400, 256), \
        f"Expected shape (2, 400, 256), got {output.shape}"
    assert output_spatial.shape == (batch_size, 256, 20, 20), \
        f"Expected shape (2, 256, 20, 20), got {output_spatial.shape}"
    print("✓ Transformer test passed!")


if __name__ == "__main__":
    test_transformer()
