"""
Transformer components for Hybrid CNN-Transformer Object Detection.

Komponen ini mengikuti arsitektur dari paper:
"Image-based onion leaf disease identification using a CNN-Transformer hybrid approach"

Komponen:
- CPSA  : Convolutional Parameter-Sharing Multi-Head Attention
- LFFN  : Local Feed-Forward Network (Sandglass DSC)
- HybridTransformerBlock : satu unit encoder (CPSA + LFFN)
- CTE   : Convolution Token Embedding
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# =============================================================================
# CPSA — Convolutional Parameter-Sharing Multi-Head Attention
# =============================================================================

class CPSA(nn.Module):
    """
    Convolutional Parameter-Sharing Multi-Head Attention.

    K dan V sama-sama dihasilkan dari shared matrix S yang dikompresi via DSC,
    sehingga kompleksitas komputasi berkurang signifikan.
    """

    def __init__(self, dim: int, num_heads: int = 4, reduction_ratio: int = 2):
        super().__init__()
        assert dim % num_heads == 0, "dim harus habis dibagi num_heads"

        self.num_heads = num_heads
        self.dim_head  = dim // num_heads
        self.scale     = self.dim_head ** -0.5

        # Q: linear projection dari input asli
        self.q_proj = nn.Linear(dim, dim, bias=False)

        # DSC untuk membuat shared matrix S
        self.sr_conv = nn.Conv2d(
            dim, dim,
            kernel_size=reduction_ratio,
            stride=reduction_ratio,
            groups=dim,
            bias=False,
        )
        self.sr_pw   = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.sr_norm = nn.LayerNorm(dim)

        # KV dari S (parameter sharing)
        self.kv_proj  = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # Query
        q = self.q_proj(x)
        q = q.reshape(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # Shared matrix S via DSC
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        s_2d = self.sr_conv(x_2d)
        s_2d = self.sr_pw(s_2d)
        M    = s_2d.shape[-2] * s_2d.shape[-1]
        s_1d = s_2d.flatten(2).permute(0, 2, 1)
        s_1d = self.sr_norm(s_1d)

        # Key & Value dari S
        kv = self.kv_proj(s_1d)
        kv = kv.reshape(B, M, 2, self.num_heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out  = self.out_proj(out)
        return out


# =============================================================================
# LFFN — Local Feed-Forward Network (Sandglass DSC)
# =============================================================================

class LFFN(nn.Module):
    """
    Local Feed-Forward Network dengan modul Sandglass DSC.

    Struktur: DWConv(low-dim) → expand → DWConv(high-dim) → reduce
    """

    def __init__(self, dim: int, expansion_ratio: int = 4, kernel_size: int = 3):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        self.dw1       = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2,
                                   groups=dim, bias=False)
        self.bn1       = nn.BatchNorm2d(dim)
        self.pw_expand = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.bn2       = nn.BatchNorm2d(hidden_dim)
        self.act       = nn.GELU()
        self.dw2       = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                                   padding=kernel_size//2, groups=hidden_dim, bias=False)
        self.bn3       = nn.BatchNorm2d(hidden_dim)
        self.pw_reduce = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.bn4       = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)

        out = self.act(self.bn1(self.dw1(x_2d)))
        out = self.act(self.bn2(self.pw_expand(out)))
        out = self.act(self.bn3(self.dw2(out)))
        out = self.bn4(self.pw_reduce(out))

        return out.flatten(2).permute(0, 2, 1)


# =============================================================================
# HybridTransformerBlock — Satu unit Encoder (CPSA + LFFN + residual + LN)
# =============================================================================

class HybridTransformerBlock(nn.Module):
    """
    Satu blok encoder: Pre-Norm CPSA + Pre-Norm LFFN dengan residual connection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int       = 4,
        reduction_ratio: int = 4,
        expansion_ratio: int = 4,
        kernel_size: int     = 3,
        dropout: float       = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = CPSA(dim, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = LFFN(dim, expansion_ratio, kernel_size)
        self.drop  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), H, W))
        x = x + self.drop(self.ffn(self.norm2(x), H, W))
        return x


# =============================================================================
# CTE — Convolution Token Embedding
# =============================================================================

class CTE(nn.Module):
    """
    Convolution Token Embedding.

    Menjembatani CNN backbone ke transformer.
    Conv → BN → ReLU → optional MaxPool (jika stride=2).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


# =============================================================================
# Sanity-check
# =============================================================================

def _test_components():
    print("=" * 55)
    print("Sanity-check komponen Hybrid Transformer")
    print("=" * 55)

    B, C, H, W = 2, 256, 14, 14
    N = H * W
    x_seq = torch.randn(B, N, C)

    cpsa = CPSA(dim=C, num_heads=4, reduction_ratio=4)
    out  = cpsa(x_seq, H, W)
    assert out.shape == (B, N, C)
    print(f"✓ CPSA  : {x_seq.shape} → {out.shape}")

    lffn = LFFN(dim=C, expansion_ratio=4, kernel_size=3)
    out  = lffn(x_seq, H, W)
    assert out.shape == (B, N, C)
    print(f"✓ LFFN  : {x_seq.shape} → {out.shape}")

    block = HybridTransformerBlock(dim=C, num_heads=4,
                                   reduction_ratio=4, expansion_ratio=4)
    out   = block(x_seq, H, W)
    assert out.shape == (B, N, C)
    print(f"✓ Block : {x_seq.shape} → {out.shape}")

    x_feat = torch.randn(B, 512, H, W)
    cte    = CTE(in_channels=512, out_channels=C, stride=1)
    out    = cte(x_feat)
    assert out.shape == (B, C, H, W)
    print(f"✓ CTE   : {x_feat.shape} → {out.shape}")

    cte2  = CTE(in_channels=256, out_channels=C, stride=2)
    x2    = torch.randn(B, 256, H * 2, W * 2)
    out2  = cte2(x2)
    assert out2.shape == (B, C, H, W)
    print(f"✓ CTE(stride=2): {x2.shape} → {out2.shape}")

    print("=" * 55)
    print("Semua komponen Transformer OK!")


if __name__ == "__main__":
    _test_components()