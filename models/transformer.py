import torch
import torch.nn as nn

class CPSA(nn.Module):
    """
    Convolutional Parameter-Sharing Multi-Head Attention
    """
    def __init__(self, dim, num_heads=4, reduction_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        # Query uses standard linear projection
        self.q = nn.Linear(dim, dim)
        
        # Shared projection S using Depthwise Convolution
        self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio, groups=dim)
        
        # Key and Value are generated from the shared matrix S
        self.kv = nn.Linear(dim, dim * 2)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        # Reshape to 2D for Depthwise Convolution (Token reduction)
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        s_2d = self.sr(x_2d)
        
        # Flatten back to 1D
        s_1d = s_2d.flatten(2).permute(0, 2, 1) # B, M, C

        kv = self.kv(s_1d).reshape(B, -1, 2, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out

class LFFN(nn.Module):
    """
    Local Feed-Forward Network with Sandglass module (DSC)
    """
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        
        # DWConv -> PWConv (Expand) -> DWConv -> PWConv (Reduce)
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        
        self.dwconv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Reshape to 2D to capture spatial local dependencies
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        out = self.dwconv1(x_2d)
        out = self.act(self.pwconv1(out))
        out = self.dwconv2(out)
        out = self.pwconv2(out)
        
        # Flatten back to 1D
        out = out.flatten(2).permute(0, 2, 1)
        return out

class HybridTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio=2, expansion_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CPSA(dim, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = LFFN(dim, expansion_ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W):
        x = x + self.dropout(self.attn(self.norm1(x), H, W))
        x = x + self.dropout(self.ffn(self.norm2(x), H, W))
        return x