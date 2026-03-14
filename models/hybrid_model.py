import torch
import torch.nn as nn
from .detection_head import DetectionHead
from .transformer import HybridTransformerBlock

class CTE(nn.Module):
    """
    Convolution Token Embedding
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # CTE block as described in paper: Conv -> MaxPool -> ReLU
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=stride, stride=stride),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)

class HybridDetector(nn.Module):
    """
    Lightweight Object Detector using CTE + CPSA + LFFN Backbone
    """
    def __init__(
        self,
        num_classes,
        image_size=512,
        transformer_dim=256, # Final dimension feeding into head
        transformer_heads=4,
        transformer_layers=4,
        transformer_ff_dim=1024,
        num_anchors=7,
        anchors=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        
        # We define 4 hierarchical stages to reach 32x32 grid from 512x512
        # (stride 2 per stage -> total reduction 16x)
        embed_dims = [32, 64, 128, transformer_dim]
        
        # Stage 1 (256x256)
        self.cte1 = CTE(3, embed_dims[0], stride=2)
        
        # Stage 2 (128x128)
        self.cte2 = CTE(embed_dims[0], embed_dims[1], stride=2)
        
        # Stage 3 (64x64)
        self.cte3 = CTE(embed_dims[1], embed_dims[2], stride=2)
        
        # Stage 4 (32x32) - Final representation space
        self.cte4 = CTE(embed_dims[2], embed_dims[3], stride=2)
        
        # Apply CPSA and LFFN at the final stage
        self.transformer_blocks = nn.ModuleList([
            HybridTransformerBlock(
                dim=embed_dims[3], 
                num_heads=transformer_heads,
                reduction_ratio=2, # Pool tokens to save compute
                expansion_ratio=transformer_ff_dim // embed_dims[3],
                dropout=dropout
            )
            for _ in range(transformer_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dims[3])

        # Existing Detection Head (Unchanged)
        self.detection_head = DetectionHead(
            in_channels=embed_dims[3],
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchors=anchors
        )

    def forward(self, x):
        # 1. Hierarchical Feature Extraction (CTE)
        x = self.cte1(x)
        x = self.cte2(x)
        x = self.cte3(x)
        x = self.cte4(x) # Output shape: (B, C, H, W)
        
        B, C, H, W = x.shape
        
        # 2. Flatten for Transformer
        x = x.flatten(2).permute(0, 2, 1) # Output shape: (B, N, C)
        
        # 3. Apply CPSA + LFFN blocks
        for blk in self.transformer_blocks:
            x = blk(x, H, W)
            
        x = self.norm(x)
        
        # 4. Reshape back to 2D for Detection Head
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 5. Detection Head
        return self.detection_head(x)

    def get_detections(self, x, conf_threshold=0.5, nms_iou_threshold=0.4, max_detections=100):
        # Pass through the forward method and get detections
        predictions = self(x)
        return self.detection_head.get_detections(
            predictions, 
            conf_threshold, 
            nms_iou_threshold, 
            max_detections
        )

    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CTE-CPSA-LFFN Backbone + Head Parameters: {total_params / 1e6:.2f} M")