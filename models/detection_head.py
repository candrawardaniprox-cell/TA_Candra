import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_tower(in_channels: int, num_layers: int = 3) -> nn.Sequential:
    layers = []
    for _ in range(num_layers):
        layers.extend([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        ])
    return nn.Sequential(*layers)

class AnchorFreeDetectionHead(nn.Module):
    """
    Anchor-Free Detection Head (Gaya FCOS).
    Memprediksi:
    1. Classification (Probabilitas Kelas)
    2. Regression (Jarak l, t, r, b ke tepi objek)
    3. Centerness (Seberapa di-tengah titik tersebut terhadap objek)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Cabang Klasifikasi (Classification Branch)
        self.cls_tower = _make_tower(in_channels)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # 2. Cabang Regresi Jarak Bounding Box (Regression Branch)
        self.bbox_tower = _make_tower(in_channels)
        # Output 4 channel: (left, top, right, bottom)
        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        # 3. Cabang Centerness (Memastikan tebakan di tengah objek nilainya lebih tinggi)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # Inisialisasi khusus Focal Loss untuk layer klasifikasi
        prior_prob = 0.01
        bias_value = -float('inf') if prior_prob == 0 else -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        """
        Input: x dari Transformer (B, C, H, W)
        """
        cls_feat = self.cls_tower(x)
        bbox_feat = self.bbox_tower(x)

        # Probabilitas masing-masing kelas (KITA HAPUS .sigmoid() DI SINI)
        logits = self.cls_logits(cls_feat)
        
        # Prediksi l, t, r, b (Eksponensial/ReLU agar jarak tidak mungkin negatif)
        # Softplus lebih stabil daripada exp untuk regresi bbox multi-level.
        reg_offsets = F.softplus(self.bbox_reg(bbox_feat)) + 1e-4
        
        # Prediksi Centerness (KITA HAPUS .sigmoid() DI SINI)
        centerness = self.centerness(bbox_feat)

        # Ubah bentuk (B, C, H, W) -> (B, H*W, C) agar mudah dihitung di Loss
        B, C, H, W = logits.shape
        logits = logits.flatten(2).permute(0, 2, 1)       # (B, H*W, num_classes)
        reg_offsets = reg_offsets.flatten(2).permute(0, 2, 1) # (B, H*W, 4)
        centerness = centerness.flatten(2).permute(0, 2, 1)   # (B, H*W, 1)

        return {
            'logits': logits,
            'reg_offsets': reg_offsets,
            'centerness': centerness,
            'grid_h': H,
            'grid_w': W
        }
