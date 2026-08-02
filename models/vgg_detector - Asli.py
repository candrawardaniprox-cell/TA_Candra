"""
vgg_detector.py — VGG-16 Object Detector (Baseline).

Referensi paper:
  Simonyan, K. & Zisserman, A. (2015).
  "Very Deep Convolutional Networks for Large-Scale Image Recognition."
  International Conference on Learning Representations (ICLR).
  arXiv: 1409.1556

Arsitektur ini menggunakan VGG-16 (with Batch Normalization) pretrained
ImageNet sebagai backbone:
  - 5 blok konvolusi, masing-masing diakhiri MaxPool2d
  - Semua konvolusi menggunakan kernel 3×3 — konsep utama VGGNet
  - Batch Normalization ditambahkan untuk stabilitas training
  - Pretrained weights dari ImageNet untuk transfer learning
  - TANPA komponen Transformer/CTE — murni CNN

VGG-16 dipilih karena:
  - Arsitektur klasik yang membuktikan pentingnya kedalaman jaringan
  - Pola konvolusi 3×3 yang uniform dan sederhana
  - Baseline yang sangat terkenal di literatur computer vision

Digunakan sebagai baseline CNN dalam untuk membandingkan arsitektur
tanpa residual connections vs ResNet vs CNN-Transformer hybrid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from .detector_base import BaseDetector


class VGGDetector(BaseDetector):
    """
    Object detector dengan backbone VGG-16-BN (Simonyan & Zisserman, ICLR 2015).

    Arsitektur backbone VGG-16-BN:
      Block1 → stride 2   (2× Conv3×3 64ch + MaxPool)
      Block2 → stride 4   (2× Conv3×3 128ch + MaxPool)  — c2
      Block3 → stride 8   (3× Conv3×3 256ch + MaxPool)  — c3
      Block4 → stride 16  (3× Conv3×3 512ch + MaxPool)  — c4
      Block5 → stride 32  (3× Conv3×3 512ch + MaxPool)  — c5

    Kemudian masuk ke FPN + Anchor-Free Detection Head (FCOS-style).
    TANPA CTE/Transformer bridge — murni CNN tanpa skip connections.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 640,
        transformer_dim: int = 256,
        **kwargs,
    ):
        fpn_channels = transformer_dim
        super().__init__(num_classes, image_size, fpn_channels)

        # Load VGG-16 dengan Batch Normalization (pretrained jika diaktifkan di config)
        from config import Config as _Cfg
        _use_pretrained = bool(getattr(_Cfg, 'BACKBONE_PRETRAINED', True))
        weights = models.VGG16_BN_Weights.DEFAULT if _use_pretrained else None
        vgg = models.vgg16_bn(weights=weights)
        features = list(vgg.features.children())

        # Cari indeks MaxPool2d untuk memisahkan blok secara robust
        pool_indices = [
            i for i, layer in enumerate(features)
            if isinstance(layer, nn.MaxPool2d)
        ]
        # Untuk VGG-16-BN: pool_indices = [6, 13, 23, 33, 43]
        # Pool1: stride 2   (setelah block1)
        # Pool2: stride 4   (setelah block2) → c2, 128 ch
        # Pool3: stride 8   (setelah block3) → c3, 256 ch
        # Pool4: stride 16  (setelah block4) → c4, 512 ch
        # Pool5: stride 32  (setelah block5) → c5, 512 ch

        # ---- Stem: Block1 (sampai pool pertama) → stride 2, 64 ch ----
        self.backbone = nn.Sequential(*features[: pool_indices[0] + 1])

        # ---- Block2: → stride 4, 128 ch ----
        self.stage_p2 = nn.Sequential(*features[pool_indices[0] + 1 : pool_indices[1] + 1])

        # ---- Block3: → stride 8, 256 ch ----
        self.stage_p3 = nn.Sequential(*features[pool_indices[1] + 1 : pool_indices[2] + 1])

        # ---- Block4: → stride 16, 512 ch ----
        self.stage_p4 = nn.Sequential(*features[pool_indices[2] + 1 : pool_indices[3] + 1])

        # ---- Block5: → stride 32, 512 ch ----
        self.stage_p5 = nn.Sequential(*features[pool_indices[3] + 1 : pool_indices[4] + 1])

        # ---- FPN + Detection Head ----
        self._build_fpn_and_head(
            c2_ch=128, c3_ch=256, c4_ch=512, c5_ch=512,
            fpn_channels=fpn_channels,
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.backbone(x)      # stride 2,  64 ch
        c2 = self.stage_p2(x)     # stride 4,  128 ch
        c3 = self.stage_p3(c2)    # stride 8,  256 ch
        c4 = self.stage_p4(c3)    # stride 16, 512 ch
        c5 = self.stage_p5(c4)    # stride 32, 512 ch
        return self._fpn_head_forward(c2, c3, c4, c5)


def _test_model():
    """Sanity-check forward pass VGGDetector."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 60)
    print("Sanity-check VGGDetector")
    print("=" * 60)

    model = VGGDetector(num_classes=3, image_size=640, transformer_dim=256)
    model.eval()

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = model(x)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameter : {total_params:.2f}M")
    print(f"logits          : {out['logits'].shape}")
    print(f"reg_offsets     : {out['reg_offsets'].shape}")
    print(f"centerness      : {out['centerness'].shape}")
    print(f"grids           : {out['grids']}")

    dets = model.get_detections(x, conf_threshold=0.01)
    print(f"Deteksi batch-0 : {len(dets[0]['boxes'])} kotak")
    print("=" * 60)
    print("VGGDetector OK!")


if __name__ == "__main__":
    _test_model()
