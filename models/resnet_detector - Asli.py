"""
resnet_detector.py — ResNet-18 Object Detector (Baseline).

Referensi paper:
  He, K., Zhang, X., Ren, S., & Sun, J. (2016).
  "Deep Residual Learning for Image Recognition."
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
  DOI: 10.1109/CVPR.2016.90

Arsitektur ini menggunakan ResNet-18 pretrained ImageNet sebagai backbone:
  - 4 residual stages menghasilkan feature maps pada stride 4/8/16/32
  - Residual (skip) connections mengatasi vanishing gradient
  - Pretrained weights dari ImageNet untuk transfer learning
  - TANPA komponen Transformer/CTE — murni CNN

ResNet-18 dipilih karena:
  - Backbone ringan dan efisien (11.7M parameter backbone)
  - Sudah terbukti efektif untuk berbagai tugas vision
  - Pretrained ImageNet memberikan fitur awal yang kuat

Digunakan sebagai baseline CNN modern untuk membandingkan efek
residual connections vs plain CNN, dan vs CNN-Transformer hybrid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from .detector_base import BaseDetector


class ResNetDetector(BaseDetector):
    """
    Object detector dengan backbone ResNet-18 (He et al., CVPR 2016).

    Arsitektur backbone ResNet-18:
      Stem   → stride 4   (Conv 7×7 s2 + BN + ReLU + MaxPool s2)
      Layer1 → stride 4   (2× BasicBlock, 64 channels)  — c2
      Layer2 → stride 8   (2× BasicBlock, 128 channels) — c3
      Layer3 → stride 16  (2× BasicBlock, 256 channels) — c4
      Layer4 → stride 32  (2× BasicBlock, 512 channels) — c5

    Kemudian masuk ke FPN + Anchor-Free Detection Head (FCOS-style).
    TANPA CTE/Transformer bridge — murni CNN dengan residual connections.
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

        # Load ResNet-18 (pretrained jika diaktifkan di config)
        from config import Config as _Cfg
        _use_pretrained = bool(getattr(_Cfg, 'BACKBONE_PRETRAINED', True))
        weights = models.ResNet18_Weights.DEFAULT if _use_pretrained else None
        resnet = models.resnet18(weights=weights)

        # ---- Stem: Conv7×7 + BN + ReLU + MaxPool → stride 4 ----
        self.backbone = nn.Sequential(
            resnet.conv1,      # Conv2d(3, 64, 7, stride=2, padding=3)
            resnet.bn1,        # BatchNorm2d(64)
            resnet.relu,       # ReLU
            resnet.maxpool,    # MaxPool2d(3, stride=2, padding=1)
        )

        # ---- Residual Stages ----
        self.stage_p2 = resnet.layer1   # stride 4,  64 ch  (2× BasicBlock)
        self.stage_p3 = resnet.layer2   # stride 8,  128 ch (2× BasicBlock)
        self.stage_p4 = resnet.layer3   # stride 16, 256 ch (2× BasicBlock)
        self.stage_p5 = resnet.layer4   # stride 32, 512 ch (2× BasicBlock)

        # ---- FPN + Detection Head ----
        self._build_fpn_and_head(
            c2_ch=64, c3_ch=128, c4_ch=256, c5_ch=512,
            fpn_channels=fpn_channels,
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.backbone(x)      # stride 4
        c2 = self.stage_p2(x)     # stride 4,  64 ch
        c3 = self.stage_p3(c2)    # stride 8,  128 ch
        c4 = self.stage_p4(c3)    # stride 16, 256 ch
        c5 = self.stage_p5(c4)    # stride 32, 512 ch
        return self._fpn_head_forward(c2, c3, c4, c5)


def _test_model():
    """Sanity-check forward pass ResNetDetector."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 60)
    print("Sanity-check ResNetDetector")
    print("=" * 60)

    model = ResNetDetector(num_classes=3, image_size=640, transformer_dim=256)
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
    print("ResNetDetector OK!")


if __name__ == "__main__":
    _test_model()
