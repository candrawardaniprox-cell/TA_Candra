"""
mobilenet_detector.py — MobileNetV2 Object Detector.

Arsitektur ini menggunakan MobileNetV2 pretrained ImageNet sebagai backbone.
MobileNetV2 sangat ringan dan efisien, cocok untuk perangkat dengan memori kecil
atau edge computing (Mobile/IoT).

Backbone mapping (resolusi 640x640):
  - P2 (stride 4)  : 160x160, 24 channels
  - P3 (stride 8)  : 80x80, 32 channels
  - P4 (stride 16) : 40x40, 96 channels
  - P5 (stride 32) : 20x20, 1280 channels

Murni CNN ringan (Inverted Residuals) + FPN + FCOS-style detection head.
TANPA komponen Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from .detector_base import BaseDetector


class MobileNetDetector(BaseDetector):
    """
    Object detector dengan backbone MobileNetV2 ringan (Sandler et al., CVPR 2018).
    Sangat cocok untuk deteksi real-time di HP/Edge Device.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 640,
        transformer_dim: int = 256,
        **kwargs,
    ):
        fpn_channels = transformer_dim  # Lebar channel di FPN (misal 256)
        super().__init__(num_classes, image_size, fpn_channels)

        # Load MobileNetV2 (pretrained jika diaktifkan di config)
        from config import Config as _Cfg
        _use_pretrained = bool(getattr(_Cfg, 'BACKBONE_PRETRAINED', True))
        weights = models.MobileNet_V2_Weights.DEFAULT if _use_pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
        features = mobilenet.features

        # Split features ke P2, P3, P4, P5 sesuai stride
        # Index 0-3  : Stride 4  (24 channels)
        # Index 4-6  : Stride 8  (32 channels)
        # Index 7-13 : Stride 16 (96 channels)
        # Index 14-18: Stride 32 (1280 channels)
        
        self.stage_p2 = nn.Sequential(*features[0:4])
        self.stage_p3 = nn.Sequential(*features[4:7])
        self.stage_p4 = nn.Sequential(*features[7:14])
        self.stage_p5 = nn.Sequential(*features[14:19])

        # ---- FPN + Detection Head ----
        self._build_fpn_and_head(
            c2_ch=24, c3_ch=32, c4_ch=96, c5_ch=1280,
            fpn_channels=fpn_channels,
        )

    def forward(self, x: torch.Tensor) -> dict:
        c2 = self.stage_p2(x)     # stride 4,  24 ch
        c3 = self.stage_p3(c2)    # stride 8,  32 ch
        c4 = self.stage_p4(c3)    # stride 16, 96 ch
        c5 = self.stage_p5(c4)    # stride 32, 1280 ch
        
        return self._fpn_head_forward(c2, c3, c4, c5)


def _test_model():
    """Sanity-check forward pass MobileNetDetector."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 60)
    print("Sanity-check MobileNetDetector")
    print("=" * 60)

    model = MobileNetDetector(num_classes=3, image_size=640, transformer_dim=256)
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
    print("MobileNetDetector OK!")


if __name__ == "__main__":
    _test_model()
