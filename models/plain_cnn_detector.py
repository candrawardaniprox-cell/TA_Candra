"""
plain_cnn_detector.py — Plain CNN Object Detector (Baseline).

Referensi paper:
  LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
  "Gradient-Based Learning Applied to Document Recognition."
  Proceedings of the IEEE, 86(11), 2278-2324.

Arsitektur ini mengikuti prinsip dasar CNN klasik dari LeCun:
  - Lapisan konvolusi bertingkat dengan channel yang meningkat progressif
  - Downsampling melalui stride convolution
  - Batch Normalization + ReLU sebagai modernisasi
  - TANPA residual/skip connections (pembeda utama dari ResNet)
  - TANPA pretrained weights (dilatih dari nol)

Channel progression: 64 → 128 → 256 → 512
Stride progression: 4 → 8 → 16 → 32 (4 level FPN)

Digunakan sebagai baseline untuk membandingkan kinerja CNN murni
terhadap arsitektur yang lebih canggih (ResNet, VGG, CNN-Transformer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .detector_base import BaseDetector


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3,
                  stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Blok dasar CNN: Conv → BatchNorm → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PlainCNNDetector(BaseDetector):
    """
    Object detector dengan backbone Plain CNN (tanpa skip connections).

    Arsitektur backbone mengikuti prinsip LeCun (1998):
      Stem  → stride 4   (Conv 7×7 stride 2 + MaxPool stride 2)
      Stage2 → stride 4   (2× Conv 3×3, 64 channels)
      Stage3 → stride 8   (Conv 3×3 stride 2 + Conv 3×3, 128 channels)
      Stage4 → stride 16  (Conv 3×3 stride 2 + Conv 3×3, 256 channels)
      Stage5 → stride 32  (Conv 3×3 stride 2 + Conv 3×3, 512 channels)

    Kemudian masuk ke FPN + Anchor-Free Detection Head (FCOS-style).
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

        # ---- Stem: stride 4 ----
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ---- Stage 2: stride 4, 64 channels ----
        self.stage_p2 = nn.Sequential(
            _conv_bn_relu(64, 64),
            _conv_bn_relu(64, 64),
        )

        # ---- Stage 3: stride 8, 128 channels ----
        self.stage_p3 = nn.Sequential(
            _conv_bn_relu(64, 128, stride=2),
            _conv_bn_relu(128, 128),
        )

        # ---- Stage 4: stride 16, 256 channels ----
        self.stage_p4 = nn.Sequential(
            _conv_bn_relu(128, 256, stride=2),
            _conv_bn_relu(256, 256),
        )

        # ---- Stage 5: stride 32, 512 channels ----
        self.stage_p5 = nn.Sequential(
            _conv_bn_relu(256, 512, stride=2),
            _conv_bn_relu(512, 512),
        )

        # ---- FPN + Detection Head ----
        self._build_fpn_and_head(
            c2_ch=64, c3_ch=128, c4_ch=256, c5_ch=512,
            fpn_channels=fpn_channels,
        )

                # Inisialisasi Kaiming untuk semua layer (karena tanpa pretrained)
        self._init_weights()

        # Panggil ulang inisialisasi Head agar Focal Loss tidak meledak!
        self.detection_head._init_weights()


    def _init_weights(self):
        """Kaiming initialization untuk semua lapisan konvolusi dan BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.backbone(x)      # stride 4
        c2 = self.stage_p2(x)     # stride 4,  64 ch
        c3 = self.stage_p3(c2)    # stride 8,  128 ch
        c4 = self.stage_p4(c3)    # stride 16, 256 ch
        c5 = self.stage_p5(c4)    # stride 32, 512 ch
        return self._fpn_head_forward(c2, c3, c4, c5)


def _test_model():
    """Sanity-check forward pass PlainCNNDetector."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 60)
    print("Sanity-check PlainCNNDetector")
    print("=" * 60)

    model = PlainCNNDetector(num_classes=3, image_size=640, transformer_dim=256)
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
    print("PlainCNNDetector OK!")


if __name__ == "__main__":
    _test_model()
