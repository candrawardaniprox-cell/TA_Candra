"""
Stage-2 classifier berbasis HTEM untuk klasifikasi crop penyakit.

Fokus utama:
- memutuskan kelas akhir dengan lebih hati-hati daripada head detector
- memisahkan keputusan `moler` vs `non-moler`
- menyediakan mode `unknown / no_action` jika confidence tidak cukup aman
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_model import HTEMBackbone


class PaperDiseaseClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        stage_dims: list[int] | None = None,
        stage_layout: list[int] | None = None,
        stage_heads: list[int] | None = None,
        stage_reductions: list[int] | None = None,
        cte_channels: int = 16,
        expansion_ratio: int = 4,
        kernel_size: int = 3,
        embed_kernel_size: int = 2,
    ):
        super().__init__()
        if num_classes != 3:
            raise ValueError("Classifier safety ini saat ini diasumsikan untuk 3 kelas penyakit.")

        stage_dims = stage_dims or [24, 32, 48, 64]
        stage_layout = stage_layout or [2, 2, 4, 2]
        stage_heads = stage_heads or [1, 2, 4, 8]
        stage_reductions = stage_reductions or [8, 4, 2, 1]

        self.num_classes = num_classes
        self.backbone = HTEMBackbone(
            stage_dims=stage_dims,
            stage_layout=stage_layout,
            stage_heads=stage_heads,
            stage_reductions=stage_reductions,
            cte_channels=cte_channels,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            kernel_size=kernel_size,
            embed_kernel_size=embed_kernel_size,
        )

        final_dim = stage_dims[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(final_dim),
            nn.Dropout(dropout),
        )

        # Hierarki keputusan:
        # 1. moler vs non-moler
        # 2. jika non-moler, pilih slabung vs ulat_grayak
        self.moler_head = nn.Linear(final_dim, 1)
        self.non_moler_head = nn.Linear(final_dim, 2)

        # Auxiliary flat head untuk membantu representasi global.
        self.flat_head = nn.Linear(final_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, _, c5 = self.backbone(x)
        feat = self.neck(self.pool(c5))
        return {
            "features": feat,
            "moler_logits": self.moler_head(feat).squeeze(-1),
            "non_moler_logits": self.non_moler_head(feat),
            "flat_logits": self.flat_head(feat),
        }

    def combined_probabilities(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        moler_prob = torch.sigmoid(outputs["moler_logits"]).unsqueeze(-1)  # [B, 1]
        non_moler_prob = 1.0 - moler_prob
        non_moler_split = torch.softmax(outputs["non_moler_logits"], dim=-1)  # [B, 2]

        probs = torch.cat(
            [
                moler_prob,
                non_moler_prob * non_moler_split[:, :1],
                non_moler_prob * non_moler_split[:, 1:],
            ],
            dim=-1,
        )

        # Gabungkan dengan head auxiliary agar ranking kelas lebih stabil.
        aux_probs = torch.softmax(outputs["flat_logits"], dim=-1)
        probs = 0.7 * probs + 0.3 * aux_probs
        return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    @torch.no_grad()
    def predict_with_safety(
        self,
        x: torch.Tensor,
        class_thresholds: list[float],
        min_margin: float,
        unknown_index: int = 3,
    ) -> dict[str, torch.Tensor]:
        outputs = self(x)
        probs = self.combined_probabilities(outputs)
        top_probs, top_classes = probs.max(dim=-1)
        top2_probs = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margins = top2_probs[:, 0] - top2_probs[:, 1] if top2_probs.shape[-1] > 1 else top2_probs[:, 0]

        thresholds = torch.tensor(class_thresholds, device=probs.device, dtype=probs.dtype)
        required = thresholds[top_classes]
        accept = (top_probs >= required) & (margins >= float(min_margin))

        final_classes = top_classes.clone()
        final_classes[~accept] = int(unknown_index)

        return {
            "probabilities": probs,
            "top_probs": top_probs,
            "margins": margins,
            "raw_classes": top_classes,
            "final_classes": final_classes,
            "accepted": accept,
        }


def hierarchical_classifier_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels.long()
    moler_target = (labels == 0).float()
    moler_loss = F.binary_cross_entropy_with_logits(outputs["moler_logits"], moler_target)

    non_moler_mask = labels != 0
    if non_moler_mask.any():
        non_moler_labels = labels[non_moler_mask] - 1
        non_moler_loss = F.cross_entropy(
            outputs["non_moler_logits"][non_moler_mask],
            non_moler_labels,
            label_smoothing=label_smoothing,
        )
    else:
        non_moler_loss = outputs["non_moler_logits"].sum() * 0.0

    flat_loss = F.cross_entropy(
        outputs["flat_logits"],
        labels,
        label_smoothing=label_smoothing,
    )

    total = 0.45 * moler_loss + 0.30 * non_moler_loss + 0.25 * flat_loss
    return total, {
        "total_loss": total.detach(),
        "moler_loss": moler_loss.detach(),
        "non_moler_loss": non_moler_loss.detach(),
        "flat_loss": flat_loss.detach(),
    }
