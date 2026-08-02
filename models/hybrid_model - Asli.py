"""
Hybrid CNN-Transformer Object Detector.

Encoder utamanya diubah agar lebih setia ke paper rujukan:
  "Image-based onion leaf disease identification using a CNN-Transformer hybrid approach"

Versi paper adalah model klasifikasi HTEM 4-stage:
  Input -> CTE -> Feature Embedding + [CPSA + LFFN] x N pada 4 stage
        -> GAP -> FC

Adaptasi untuk deteksi pada repo ini:
  Input -> HTEM-style hierarchical encoder (stride 4/8/16/32)
        -> FPN top-down fusion
        -> Anchor-free detection head pada P2/P3/P4/P5

Konfigurasi default encoder mengikuti tabel/ablation paper:
  - CTE channels            : 16
  - stage dims              : [24, 32, 48, 64]
  - stage layout            : [2, 2, 4, 2]
  - heads per stage         : [1, 2, 4, 8]
  - reduction per stage     : [8, 4, 2, 1]
  - LFFN expansion ratio    : 4
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DynamicCNNBackbone
from .detection_head import AnchorFreeDetectionHead
from .transformer import CTE, HybridTransformerBlock


class ConvFeatureAdapter(nn.Module):
    """Proyeksi fitur sederhana saat CTE dimatikan."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureEmbedding(nn.Module):
    """Feature embedding layer ala paper: downsample x2 lalu proyeksi channel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HybridStage(nn.Module):
    """
    Satu stage hierarkis paper:
      Feature Embedding (opsional) -> [CPSA + LFFN] x num_blocks -> LayerNorm
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_blocks: int,
        num_heads: int,
        reduction_ratio: int,
        expansion_ratio: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
        embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList(
            [
                HybridTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    reduction_ratio=reduction_ratio,
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)

        b, c, h, w = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        tokens = self.norm(tokens)
        return tokens.permute(0, 2, 1).reshape(b, c, h, w)


class HTEMBackbone(nn.Module):
    """
    Backbone HTEM 4-stage seperti paper.

    Output:
      c2 : stride 4
      c3 : stride 8
      c4 : stride 16
      c5 : stride 32
    """

    def __init__(
        self,
        stage_dims: list[int],
        stage_layout: list[int],
        stage_heads: list[int],
        stage_reductions: list[int],
        cte_channels: int = 16,
        expansion_ratio: int = 4,
        dropout: float = 0.0,
        kernel_size: int = 3,
        embed_kernel_size: int = 2,
    ):
        super().__init__()
        if not (len(stage_dims) == len(stage_layout) == len(stage_heads) == len(stage_reductions) == 4):
            raise ValueError("Konfigurasi HTEM backbone harus berisi 4 stage.")

        # Paper: conv 3x3 stride 1 -> BN -> ReLU -> maxpool 3x3 stride 2.
        self.cte = nn.Sequential(
            nn.Conv2d(3, cte_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cte_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = HybridStage(
            in_channels=cte_channels,
            dim=stage_dims[0],
            num_blocks=stage_layout[0],
            num_heads=stage_heads[0],
            reduction_ratio=stage_reductions[0],
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            kernel_size=kernel_size,
            embedding=FeatureEmbedding(cte_channels, stage_dims[0], kernel_size=embed_kernel_size, stride=2),
        )
        self.stage2 = HybridStage(
            in_channels=stage_dims[0],
            dim=stage_dims[1],
            num_blocks=stage_layout[1],
            num_heads=stage_heads[1],
            reduction_ratio=stage_reductions[1],
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            kernel_size=kernel_size,
            embedding=FeatureEmbedding(stage_dims[0], stage_dims[1], kernel_size=embed_kernel_size, stride=2),
        )
        self.stage3 = HybridStage(
            in_channels=stage_dims[1],
            dim=stage_dims[2],
            num_blocks=stage_layout[2],
            num_heads=stage_heads[2],
            reduction_ratio=stage_reductions[2],
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            kernel_size=kernel_size,
            embedding=FeatureEmbedding(stage_dims[1], stage_dims[2], kernel_size=embed_kernel_size, stride=2),
        )
        self.stage4 = HybridStage(
            in_channels=stage_dims[2],
            dim=stage_dims[3],
            num_blocks=stage_layout[3],
            num_heads=stage_heads[3],
            reduction_ratio=stage_reductions[3],
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            kernel_size=kernel_size,
            embedding=FeatureEmbedding(stage_dims[2], stage_dims[3], kernel_size=embed_kernel_size, stride=2),
        )

        self.c2_channels = stage_dims[0]
        self.c3_channels = stage_dims[1]
        self.c4_channels = stage_dims[2]
        self.c5_channels = stage_dims[3]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.cte(x)      # stride 2
        c2 = self.stage1(x)  # stride 4
        c3 = self.stage2(c2) # stride 8
        c4 = self.stage3(c3) # stride 16
        c5 = self.stage4(c4) # stride 32
        return c2, c3, c4, c5


class Scale(nn.Module):
    """Learnable scalar per FPN level untuk menstabilkan bbox regression."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.exp()


class HybridDetector(nn.Module):
    """
    Model deteksi hybrid dengan front-end yang bisa diganti via config.

    Alur utama:
      1. Front-end menghasilkan 4 feature maps hierarkis (stride 4/8/16/32)
      2. Semua level diproyeksikan ke channel FPN yang sama
      3. FPN top-down fusion membentuk P2/P3/P4/P5
      4. Detection head dijalankan di semua level
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        transformer_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 1024,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        from config import Config

        self.image_size = image_size
        self.num_classes = num_classes
        self.use_centerness_in_score = bool(getattr(Config, "USE_CENTERNESS_IN_SCORE", False))
        self.centerness_score_weight = float(getattr(Config, "CENTERNESS_SCORE_WEIGHT", 0.25))
        self.det_pre_nms_topk = int(getattr(Config, "DET_PRE_NMS_TOPK", 300))
        self.class_metric_use_second_nms = bool(getattr(Config, "CLASS_METRIC_USE_SECOND_NMS", True))
        self.class_metric_second_nms_iou_threshold = float(
            getattr(Config, "CLASS_METRIC_SECOND_NMS_IOU_THRESHOLD", 0.20)
        )

        self.backbone_name = str(getattr(Config, "BACKBONE_NAME", "resnet18")).lower()
        self.use_backbone = bool(getattr(Config, "DETECTOR_USE_BACKBONE", True))
        self.use_cte = bool(getattr(Config, "DETECTOR_USE_CTE", True))
        self.use_legacy_paper_mode = self.backbone_name == "paper"
        if self.use_legacy_paper_mode:
            self.use_backbone = False
            self.use_cte = True

        if not self.use_backbone and not self.use_cte:
            raise ValueError("Detector membutuhkan minimal satu front-end aktif: backbone atau CTE.")

        self.use_paper_encoder = (not self.use_backbone) and self.use_cte

        if self.use_paper_encoder:
            stage_dims = list(getattr(Config, "PAPER_STAGE_DIMS", [24, 32, 48, 64]))
            stage_layout = list(getattr(Config, "PAPER_STAGE_LAYOUT", [2, 2, 4, 2]))
            stage_heads = list(getattr(Config, "PAPER_STAGE_HEADS", [1, 2, 4, 8]))
            stage_reductions = list(getattr(Config, "PAPER_STAGE_REDUCTIONS", [8, 4, 2, 1]))
            expansion_ratio = int(getattr(Config, "PAPER_LFFN_EXPANSION_RATIO", 4))
            kernel_size = int(getattr(Config, "PAPER_LFFN_KERNEL_SIZE", 3))
            cte_channels = int(getattr(Config, "PAPER_CTE_CHANNELS", 16))
            embed_kernel_size = int(getattr(Config, "PAPER_EMBED_KERNEL_SIZE", 2))

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

            c2_ch, c3_ch, c4_ch, c5_ch = stage_dims

            # Alias agar logging/train yang ada tetap jalan.
            self.stage_p2 = self.backbone.stage1
            self.stage_p3 = self.backbone.stage2
            self.stage_p4 = self.backbone.stage3
            self.stage_p5 = self.backbone.stage4
        else:
            expansion_ratio = max(1, transformer_ff_dim // transformer_dim)
            self.backbone = DynamicCNNBackbone(
                backbone_name=self.backbone_name,
                pretrained=Config.BACKBONE_PRETRAINED,
                pretrain_source=getattr(Config, "BACKBONE_PRETRAIN_SOURCE", "imagenet"),
                custom_weights_path=getattr(Config, "BACKBONE_CUSTOM_WEIGHTS_PATH", None),
            )

            c3_in = self.backbone.c3_channels
            c4_in = self.backbone.c4_channels
            c2_ch = transformer_dim
            embedding_cls = CTE if self.use_cte else ConvFeatureAdapter

            self.stage_p3 = HybridStage(
                in_channels=c3_in,
                dim=transformer_dim,
                num_blocks=transformer_layers,
                num_heads=transformer_heads,
                reduction_ratio=2,
                expansion_ratio=expansion_ratio,
                dropout=dropout,
                embedding=embedding_cls(in_channels=c3_in, out_channels=transformer_dim, stride=1),
            )
            self.stage_p4 = HybridStage(
                in_channels=c4_in,
                dim=transformer_dim,
                num_blocks=transformer_layers,
                num_heads=transformer_heads,
                reduction_ratio=2,
                expansion_ratio=expansion_ratio,
                dropout=dropout,
                embedding=embedding_cls(in_channels=c4_in, out_channels=transformer_dim, stride=1),
            )
            self.stage_p5 = HybridStage(
                in_channels=transformer_dim,
                dim=transformer_dim,
                num_blocks=transformer_layers,
                num_heads=transformer_heads,
                reduction_ratio=2,
                expansion_ratio=expansion_ratio,
                dropout=dropout,
                embedding=embedding_cls(in_channels=transformer_dim, out_channels=transformer_dim, stride=2),
            )

            # Tambah level stride 4 ringan agar FPN tetap 4 level.
            self.stage_p2 = nn.Sequential(
                nn.Conv2d(c3_in, transformer_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(transformer_dim),
                nn.ReLU(inplace=True),
            )
            c3_ch = c4_ch = c5_ch = transformer_dim

        # Semua level diproyeksikan ke channel FPN yang sama.
        self.p2_proj = nn.Sequential(
            nn.Conv2d(c2_ch, transformer_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
        )
        self.p3_proj = nn.Sequential(
            nn.Conv2d(c3_ch, transformer_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
        )
        self.p4_proj = nn.Sequential(
            nn.Conv2d(c4_ch, transformer_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
        )
        self.p5_proj = nn.Sequential(
            nn.Conv2d(c5_ch, transformer_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
        )

        # Alias nama lateral lama agar train.py yang ada tetap kompatibel.
        self.lat_p2 = self.p2_proj
        self.lat_p3 = self.p3_proj
        self.lat_p4 = self.p4_proj
        self.lat_p5 = self.p5_proj

        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.ReLU(inplace=True),
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.ReLU(inplace=True),
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.ReLU(inplace=True),
        )
        self.smooth_p5 = nn.Identity()

        self.detection_head = AnchorFreeDetectionHead(
            in_channels=transformer_dim,
            num_classes=num_classes,
        )
        self.reg_scales = nn.ModuleList([Scale(0.0) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> dict:
        if self.use_paper_encoder:
            c2, c3, c4, c5 = self.backbone(x)
        else:
            c3_raw, c4_raw = self.backbone(x)
            c2 = self.stage_p2(c3_raw)
            c3 = self.stage_p3(c3_raw)
            c4 = self.stage_p4(c4_raw)
            c5 = self.stage_p5(c4)

        p5 = self.smooth_p5(self.p5_proj(c5))
        p4 = self.smooth_p4(
            self.p4_proj(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        )
        p3 = self.smooth_p3(
            self.p3_proj(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        )
        p2 = self.smooth_p2(
            self.p2_proj(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        )

        out2 = self.detection_head(p2)
        out3 = self.detection_head(p3)
        out4 = self.detection_head(p4)
        out5 = self.detection_head(p5)

        out2["reg_offsets"] = self.reg_scales[0](out2["reg_offsets"])
        out3["reg_offsets"] = self.reg_scales[1](out3["reg_offsets"])
        out4["reg_offsets"] = self.reg_scales[2](out4["reg_offsets"])
        out5["reg_offsets"] = self.reg_scales[3](out5["reg_offsets"])

        return {
            "logits": torch.cat(
                [out2["logits"], out3["logits"], out4["logits"], out5["logits"]],
                dim=1,
            ),
            "reg_offsets": torch.cat(
                [out2["reg_offsets"], out3["reg_offsets"], out4["reg_offsets"], out5["reg_offsets"]],
                dim=1,
            ),
            "centerness": torch.cat(
                [out2["centerness"], out3["centerness"], out4["centerness"], out5["centerness"]],
                dim=1,
            ),
            "grids": [
                (out2["grid_h"], out2["grid_w"]),
                (out3["grid_h"], out3["grid_w"]),
                (out4["grid_h"], out4["grid_w"]),
                (out5["grid_h"], out5["grid_w"]),
            ],
        }

    @torch.no_grad()
    def get_detections(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 100,
        outputs: dict = None,
    ):
        """
        Jalankan forward + decode + NMS untuk inference.

        Returns:
            List[dict] dengan kunci 'boxes', 'scores', 'classes'
            (panjang list = batch size)
        """
        from torchvision.ops import batched_nms as tv_batched_nms

        outputs = self(x) if outputs is None else outputs
        logits = outputs["logits"]
        reg_offsets = outputs["reg_offsets"]
        centerness = outputs["centerness"].sigmoid()
        grids = outputs["grids"]

        b = logits.shape[0]
        device = logits.device

        locations = []
        level_strides = []
        for h, w in grids:
            stride = self.image_size // h
            sy, sx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            sx = (sx + 0.5) * stride
            sy = (sy + 0.5) * stride
            locations.append(torch.stack([sx, sy], dim=-1).reshape(-1, 2))
            level_strides.append(torch.full((h * w,), float(stride), device=device))

        locations = torch.cat(locations, dim=0)
        level_strides = torch.cat(level_strides, dim=0)
        all_detections = []

        for i in range(b):
            class_scores = logits[i].sigmoid()
            if self.use_centerness_in_score:
                scores_map = class_scores * centerness[i].pow(self.centerness_score_weight)
            else:
                scores_map = class_scores
            max_scores, class_preds = scores_map.max(dim=-1)

            mask = max_scores > conf_threshold
            if not mask.any():
                all_detections.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "classes": torch.zeros((0,), device=device),
                    }
                )
                continue

            s_scores = max_scores[mask]
            s_classes = class_preds[mask]
            s_offsets = reg_offsets[i][mask] * level_strides[mask].unsqueeze(-1)
            s_locs = locations[mask]

            if self.det_pre_nms_topk > 0 and s_scores.numel() > self.det_pre_nms_topk:
                topk_idx = torch.argsort(s_scores, descending=True)[:self.det_pre_nms_topk]
                s_scores = s_scores[topk_idx]
                s_classes = s_classes[topk_idx]
                s_offsets = s_offsets[topk_idx]
                s_locs = s_locs[topk_idx]

            x1 = s_locs[:, 0] - s_offsets[:, 0]
            y1 = s_locs[:, 1] - s_offsets[:, 1]
            x2 = s_locs[:, 0] + s_offsets[:, 2]
            y2 = s_locs[:, 1] + s_offsets[:, 3]
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0.0, float(self.image_size))
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0.0, float(self.image_size))
            valid_boxes = (
                (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) >= 2.0
            ) & (
                (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) >= 2.0
            )
            if not valid_boxes.any():
                all_detections.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "classes": torch.zeros((0,), device=device),
                    }
                )
                continue
            boxes_xyxy = boxes_xyxy[valid_boxes]
            s_scores = s_scores[valid_boxes]
            s_classes = s_classes[valid_boxes]

            keep = tv_batched_nms(boxes_xyxy, s_scores, s_classes, nms_iou_threshold)
            keep = keep[:max_detections]

            all_detections.append(
                {
                    "boxes": boxes_xyxy[keep],
                    "scores": s_scores[keep],
                    "classes": s_classes[keep].float(),
                }
            )

        return all_detections

    @torch.no_grad()
    def get_class_oriented_detections(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.35,
        nms_iou_threshold: float = 0.35,
        max_detections: int = 12,
        outputs: dict = None,
        use_centerness_in_score: bool | None = None,
        centerness_score_weight: float | None = None,
        use_second_nms: bool | None = None,
        second_nms_iou_threshold: float | None = None,
    ):
        """
        Inference class-oriented untuk metrik kelas dan confusion matrix kelas.

        Skor kelas bisa dikalibrasi ringan oleh centerness jika diaktifkan
        di config. Bbox tetap diambil dari regression head agar output visual
        masih menampilkan kotak prediksi.
        """
        from torchvision.ops import batched_nms as tv_batched_nms

        outputs = self(x) if outputs is None else outputs
        logits = outputs["logits"]
        reg_offsets = outputs["reg_offsets"]
        grids = outputs["grids"]

        b = logits.shape[0]
        device = logits.device

        locations = []
        level_strides = []
        for h, w in grids:
            stride = self.image_size // h
            sy, sx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            sx = (sx + 0.5) * stride
            sy = (sy + 0.5) * stride
            locations.append(torch.stack([sx, sy], dim=-1).reshape(-1, 2))
            level_strides.append(torch.full((h * w,), float(stride), device=device))

        locations = torch.cat(locations, dim=0)
        level_strides = torch.cat(level_strides, dim=0)
        all_detections = []
        centerness = outputs["centerness"].sigmoid()

        if use_centerness_in_score is None:
            use_centerness_in_score = self.use_centerness_in_score
        if centerness_score_weight is None:
            centerness_score_weight = self.centerness_score_weight
        if use_second_nms is None:
            use_second_nms = self.class_metric_use_second_nms
        if second_nms_iou_threshold is None:
            second_nms_iou_threshold = self.class_metric_second_nms_iou_threshold

        for i in range(b):
            class_scores = logits[i].sigmoid()
            if use_centerness_in_score:
                class_scores = class_scores * centerness[i].pow(centerness_score_weight)
            det_boxes = []
            det_scores = []
            det_classes = []

            for cls_id in range(class_scores.shape[1]):
                cls_scores = class_scores[:, cls_id]
                mask = cls_scores > conf_threshold
                if not mask.any():
                    continue

                s_scores = cls_scores[mask]
                s_offsets = reg_offsets[i][mask] * level_strides[mask].unsqueeze(-1)
                s_locs = locations[mask]
                x1 = s_locs[:, 0] - s_offsets[:, 0]
                y1 = s_locs[:, 1] - s_offsets[:, 1]
                x2 = s_locs[:, 0] + s_offsets[:, 2]
                y2 = s_locs[:, 1] + s_offsets[:, 3]
                s_boxes = torch.stack([x1, y1, x2, y2], dim=1)
                s_boxes[:, [0, 2]] = s_boxes[:, [0, 2]].clamp(0.0, float(self.image_size))
                s_boxes[:, [1, 3]] = s_boxes[:, [1, 3]].clamp(0.0, float(self.image_size))
                valid_boxes = (
                    (s_boxes[:, 2] - s_boxes[:, 0]) >= 2.0
                ) & (
                    (s_boxes[:, 3] - s_boxes[:, 1]) >= 2.0
                )
                if not valid_boxes.any():
                    continue
                s_boxes = s_boxes[valid_boxes]
                s_scores = s_scores[valid_boxes]
                s_classes = torch.full(
                    (s_scores.shape[0],),
                    cls_id,
                    dtype=torch.long,
                    device=device,
                )

                keep = tv_batched_nms(
                    s_boxes,
                    s_scores,
                    torch.zeros_like(s_classes),
                    nms_iou_threshold,
                )

                det_boxes.append(s_boxes[keep])
                det_scores.append(s_scores[keep])
                det_classes.append(s_classes[keep].float())

            if not det_boxes:
                all_detections.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "classes": torch.zeros((0,), device=device),
                    }
                )
                continue

            boxes_cat = torch.cat(det_boxes, dim=0)
            scores_cat = torch.cat(det_scores, dim=0)
            classes_cat = torch.cat(det_classes, dim=0)

            if use_second_nms and boxes_cat.shape[0] > 0:
                second_keep = tv_batched_nms(
                    boxes_cat,
                    scores_cat,
                    torch.zeros_like(classes_cat, dtype=torch.long),
                    second_nms_iou_threshold,
                )
                boxes_cat = boxes_cat[second_keep]
                scores_cat = scores_cat[second_keep]
                classes_cat = classes_cat[second_keep]

            order = torch.argsort(scores_cat, descending=True)[:max_detections]

            all_detections.append(
                {
                    "boxes": boxes_cat[order],
                    "scores": scores_cat[order],
                    "classes": classes_cat[order],
                }
            )

        return all_detections


def _test_model():
    """Uji forward pass model secara menyeluruh."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    class _Cfg:
        BACKBONE_NAME = "paper"
        BACKBONE_PRETRAINED = False
        IMAGE_SIZE = 256
        NUM_CLASSES = 3
        TRANSFORMER_DIM = 128
        TRANSFORMER_HEADS = 4
        TRANSFORMER_LAYERS = 2
        TRANSFORMER_FF_DIM = 512
        TRANSFORMER_DROPOUT = 0.1
        PAPER_CTE_CHANNELS = 16
        PAPER_STAGE_DIMS = [24, 32, 48, 64]
        PAPER_STAGE_LAYOUT = [2, 2, 4, 2]
        PAPER_STAGE_HEADS = [1, 2, 4, 8]
        PAPER_STAGE_REDUCTIONS = [8, 4, 2, 1]
        PAPER_LFFN_EXPANSION_RATIO = 4

    import config as cfg_mod

    cfg_mod.Config.BACKBONE_NAME = _Cfg.BACKBONE_NAME
    cfg_mod.Config.BACKBONE_PRETRAINED = _Cfg.BACKBONE_PRETRAINED
    cfg_mod.Config.PAPER_CTE_CHANNELS = _Cfg.PAPER_CTE_CHANNELS
    cfg_mod.Config.PAPER_STAGE_DIMS = _Cfg.PAPER_STAGE_DIMS
    cfg_mod.Config.PAPER_STAGE_LAYOUT = _Cfg.PAPER_STAGE_LAYOUT
    cfg_mod.Config.PAPER_STAGE_HEADS = _Cfg.PAPER_STAGE_HEADS
    cfg_mod.Config.PAPER_STAGE_REDUCTIONS = _Cfg.PAPER_STAGE_REDUCTIONS
    cfg_mod.Config.PAPER_LFFN_EXPANSION_RATIO = _Cfg.PAPER_LFFN_EXPANSION_RATIO

    print("=" * 60)
    print("Sanity-check HybridDetector")
    print("=" * 60)

    model = HybridDetector(
        num_classes=_Cfg.NUM_CLASSES,
        image_size=_Cfg.IMAGE_SIZE,
        transformer_dim=_Cfg.TRANSFORMER_DIM,
        transformer_heads=_Cfg.TRANSFORMER_HEADS,
        transformer_layers=_Cfg.TRANSFORMER_LAYERS,
        transformer_ff_dim=_Cfg.TRANSFORMER_FF_DIM,
        dropout=_Cfg.TRANSFORMER_DROPOUT,
    )
    model.eval()

    x = torch.randn(2, 3, _Cfg.IMAGE_SIZE, _Cfg.IMAGE_SIZE)
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
    print("HybridDetector OK!")


if __name__ == "__main__":
    _test_model()
