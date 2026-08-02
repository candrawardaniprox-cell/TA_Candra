"""
detector_base.py — Base detector class dengan FPN dan detection head.

Semua baseline detector (Plain CNN, ResNet, VGG) mewarisi kelas ini
agar output dan interface-nya identik dengan HybridDetector.

Komponen shared:
  - FPN top-down fusion (4 level: P2/P3/P4/P5)
  - AnchorFreeDetectionHead (gaya FCOS)
  - Learnable reg_scales per level
  - get_detections() dan get_class_oriented_detections()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detection_head import AnchorFreeDetectionHead


class Scale(nn.Module):
    """Learnable scalar per FPN level untuk menstabilkan bbox regression."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.exp()


class BaseDetector(nn.Module):
    """
    Base class untuk semua model detector di proyek ini.

    Subclass HARUS:
      1. Mendefinisikan self.backbone, self.stage_p2, self.stage_p3,
         self.stage_p4, self.stage_p5  (sebagai nn.Module)
      2. Memanggil self._build_fpn_and_head(c2_ch, c3_ch, c4_ch, c5_ch, fpn_channels)
         di akhir __init__
      3. Mengimplementasikan forward(x) yang memanggil backbone stages lalu
         return self._fpn_head_forward(c2, c3, c4, c5)
    """

    def __init__(self, num_classes: int, image_size: int = 640, fpn_channels: int = 256):
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

    def _build_fpn_and_head(
        self,
        c2_ch: int,
        c3_ch: int,
        c4_ch: int,
        c5_ch: int,
        fpn_channels: int,
    ):
        """Build FPN lateral projections, smoothing, detection head, reg scales."""

        # ---- Lateral / 1×1 projections ----
        self.p2_proj = nn.Sequential(
            nn.Conv2d(c2_ch, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
        )
        self.p3_proj = nn.Sequential(
            nn.Conv2d(c3_ch, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
        )
        self.p4_proj = nn.Sequential(
            nn.Conv2d(c4_ch, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
        )
        self.p5_proj = nn.Sequential(
            nn.Conv2d(c5_ch, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
        )

        # Alias agar train.py / print_model_config tetap kompatibel
        self.lat_p2 = self.p2_proj
        self.lat_p3 = self.p3_proj
        self.lat_p4 = self.p4_proj
        self.lat_p5 = self.p5_proj

        # ---- Smoothing 3×3 ----
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth_p5 = nn.Identity()

        # ---- Detection head (shared) + learnable scales ----
        self.detection_head = AnchorFreeDetectionHead(
            in_channels=fpn_channels,
            num_classes=self.num_classes,
        )
        self.reg_scales = nn.ModuleList([Scale(0.0) for _ in range(4)])

    # ------------------------------------------------------------------
    # FPN + Head forward  (dipanggil oleh subclass)
    # ------------------------------------------------------------------

    def _fpn_head_forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> dict:
        """FPN top-down fusion + detection head → output dict."""

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

    # ------------------------------------------------------------------
    # Inference — identik dengan HybridDetector
    # ------------------------------------------------------------------

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
                topk_idx = torch.argsort(s_scores, descending=True)[: self.det_pre_nms_topk]
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
            valid_boxes = ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) >= 2.0) & (
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
                valid_boxes = ((s_boxes[:, 2] - s_boxes[:, 0]) >= 2.0) & (
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
