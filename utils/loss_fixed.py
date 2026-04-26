"""
loss_fixed.py — FCOS Anchor-Free Loss dengan bobot kelas diperkuat untuk
slabung dan ulat_grayak.

Distribusi data nyata:
  Train : moler=2135, slabung=1303, ulat_grayak=1467

Strategi pembobotan (2 lapis):
  [A] FOCAL ALPHA per kelas — mengurangi kontribusi easy negatives:
        moler       : alpha = 0.20  (kelas dominan, dikurangi)
        slabung     : alpha = 0.55  (kelas paling sedikit, ditingkatkan)
        ulat_grayak : alpha = 0.45  (kelas sedang, ditingkatkan)

  [B] CLASS MULTIPLIER — bobot tambahan langsung pada loss:
        moler       : ×1.0
        slabung     : ×4.0  (paling sedikit data → paling butuh perhatian)
        ulat_grayak : ×3.0

  [C] FOCAL GAMMA = 3.0 — penalti besar pada prediksi yang salah.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ── Konfigurasi bobot kelas ─────────────────────────────────────────────────────
# [A] Alpha focal per kelas
CLASS_ALPHA = list(getattr(Config, 'LOSS_CLASS_ALPHA', [0.35, 0.40, 0.35]))

# [B] Pengali bobot tambahan per kelas
CLASS_MULTIPLIER = list(getattr(Config, 'LOSS_CLASS_MULTIPLIER', [2.0, 2.2, 2.0]))

# [C] Focal gamma
FOCAL_GAMMA = float(getattr(Config, 'FOCAL_GAMMA', 2.0))


class AnchorFreeLoss(nn.Module):
    """
    FCOS-style Anchor-Free Loss.

    Perbaikan vs versi awal:
    - Bug #1: GT (cx,cy,w,h) → (x1,y1,x2,y2) sebelum hitung l,t,r,b.
    - Bug #2: FPN scale pakai max(w,h) dari GT box.
    - Focal Loss dengan alpha & gamma per kelas untuk class imbalance.
    - GIoU regression loss dengan dynamic small-object weighting.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.current_epoch = 1

        # Simpan sebagai buffer (tidak ikut gradient)
        alpha_t = torch.tensor(CLASS_ALPHA[:num_classes],      dtype=torch.float32)
        mult_t  = torch.tensor(CLASS_MULTIPLIER[:num_classes], dtype=torch.float32)
        self.register_buffer('class_alpha',      alpha_t)
        self.register_buffer('class_multiplier', mult_t)

    def set_epoch(self, epoch: int) -> None:
        """Dipanggil dari train.py agar bobot loss bisa mengikuti fase training."""
        self.current_epoch = max(1, int(epoch))

    def _get_loss_weights(self) -> tuple[float, float, float]:
        """
        Class-oriented objective untuk semua epoch:
        - class loss selalu paling dominan
        - bbox dan centerness tetap aktif dengan bobot kecil
        - tujuan utamanya selaras dengan pemilihan kelas penyakit
        """
        class_w = float(Config.LAMBDA_CLASS)
        bbox_target = float(Config.LAMBDA_BBOX)
        obj_target = float(Config.LAMBDA_OBJ)

        if not getattr(Config, 'CLASS_PRIORITY_MODE', True):
            return class_w, bbox_target, obj_target
        return class_w, bbox_target, obj_target

    def _sanitize_gt(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sanitasi tambahan tepat sebelum target dipakai oleh kernel CUDA.

        Ini adalah lapisan pengaman terakhir jika ada data aneh yang lolos dari dataset/dataloader.
        """
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
        else:
            boxes = boxes.to(device=device, dtype=torch.float32)

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device=device).reshape(-1).long()

        if boxes.numel() == 0:
            return boxes.reshape(0, 4), labels.reshape(0)

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            return boxes.new_zeros((0, 4)), labels.new_zeros((0,), dtype=torch.long)

        pair_count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:pair_count]
        labels = labels[:pair_count]

        valid = torch.isfinite(boxes).all(dim=1)
        valid &= torch.isfinite(labels.float())
        valid &= boxes[:, 2] > 1e-6
        valid &= boxes[:, 3] > 1e-6
        valid &= labels >= 0
        valid &= labels < self.num_classes

        boxes = boxes[valid]
        labels = labels[valid]

        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4)), labels.new_zeros((0,), dtype=torch.long)

        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0].clamp(0.0, float(Config.IMAGE_SIZE))
        boxes[:, 1] = boxes[:, 1].clamp(0.0, float(Config.IMAGE_SIZE))
        boxes[:, 2] = boxes[:, 2].clamp(1e-6, float(Config.IMAGE_SIZE))
        boxes[:, 3] = boxes[:, 3].clamp(1e-6, float(Config.IMAGE_SIZE))
        return boxes, labels.long()

    # ── Focal loss dengan alpha per kelas ──────────────────────────────────────
    def _focal_loss(
        self,
        logits:     torch.Tensor,   # (N, C)  raw logits
        cls_target: torch.Tensor,   # (N, C)  one-hot
    ) -> torch.Tensor:
        """Focal Loss dengan alpha & multiplier berbeda per kelas."""
        p    = torch.sigmoid(logits)
        p_t  = torch.where(cls_target == 1, p, 1 - p)

        bce  = F.binary_cross_entropy_with_logits(logits, cls_target, reduction='none')
        focal_w = (1.0 - p_t) ** FOCAL_GAMMA

        alpha = self.class_alpha.to(logits.device)
        alpha_t = torch.where(
            cls_target == 1,
            alpha.unsqueeze(0).expand_as(cls_target),
            1.0 - alpha.unsqueeze(0).expand_as(cls_target),
        )

        # Pengali per kelas
        mult = self.class_multiplier.to(logits.device)  # (C,)
        mult_matrix = mult.unsqueeze(0).expand_as(cls_target)   # (N, C)

        loss = alpha_t * focal_w * bce * mult_matrix
        return loss  # (N, C)

    # ── Forward ────────────────────────────────────────────────────────────────
    def forward(self, outputs: dict, targets: dict) -> dict:
        logits      = outputs['logits']       # (B, N, C)
        reg_offsets = outputs['reg_offsets']  # (B, N, 4)
        centerness  = outputs['centerness']   # (B, N, 1)
        grids       = outputs['grids']

        B, N, _ = logits.shape
        device  = logits.device

        # ── 1. Lokasi anchor dan batas FPN ────────────────────────────────────
        locations, fpn_min, fpn_max, level_strides = [], [], [], []
        for H, W in grids:
            stride = Config.IMAGE_SIZE // H
            sy, sx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij',
            )
            sx = (sx + 0.5) * stride
            sy = (sy + 0.5) * stride
            locations.append(torch.stack([sx, sy], dim=-1).reshape(-1, 2))

            if stride == 4:
                lo, hi = 0.0, 32.0
            elif stride == 8:
                lo, hi = 32.0, 64.0
            elif stride == 16:
                lo, hi = 64.0, 128.0
            else:
                lo, hi = 128.0, 99999.0

            fpn_min.append(torch.full((H * W,), lo, device=device))
            fpn_max.append(torch.full((H * W,), hi, device=device))
            level_strides.append(torch.full((H * W,), float(stride), device=device))

        locations = torch.cat(locations, dim=0)
        fpn_min   = torch.cat(fpn_min,   dim=0)
        fpn_max   = torch.cat(fpn_max,   dim=0)
        level_strides = torch.cat(level_strides, dim=0)

        # ── 2. Hitung loss per sampel ─────────────────────────────────────────
        cls_losses, reg_losses, ctr_losses = [], [], []
        num_positives = 0

        for i in range(B):
            gt_cxywh, gt_labels = self._sanitize_gt(
                targets['boxes'][i],
                targets['labels'][i],
                device,
            )

            # Gambar tanpa objek
            if len(gt_cxywh) == 0:
                cls_target = torch.zeros_like(logits[i])
                cls_losses.append(self._focal_loss(logits[i], cls_target).sum())
                continue

            # Bug #1 Fix: cxywh → xyxy
            cx, cy, w, h = gt_cxywh.unbind(dim=-1)
            gt_x1 = cx - w / 2.0
            gt_y1 = cy - h / 2.0
            gt_x2 = cx + w / 2.0
            gt_y2 = cy + h / 2.0

            xs = locations[:, 0]
            ys = locations[:, 1]

            l = xs[:, None] - gt_x1[None, :]
            t = ys[:, None] - gt_y1[None, :]
            r = gt_x2[None, :] - xs[:, None]
            b = gt_y2[None, :] - ys[:, None]

            reg_targets_per_im = torch.stack([l, t, r, b], dim=-1)  # (N, G, 4)
            is_in_boxes = reg_targets_per_im.min(dim=-1)[0] > 0

            # Bug #2 Fix: scale pakai max(w, h)
            gt_max_wh = torch.max(w, h)
            is_in_level = (
                (gt_max_wh[None, :] >= fpn_min[:, None]) &
                (gt_max_wh[None, :] <  fpn_max[:, None])
            )
            is_in_boxes = is_in_boxes & is_in_level

            gt_areas      = w * h
            areas_per_loc = gt_areas[None, :].repeat(N, 1)
            areas_per_loc[~is_in_boxes] = 1e8

            min_area, min_idx = areas_per_loc.min(dim=1)
            loc_mask = min_area < 1e8
            num_positives += loc_mask.sum().item()

            reg_targets_raw = reg_targets_per_im[torch.arange(N), min_idx]
            reg_targets    = reg_targets_raw / level_strides.unsqueeze(-1)
            matched_labels = gt_labels[min_idx].long()

            # Classification loss
            cls_target = torch.zeros_like(logits[i])
            if loc_mask.any():
                pos_idx = torch.nonzero(loc_mask, as_tuple=False).squeeze(1)
                pos_labels = matched_labels[loc_mask]
                valid_pos = (pos_labels >= 0) & (pos_labels < self.num_classes)
                if valid_pos.any():
                    cls_target[pos_idx[valid_pos], pos_labels[valid_pos]] = 1.0
            cls_losses.append(self._focal_loss(logits[i], cls_target).sum())

            if loc_mask.sum() == 0:
                continue

            # Regression (GIoU) + Centerness
            pos_pred   = reg_offsets[i][loc_mask]
            pos_target = reg_targets[loc_mask]
            pos_target_raw = reg_targets_raw[loc_mask]
            finite_pos = torch.isfinite(pos_pred).all(dim=1) & torch.isfinite(pos_target).all(dim=1)
            finite_pos &= (pos_target > 0).all(dim=1)
            if not finite_pos.any():
                continue
            pos_pred = pos_pred[finite_pos]
            pos_target = pos_target[finite_pos]
            pos_target_raw = pos_target_raw[finite_pos]

            pl, pt, pr, pb = pos_pred.unbind(-1)
            tl, tt, tr, tb = pos_target.unbind(-1)
            tl_raw, tt_raw, tr_raw, tb_raw = pos_target_raw.unbind(-1)

            pred_area = (pl + pr) * (pt + pb)
            tgt_area  = (tl + tr) * (tt + tb)

            w_inter    = torch.clamp(torch.min(pl, tl) + torch.min(pr, tr), min=0)
            h_inter    = torch.clamp(torch.min(pt, tt) + torch.min(pb, tb), min=0)
            area_inter = w_inter * h_inter
            area_union = pred_area + tgt_area - area_inter + 1e-6
            iou        = area_inter / area_union

            w_c = torch.max(pl, tl) + torch.max(pr, tr)
            h_c = torch.max(pt, tt) + torch.max(pb, tb)
            area_convex = w_c * h_c + 1e-6

            giou     = iou - (area_convex - area_union) / area_convex
            reg_loss = 1.0 - giou

            # Dynamic weight: objek kecil lebih diutamakan
            tgt_area_raw = (tl_raw + tr_raw) * (tt_raw + tb_raw)
            loss_w = torch.where(tgt_area_raw < 1024.0, 5.0,
                     torch.where(tgt_area_raw < 4096.0, 2.0, 1.0))
            reg_losses.append((reg_loss * loss_w).sum())

            # Centerness
            lr = torch.stack([tl, tr], dim=-1)
            tb_ = torch.stack([tt, tb], dim=-1)
            ctr_target = torch.sqrt(
                (lr.min(-1)[0]  / lr.max(-1)[0].clamp(min=1e-6)) *
                (tb_.min(-1)[0] / tb_.max(-1)[0].clamp(min=1e-6))
            )
            ctr_target = torch.nan_to_num(ctr_target, nan=0.0, posinf=0.0, neginf=0.0)
            pos_ctr_pred = centerness[i][loc_mask].squeeze(-1)
            pos_ctr_pred = pos_ctr_pred[finite_pos]
            ctr_loss = F.binary_cross_entropy_with_logits(
                pos_ctr_pred, ctr_target, reduction='none'
            )
            ctr_losses.append((ctr_loss * loss_w).sum())

        # ── 3. Normalisasi ─────────────────────────────────────────────────────
        num_pos   = max(1.0, float(num_positives))
        total_cls = sum(cls_losses) / num_pos
        total_reg = sum(reg_losses) / num_pos if reg_losses else logits.sum() * 0.0
        total_ctr = sum(ctr_losses) / num_pos if ctr_losses else centerness.sum() * 0.0

        class_w, bbox_w, obj_w = self._get_loss_weights()
        total_loss = class_w * total_cls + bbox_w * total_reg + obj_w * total_ctr

        return {
            'total_loss': total_loss,
            'class_loss': total_cls,
            'bbox_loss':  total_reg,
            'obj_loss':   total_ctr,
            'class_weight': torch.tensor(class_w, device=device),
            'bbox_weight': torch.tensor(bbox_w, device=device),
            'obj_weight': torch.tensor(obj_w, device=device),
        }
