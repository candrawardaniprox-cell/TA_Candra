"""
Evaluation metrics untuk object detection.

PERBAIKAN vs versi lama:
- Bug #4 FIXED: calculate_iou_batch() sekarang menerima format xyxy (prediksi dari
  model) dan xywh-center (GT dari dataset) secara terpisah dan mengonversinya dengan
  benar sebelum menghitung IoU.
- calculate_precision_recall() dan calculate_ap() disesuaikan agar format konsisten.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Format helpers ─────────────────────────────────────────────────────────────

def _cxywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Konversi (cx, cy, w, h) → (x1, y1, x2, y2)."""
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
    return out


def _xyxy_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area dari boxes format xyxy."""
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * \
           (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


# ── IoU ────────────────────────────────────────────────────────────────────────

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    IoU antara dua box.
    Input format: (cx, cy, w, h) — format GT dataset.
    """
    b1 = _cxywh_to_xyxy(box1)
    b2 = _cxywh_to_xyxy(box2)

    ix1 = max(b1[0].item(), b2[0].item())
    iy1 = max(b1[1].item(), b2[1].item())
    ix2 = min(b1[2].item(), b2[2].item())
    iy2 = min(b1[3].item(), b2[3].item())

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area1 = (b1[2] - b1[0]).item() * (b1[3] - b1[1]).item()
    area2 = (b2[2] - b2[0]).item() * (b2[3] - b2[1]).item()
    union = area1 + area2 - inter + 1e-6
    return inter / union


def calculate_iou_batch(
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes_cxywh:  torch.Tensor,
) -> torch.Tensor:
    """
    Hitung IoU antara prediksi dan GT.

    BUG #4 FIX:
    - pred_boxes_xyxy : (N, 4) format x1,y1,x2,y2  ← output dari model.get_detections()
    - gt_boxes_cxywh  : (M, 4) format cx,cy,w,h    ← format GT di dataset

    Returns:
        iou_matrix : (N, M)
    """
    if pred_boxes_xyxy.numel() == 0 or gt_boxes_cxywh.numel() == 0:
        return torch.zeros(
            (len(pred_boxes_xyxy), len(gt_boxes_cxywh)),
            device=pred_boxes_xyxy.device,
        )

    # Konversi GT ke xyxy agar bisa dibandingkan dengan prediksi
    gt_xyxy = _cxywh_to_xyxy(gt_boxes_cxywh)  # (M, 4)

    # Intersection
    inter_x1 = torch.max(pred_boxes_xyxy[:, None, 0], gt_xyxy[None, :, 0])
    inter_y1 = torch.max(pred_boxes_xyxy[:, None, 1], gt_xyxy[None, :, 1])
    inter_x2 = torch.min(pred_boxes_xyxy[:, None, 2], gt_xyxy[None, :, 2])
    inter_y2 = torch.min(pred_boxes_xyxy[:, None, 3], gt_xyxy[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h  # (N, M)

    # Area masing-masing
    pred_area = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]).clamp(min=0) * \
                (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1]).clamp(min=0)  # (N,)
    gt_area   = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=0) * \
                (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=0)                  # (M,)

    union_area = pred_area[:, None] + gt_area[None, :] - inter_area + 1e-6   # (N, M)

    return inter_area / union_area


# ── Precision & Recall ─────────────────────────────────────────────────────────

def calculate_precision_recall(
    predictions: List[Dict[str, torch.Tensor]],
    targets:     List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    num_classes:   int   = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hitung precision dan recall per kelas.

    predictions[i]['boxes']   : format xyxy  (output model)
    targets[i]['boxes']       : format cxywh (GT dataset)
    """
    true_positives  = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for pred, target in zip(predictions, targets):
        pred_boxes   = pred['boxes']    # xyxy
        pred_scores  = pred['scores']
        pred_classes = pred['classes']
        target_boxes = target['boxes']  # cxywh
        target_labels = target['labels']

        for class_id in range(num_classes):
            mask_pred   = pred_classes == class_id
            mask_target = target_labels == class_id

            boxes_pred   = pred_boxes[mask_pred]
            scores_pred  = pred_scores[mask_pred]
            boxes_target = target_boxes[mask_target]

            false_negatives[class_id] += len(boxes_target)

            if len(boxes_pred) == 0:
                continue
            if len(boxes_target) == 0:
                false_positives[class_id] += len(boxes_pred)
                continue

            # BUG #4 FIX: kirim pred=xyxy, gt=cxywh ke fungsi yang sudah benar
            iou_matrix = calculate_iou_batch(boxes_pred, boxes_target)  # (P, T)
            sorted_idx = torch.argsort(scores_pred, descending=True)
            matched    = set()

            for idx in sorted_idx:
                ious           = iou_matrix[idx]
                max_iou, max_j = ious.max(dim=0)
                if max_iou >= iou_threshold and max_j.item() not in matched:
                    true_positives[class_id]  += 1
                    false_negatives[class_id] -= 1
                    matched.add(max_j.item())
                else:
                    false_positives[class_id] += 1

    precision = np.zeros(num_classes)
    recall    = np.zeros(num_classes)
    for c in range(num_classes):
        tp = true_positives[c]
        fp = false_positives[c]
        fn = false_negatives[c]
        if tp + fp > 0:
            precision[c] = tp / (tp + fp)
        if tp + fn > 0:
            recall[c]    = tp / (tp + fn)

    return precision, recall


# ── Average Precision ──────────────────────────────────────────────────────────

def calculate_ap(
    predictions:   List[Dict[str, torch.Tensor]],
    targets:       List[Dict[str, torch.Tensor]],
    class_id:      int,
    iou_threshold: float = 0.5,
) -> float:
    """AP untuk satu kelas. pred boxes=xyxy, gt boxes=cxywh."""
    all_pred_boxes  = []
    all_pred_scores = []
    all_image_ids   = []
    all_target_data = []

    for img_id, (pred, target) in enumerate(zip(predictions, targets)):
        mask_pred = pred['classes'] == class_id
        if mask_pred.any():
            all_pred_boxes.append(pred['boxes'][mask_pred])
            all_pred_scores.append(pred['scores'][mask_pred])
            all_image_ids.extend([img_id] * mask_pred.sum().item())

        mask_target = target['labels'] == class_id
        if mask_target.any():
            all_target_data.append((img_id, target['boxes'][mask_target]))

    if not all_pred_boxes or not all_target_data:
        return 0.0

    all_pred_boxes  = torch.cat(all_pred_boxes,  dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)

    sorted_idx      = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes  = all_pred_boxes[sorted_idx]
    all_pred_scores = all_pred_scores[sorted_idx]
    all_image_ids   = [all_image_ids[i] for i in sorted_idx.cpu().numpy()]

    target_lookup  = {iid: boxes for iid, boxes in all_target_data}
    total_targets  = sum(len(b) for _, b in all_target_data)
    matched_targets = defaultdict(set)

    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))

    for i, (pred_box, img_id) in enumerate(zip(all_pred_boxes, all_image_ids)):
        if img_id not in target_lookup:
            fp[i] = 1
            continue

        gt_boxes_cxywh = target_lookup[img_id]
        # BUG #4 FIX: pred=xyxy, gt=cxywh
        ious    = calculate_iou_batch(pred_box.unsqueeze(0), gt_boxes_cxywh).squeeze(0)
        max_iou, max_j = ious.max(dim=0)

        if max_iou >= iou_threshold and max_j.item() not in matched_targets[img_id]:
            tp[i] = 1
            matched_targets[img_id].add(max_j.item())
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls    = tp_cum / (total_targets + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    recalls    = np.concatenate([[0], recalls,    [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return float(ap)


# ── mAP ───────────────────────────────────────────────────────────────────────

def calculate_map(
    predictions:    List[Dict[str, torch.Tensor]],
    targets:        List[Dict[str, torch.Tensor]],
    num_classes:    int         = 80,
    iou_thresholds: List[float] = [0.5],
) -> Dict[str, float]:
    results = {}
    for iou_threshold in iou_thresholds:
        aps = []
        for class_id in range(num_classes):
            ap = calculate_ap(predictions, targets, class_id, iou_threshold)
            aps.append(ap)
            results[f'AP@{iou_threshold:.2f}_class_{class_id}'] = ap
        results[f'mAP@{iou_threshold:.2f}'] = float(np.mean(aps))

    if len(iou_thresholds) > 1:
        results['mAP@[0.5:0.95]'] = float(
            np.mean([results[f'mAP@{t:.2f}'] for t in iou_thresholds])
        )
    return results


def _label_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int32)
    if labels.size == 0:
        return counts

    labels = labels.astype(np.int64, copy=False)
    valid = (labels >= 0) & (labels < num_classes)
    if valid.any():
        counts += np.bincount(labels[valid], minlength=num_classes).astype(np.int32)
    return counts


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    """Pembagian aman yang mengembalikan 0 saat denominator = 0."""
    numerator_arr = np.asarray(numerator, dtype=np.float64)
    denominator_arr = np.asarray(denominator, dtype=np.float64)
    out = np.zeros_like(numerator_arr, dtype=np.float64)
    valid = denominator_arr != 0
    np.divide(numerator_arr, denominator_arr, out=out, where=valid)
    if np.isscalar(numerator) and np.isscalar(denominator):
        return float(out.item())
    return out


def _compute_one_vs_all_stats(cm: np.ndarray, num_classes: int, beta: float = 1.0) -> Dict[str, np.ndarray]:
    """Hitung TP/FP/FN/TN dan metrik one-vs-all per kelas dari confusion matrix."""
    summary = summarize_confusion_matrix(cm, num_classes)
    tp = summary['tp']
    fp = summary['fp']
    fn = summary['fn']
    tn = summary['tn']
    total = tp + fp + fn + tn
    beta_sq = float(beta) ** 2

    accuracy = _safe_divide(tp + tn, total)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    fscore = _safe_divide((beta_sq + 1.0) * tp, (beta_sq + 1.0) * tp + beta_sq * fn + fp)
    specificity = _safe_divide(tn, fp + tn)
    error_rate = _safe_divide(fp + fn, total)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'specificity': specificity,
        'error_rate': error_rate,
    }


def build_class_confusion_matrix(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
) -> np.ndarray:
    """Confusion matrix berbasis jumlah kelas per gambar tanpa IoU."""
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    bg = num_classes

    for pred, tgt in zip(predictions, targets):
        pred_labels = pred['classes'].cpu().numpy() if isinstance(pred['classes'], torch.Tensor) else np.asarray(pred['classes'])
        gt_labels = tgt['labels'].cpu().numpy() if isinstance(tgt['labels'], torch.Tensor) else np.asarray(tgt['labels'])

        pred_counts = _label_counts(pred_labels, num_classes)
        gt_counts = _label_counts(gt_labels, num_classes)

        same_class = np.minimum(pred_counts, gt_counts)
        for cls_id in range(num_classes):
            cm[cls_id, cls_id] += int(same_class[cls_id])

        pred_remaining = pred_counts - same_class
        gt_remaining = gt_counts - same_class

        for gt_cls in range(num_classes):
            if gt_remaining[gt_cls] <= 0:
                continue

            for pred_cls in range(num_classes):
                if pred_cls == gt_cls or pred_remaining[pred_cls] <= 0:
                    continue

                moved = min(gt_remaining[gt_cls], pred_remaining[pred_cls])
                if moved <= 0:
                    continue

                cm[gt_cls, pred_cls] += int(moved)
                gt_remaining[gt_cls] -= moved
                pred_remaining[pred_cls] -= moved

                if gt_remaining[gt_cls] == 0:
                    break

        for gt_cls in range(num_classes):
            if gt_remaining[gt_cls] > 0:
                cm[gt_cls, bg] += int(gt_remaining[gt_cls])

        for pred_cls in range(num_classes):
            if pred_remaining[pred_cls] > 0:
                cm[bg, pred_cls] += int(pred_remaining[pred_cls])

    return cm


def calculate_classification_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
) -> Dict[str, object]:
    """Hitung Accuracy, Precision, Recall, dan F1 berbasis confusion matrix kelas."""
    cm = build_class_confusion_matrix(predictions, targets, num_classes)
    stats = _compute_one_vs_all_stats(cm, num_classes, beta=1.0)
    total = float(cm.sum())
    accuracy = stats['accuracy']
    precision = stats['precision']
    recall = stats['recall']
    f1 = stats['fscore']

    tp_total = float(sum(cm[c, c] for c in range(num_classes)))
    fp_total = float(sum(cm[:, c].sum() - cm[c, c] for c in range(num_classes)))
    fn_total = float(sum(cm[c, :].sum() - cm[c, c] for c in range(num_classes)))

    total_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    total_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0 else 0.0
    )
    total_accuracy = tp_total / total if total > 0 else 0.0

    return {
        'confusion_matrix': cm,
        'accuracy_per_class': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'accuracy_total': float(total_accuracy),
        'precision_total': float(total_precision),
        'recall_total': float(total_recall),
        'f1_total': float(total_f1),
    }


def calculate_multiclass_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    beta: float = 1.0,
) -> Dict[str, object]:
    """
    Hitung metrik multi-kelas sesuai rumus paper.

    Rumus yang dipakai:
    - Average Accuracy
    - Error Rate
    - Precision / Recall / F-score Micro
    - Precision / Recall / F-score Macro
    - Specificity per kelas
    """
    cm = build_class_confusion_matrix(predictions, targets, num_classes)
    stats = _compute_one_vs_all_stats(cm, num_classes, beta=beta)
    tp = stats['tp']
    fp = stats['fp']
    fn = stats['fn']
    tn = stats['tn']
    beta_sq = float(beta) ** 2

    precision_micro = _safe_divide(tp.sum(), (tp + fp).sum())
    recall_micro = _safe_divide(tp.sum(), (tp + fn).sum())
    f1_micro = _safe_divide(
        (beta_sq + 1.0) * precision_micro * recall_micro,
        beta_sq * precision_micro + recall_micro,
    )

    precision_macro = float(np.mean(stats['precision']))
    recall_macro = float(np.mean(stats['recall']))
    f1_macro = _safe_divide(
        (beta_sq + 1.0) * precision_macro * recall_macro,
        beta_sq * precision_macro + recall_macro,
    )

    return {
        'multiclass_confusion_matrix': cm,
        'multi_accuracy_per_class': stats['accuracy'],
        'multi_precision_per_class': stats['precision'],
        'multi_recall_per_class': stats['recall'],
        'multi_f1_per_class': stats['fscore'],
        'multi_specificity_per_class': stats['specificity'],
        'average_accuracy': float(np.mean(stats['accuracy'])),
        'error_rate': float(np.mean(stats['error_rate'])),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'specificity_macro': float(np.mean(stats['specificity'])),
    }


def summarize_confusion_matrix(cm: np.ndarray, num_classes: int) -> Dict[str, np.ndarray]:
    """Ringkas confusion matrix menjadi TP/FP/FN/TN per kelas."""
    total = float(cm.sum())
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    tn = np.zeros(num_classes, dtype=np.float64)

    for cls_id in range(num_classes):
        tp[cls_id] = float(cm[cls_id, cls_id])
        fp[cls_id] = float(cm[:, cls_id].sum() - tp[cls_id])
        fn[cls_id] = float(cm[cls_id, :].sum() - tp[cls_id])
        tn[cls_id] = float(total - tp[cls_id] - fp[cls_id] - fn[cls_id])

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total': total,
    }


def _plot_confusion_matrix_with_summary(
    cm: np.ndarray,
    num_classes: int,
    classes: List[str],
    title: str,
    fname: str,
):
    """Simpan heatmap confusion matrix + ringkasan TP/FP/FN/TN per kelas."""
    summary = summarize_confusion_matrix(cm, num_classes)

    fig, (ax_cm, ax_text) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={'width_ratios': [3.6, 1.9]},
    )

    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                str(int(cm[i, j])),
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=11,
                fontweight='bold',
            )

    ax_cm.set_ylabel('Ground Truth')
    ax_cm.set_xlabel('Prediction')

    ax_text.axis('off')
    lines = ["Ringkasan TP / FP / FN / TN", ""]
    for cls_id in range(num_classes):
        lines.extend([
            f"{classes[cls_id]}",
            f"TP: {int(summary['tp'][cls_id])}",
            f"FP: {int(summary['fp'][cls_id])}",
            f"FN: {int(summary['fn'][cls_id])}",
            f"TN: {int(summary['tn'][cls_id])}",
            "",
        ])

    lines.extend([
        "Catatan:",
        f"Baris {classes[-1]} -> false positive",
        f"Kolom {classes[-1]} -> false negative",
    ])

    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        va='top',
        ha='left',
        fontsize=10,
        family='monospace',
    )

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_confusion_matrix(
    predictions: List[Dict[str, torch.Tensor]],
    targets:     List[Dict[str, torch.Tensor]],
    num_classes: int,
    class_names: list = None,
    fname: str = None,
    save_fig: bool = True,
):
    """Buat confusion matrix berbasis jumlah kelas per gambar (tanpa IoU).

    Baris = ground truth, Kolom = prediction.
    Index terakhir (num_classes) = Background / unmatched.
    Mengembalikan matriks numpy `cm`.
    Jika `save_fig` dan `fname` diberikan, simpan gambar heatmap ke fname.
    """
    cm = build_class_confusion_matrix(predictions, targets, num_classes)

    if save_fig and fname:
        classes = (class_names[:num_classes] if class_names is not None else [f'C{i}' for i in range(num_classes)]) + ['Background']
        _plot_confusion_matrix_with_summary(
            cm=cm,
            num_classes=num_classes,
            classes=classes,
            title='Confusion Matrix Berbasis Jumlah Kelas',
            fname=fname,
        )

    return cm


def generate_detection_confusion_matrix(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    class_names: list = None,
    iou_threshold: float = 0.5,
    fname: str = None,
    save_fig: bool = True,
):
    """Confusion matrix detection berbasis IoU one-to-one.

    Baris = ground truth, kolom = prediction.
    Index terakhir = Background / unmatched.
    """
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    bg = num_classes

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred['boxes'] if isinstance(pred['boxes'], torch.Tensor) else torch.as_tensor(pred['boxes'])
        pred_classes = pred['classes'] if isinstance(pred['classes'], torch.Tensor) else torch.as_tensor(pred['classes'])
        pred_scores = pred['scores'] if isinstance(pred['scores'], torch.Tensor) else torch.as_tensor(pred['scores'])
        tgt_boxes = tgt['boxes'] if isinstance(tgt['boxes'], torch.Tensor) else torch.as_tensor(tgt['boxes'])
        tgt_labels = tgt['labels'] if isinstance(tgt['labels'], torch.Tensor) else torch.as_tensor(tgt['labels'])

        pred_boxes = pred_boxes.detach().cpu()
        pred_classes = pred_classes.detach().cpu().long()
        pred_scores = pred_scores.detach().cpu()
        tgt_boxes = tgt_boxes.detach().cpu()
        tgt_labels = tgt_labels.detach().cpu().long()

        if len(tgt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(tgt_boxes) == 0:
            for cls_id in pred_classes.tolist():
                cm[bg, int(cls_id)] += 1
            continue
        if len(pred_boxes) == 0:
            for cls_id in tgt_labels.tolist():
                cm[int(cls_id), bg] += 1
            continue

        iou_matrix = calculate_iou_batch(pred_boxes, tgt_boxes)
        order = torch.argsort(pred_scores, descending=True)
        matched = set()

        for pred_idx in order.tolist():
            best_iou = 0.0
            best_gt = -1
            for gt_idx in range(len(tgt_boxes)):
                if gt_idx in matched:
                    continue
                iou_value = float(iou_matrix[pred_idx, gt_idx].item())
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt = gt_idx

            pred_cls = int(pred_classes[pred_idx].item())
            if best_iou >= iou_threshold and best_gt >= 0:
                gt_cls = int(tgt_labels[best_gt].item())
                cm[gt_cls, pred_cls] += 1
                matched.add(best_gt)
            else:
                cm[bg, pred_cls] += 1

        for gt_idx, gt_cls in enumerate(tgt_labels.tolist()):
            if gt_idx not in matched:
                cm[int(gt_cls), bg] += 1

    if save_fig and fname:
        classes = (class_names[:num_classes] if class_names is not None else [f'C{i}' for i in range(num_classes)]) + ['Background']
        _plot_confusion_matrix_with_summary(
            cm=cm,
            num_classes=num_classes,
            classes=classes,
            title=f'Detection Confusion Matrix @ IoU {iou_threshold:.2f}',
            fname=fname,
        )

    return cm
