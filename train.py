"""
train.py - Training script untuk Hybrid CNN-Transformer object detection.

Versi ini memakai evaluasi confusion matrix berbasis jumlah kelas per gambar:
- Mengabaikan IoU/bounding-box matching untuk confusion matrix dan metrik kelas.
- Fokus ke Accuracy, Precision, Recall, dan F1-Score.
- Menjaga output visual bbox agar warna kelas konsisten antara GT dan prediksi.
"""

import argparse
import gc
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

matplotlib.use('Agg')

from config import Config
from data import (
    ObjectDetectionDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
)
from models import HybridDetector
from utils import AnchorFreeLoss, calculate_map
from utils.metrics_fixed import (
    calculate_classification_metrics,
    generate_confusion_matrix,
    generate_detection_confusion_matrix,
)
from utils.visualization import draw_bounding_boxes
from train_classifier import train_classifier as train_stage2_classifier

MAP_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
METRIC_ROW_ORDER = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1-Score',
    'mAP@0.50',
    'mAP@[0.50:0.95]',
]


def sanitize_targets(targets, num_classes, image_size, logger=None, batch_name=None):
    """
    Bersihkan target batch di CPU sebelum dipakai model/loss.

    Tujuan:
    - mencegah label out-of-range dipakai sebagai index CUDA
    - membuang bbox NaN/Inf atau ukuran <= 0
    - merapikan shape tensor yang tidak sesuai
    """
    clean_boxes = []
    clean_labels = []
    skipped = []

    for idx, (boxes, labels) in enumerate(zip(targets['boxes'], targets['labels'])):
        image_id = targets.get('image_ids', [None] * len(targets['boxes']))[idx]

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = boxes.detach().cpu().float()

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        else:
            labels = labels.detach().cpu().long()

        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
            labels = labels.reshape(0)
            clean_boxes.append(boxes)
            clean_labels.append(labels)
            continue

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            skipped.append(f"image_id={image_id} shape_boxes={tuple(boxes.shape)}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            clean_boxes.append(boxes)
            clean_labels.append(labels)
            continue

        labels = labels.reshape(-1)
        pair_count = min(boxes.shape[0], labels.shape[0])
        if pair_count != boxes.shape[0] or pair_count != labels.shape[0]:
            skipped.append(
                f"image_id={image_id} mismatch_boxes={boxes.shape[0]} mismatch_labels={labels.shape[0]}"
            )
        boxes = boxes[:pair_count]
        labels = labels[:pair_count]

        if pair_count == 0:
            clean_boxes.append(torch.zeros((0, 4), dtype=torch.float32))
            clean_labels.append(torch.zeros((0,), dtype=torch.long))
            continue

        valid = torch.isfinite(boxes).all(dim=1)
        valid &= torch.isfinite(labels.float())
        valid &= boxes[:, 2] > 1e-6
        valid &= boxes[:, 3] > 1e-6
        valid &= labels >= 0
        valid &= labels < num_classes

        removed = int((~valid).sum().item())
        if removed:
            skipped.append(f"image_id={image_id} removed={removed}")

        boxes = boxes[valid]
        labels = labels[valid]

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes[:, 0] = boxes[:, 0].clamp(0.0, float(image_size))
            boxes[:, 1] = boxes[:, 1].clamp(0.0, float(image_size))
            boxes[:, 2] = boxes[:, 2].clamp(1e-6, float(image_size))
            boxes[:, 3] = boxes[:, 3].clamp(1e-6, float(image_size))
            labels = labels.long()

        clean_boxes.append(boxes)
        clean_labels.append(labels)

    if skipped and logger is not None:
        prefix = f"{batch_name} | " if batch_name else ""
        logger.warning("%sTarget sanitization: %s", prefix, "; ".join(skipped[:8]))

    return {
        'boxes': clean_boxes,
        'labels': clean_labels,
        'image_ids': list(targets.get('image_ids', [])),
    }


def cleanup_cuda_after_error():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def setup_run(resume_checkpoint: Path = None) -> Path:
    Config.create_directories()
    if resume_checkpoint is not None:
        run_dir = resume_checkpoint.resolve().parent.parent
    else:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Config.BASE_OUTPUT_DIR / run_name
    Config.setup_run_dirs(run_dir)
    return run_dir


def resolve_resume_checkpoint(resume_arg: str) -> Path | None:
    if not resume_arg:
        return None

    path = Path(resume_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Resume path tidak ditemukan: {path}")

    ckpt_dir = path / "checkpoints" if (path / "checkpoints").exists() else path

    latest_ckpt = ckpt_dir / "latest_checkpoint.pth"
    if latest_ckpt.exists():
        return latest_ckpt

    epoch_ckpts = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if epoch_ckpts:
        return epoch_ckpts[-1]

    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        return best_ckpt

    raise FileNotFoundError(f"Tidak menemukan checkpoint resume di: {ckpt_dir}")


def resolve_evaluation_checkpoint(checkpoint_arg: str) -> Path | None:
    if not checkpoint_arg:
        return None

    path = Path(checkpoint_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint evaluasi tidak ditemukan: {path}")

    ckpt_dir = path / "checkpoints" if (path / "checkpoints").exists() else path

    best_ckpt = ckpt_dir / "best_model.pth"
    if best_ckpt.exists():
        return best_ckpt

    latest_ckpt = ckpt_dir / "latest_checkpoint.pth"
    if latest_ckpt.exists():
        return latest_ckpt

    epoch_ckpts = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if epoch_ckpts:
        return epoch_ckpts[-1]

    raise FileNotFoundError(f"Tidak menemukan checkpoint evaluasi di: {ckpt_dir}")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('object_detection')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    log_file = Config.LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_time(seconds: float) -> str:
    return f"{int(seconds) // 60}m {int(seconds) % 60:02d}s"


def get_realtime_elapsed(session_start_time: float | None, elapsed_offset: float = 0.0) -> float:
    if session_start_time is None:
        return float(elapsed_offset)
    return max(0.0, float(elapsed_offset) + (time.time() - session_start_time))


def init_metric_bundle(num_classes):
    zeros = [0.0] * num_classes
    return {
        metric_name: {
            'per_class': list(zeros),
            'global': 0.0,
        }
        for metric_name in METRIC_ROW_ORDER
    }


def extract_metric_bundle(metrics, num_classes):
    map50_per_class = [
        float(metrics.get(f'AP@0.50_class_{class_id}', 0.0))
        for class_id in range(num_classes)
    ]
    map5095_per_class = []
    for class_id in range(num_classes):
        ap_values = [
            float(metrics.get(f'AP@{thr:.2f}_class_{class_id}', 0.0))
            for thr in MAP_IOU_THRESHOLDS
        ]
        map5095_per_class.append(float(np.mean(ap_values)) if ap_values else 0.0)

    return {
        'Accuracy': {
            'per_class': list(metrics.get('accuracy_per_class', np.zeros(num_classes))[:num_classes]),
            'global': float(metrics.get('accuracy_total', 0.0)),
        },
        'Precision': {
            'per_class': list(metrics.get('precision_per_class', np.zeros(num_classes))[:num_classes]),
            'global': float(metrics.get('precision_total', 0.0)),
        },
        'Recall': {
            'per_class': list(metrics.get('recall_per_class', np.zeros(num_classes))[:num_classes]),
            'global': float(metrics.get('recall_total', 0.0)),
        },
        'F1-Score': {
            'per_class': list(metrics.get('f1_per_class', np.zeros(num_classes))[:num_classes]),
            'global': float(metrics.get('f1_total', 0.0)),
        },
        'mAP@0.50': {
            'per_class': map50_per_class,
            'global': float(metrics.get('mAP@0.50', 0.0)),
        },
        'mAP@[0.50:0.95]': {
            'per_class': map5095_per_class,
            'global': float(metrics.get('mAP@[0.5:0.95]', 0.0)),
        },
    }


def update_best_metric_bundle(best_bundle, current_bundle):
    for metric_name in METRIC_ROW_ORDER:
        best_bundle[metric_name]['per_class'] = [
            max(best_value, current_value)
            for best_value, current_value in zip(
                best_bundle[metric_name]['per_class'],
                current_bundle[metric_name]['per_class'],
            )
        ]
        best_bundle[metric_name]['global'] = max(
            best_bundle[metric_name]['global'],
            current_bundle[metric_name]['global'],
        )


def _table_border(label_width, value_width, num_value_cols, fill='-'):
    return "  +" + fill * (label_width + 2) + "+" + "+".join(
        fill * (value_width + 2) for _ in range(num_value_cols)
    ) + "+"


def _table_row(label_width, value_width, label, values, global_value):
    items = list(values) + [global_value]
    cells = " | ".join(f"{item:>{value_width}.4f}" for item in items)
    return f"  | {label:<{label_width}} | {cells} |"


def log_metric_table(logger, title, class_names, metric_bundle):
    label_width = 26
    value_width = max(12, max(len(name) for name in class_names + ['GLOBAL']) + 2)
    top = _table_border(label_width, value_width, len(class_names) + 1, '=')
    mid = _table_border(label_width, value_width, len(class_names) + 1, '-')
    header = (
        f"  | {'METRIK':<{label_width}} | "
        + " | ".join(f"{name.upper():>{value_width}}" for name in class_names)
        + f" | {'GLOBAL':>{value_width}} |"
    )

    logger.info(top)
    logger.info(f"  | {title:^{label_width + (value_width + 3) * (len(class_names) + 1) + 1}}|")
    logger.info(top)
    logger.info(header)
    logger.info(mid)
    for metric_name in METRIC_ROW_ORDER:
        logger.info(
            _table_row(
                label_width,
                value_width,
                metric_name,
                metric_bundle[metric_name]['per_class'],
                metric_bundle[metric_name]['global'],
            )
        )
    logger.info(top)


def count_bbox_per_class(dataset, num_classes):
    counts = {i: 0 for i in range(num_classes)}
    total = 0
    for anns in dataset.image_annotations.values():
        for ann in anns:
            if 'bbox' in ann and ann.get('area', 0) > 0:
                idx = dataset.category_id_to_idx[ann['category_id']]
                if idx < num_classes:
                    counts[idx] += 1
                    total += 1
    return counts, total


def print_dataset_summary(logger, train_ds, val_ds, test_ds, num_classes, cls_names):
    sep = "=" * 70
    tc, tt = count_bbox_per_class(train_ds, num_classes)
    vc, vt = count_bbox_per_class(val_ds, num_classes)
    ec, et = (
        count_bbox_per_class(test_ds, num_classes)
        if test_ds else ({i: 0 for i in range(num_classes)}, 0)
    )

    logger.info(sep)
    logger.info("  RINGKASAN DATASET")
    logger.info(sep)
    logger.info(f"  {'Split':<10} {'Gambar':>10} {'BBox Total':>12}")
    logger.info(f"  {'Train':<10} {len(train_ds.image_ids):>10,} {tt:>12,}")
    logger.info(f"  {'Val':<10} {len(val_ds.image_ids):>10,} {vt:>12,}")
    logger.info(f"  {'Test':<10} {len(test_ds.image_ids) if test_ds else 0:>10,} {et:>12,}")
    logger.info(sep)
    logger.info("  BBOX PER KELAS")
    logger.info(f"  {'Kelas':<15} {'Train':>8} {'Val':>8} {'Test':>8}")
    for i in range(num_classes):
        name = cls_names[i].upper() if i < len(cls_names) else f"CLASS_{i}"
        logger.info(f"  {name:<15} {tc[i]:>8,} {vc[i]:>8,} {ec[i]:>8,}")
    logger.info(sep + "\n")


def print_model_config(logger, model, num_classes, image_size, batch_size, lr, epochs, device, use_amp):
    sep = "=" * 70

    def num_params(module):
        return sum(p.numel() for p in module.parameters())

    logger.info(sep)
    logger.info("  KONFIGURASI MODEL & TRAINING")
    logger.info(sep)
    logger.info(f"  Image Size        : {image_size}x{image_size}")
    logger.info(f"  Num Classes       : {num_classes}")
    logger.info(f"  Batch Size        : {batch_size}")
    logger.info(f"  Learning Rate     : {lr}")
    logger.info(f"  Epochs            : {epochs}")
    logger.info(f"  Optimize          : AdamW")
    logger.info(f"  Weight Decay      : {getattr(Config, 'WEIGHT_DECAY', 0.0)}")
    logger.info(f"  Scheduler         : {getattr(Config, 'LR_SCHEDULER', 'none')}")
    logger.info(f"  Warmup Epochs     : {getattr(Config, 'WARMUP_EPOCHS', 0)}")
    logger.info(f"  Grad Clip Norm    : {getattr(Config, 'GRAD_CLIP_NORM', 0.0)}")
    logger.info(f"  Device            : {device}")
    logger.info(f"  Mixed Precision   : {use_amp}")
    logger.info(f"  cuDNN Benchmark   : {getattr(Config, 'CUDA_BENCHMARK', False)}")
    logger.info(f"  TF32              : {getattr(Config, 'ALLOW_TF32', False)}")
    logger.info(f"  Class Priority    : {getattr(Config, 'CLASS_PRIORITY_MODE', False)}")
    logger.info(f"  Loss Objective    : class-oriented untuk semua epoch")
    logger.info(f"  Loss Weights      : cls={Config.LAMBDA_CLASS:.2f} | bbox={Config.LAMBDA_BBOX:.2f} | obj={Config.LAMBDA_OBJ:.2f}")
    logger.info(f"  Loss Type         : bbox={getattr(Config, 'BBOX_LOSS_TYPE', 'unknown')} | focal={getattr(Config, 'USE_FOCAL_LOSS', False)}")
    logger.info(f"  Focal Params      : alpha={getattr(Config, 'FOCAL_ALPHA', 0.0)} | gamma={getattr(Config, 'FOCAL_GAMMA', 0.0)}")
    logger.info(f"  IoU Assign        : pos={getattr(Config, 'IOU_THRESHOLD_POS', 0.0)} | neg={getattr(Config, 'IOU_THRESHOLD_NEG', 0.0)}")
    logger.info(f"  Checkpoint Metric : {getattr(Config, 'CHECKPOINT_METRIC', 'mAP@0.50')}")
    logger.info(f"  Metrics           : Accuracy | Precision | Recall | F1 | mAP@0.50 | mAP@[0.50:0.95]")
    logger.info(
        f"  Inference Class   : conf={getattr(Config, 'CLASS_CONF_THRESHOLD', getattr(Config, 'CONF_THRESHOLD', 0.0))} | "
        f"nms={getattr(Config, 'CLASS_NMS_IOU_THRESHOLD', getattr(Config, 'NMS_IOU_THRESHOLD', 0.0))} | "
        f"max_det={getattr(Config, 'CLASS_MAX_DETECTIONS', getattr(Config, 'MAX_DETECTIONS', 0))}"
    )
    logger.info(
        f"  Metric Class Eval : conf={getattr(Config, 'CLASS_METRIC_CONF_THRESHOLD', getattr(Config, 'CLASS_CONF_THRESHOLD', 0.0))} | "
        f"nms={getattr(Config, 'CLASS_METRIC_NMS_IOU_THRESHOLD', getattr(Config, 'CLASS_NMS_IOU_THRESHOLD', 0.0))} | "
        f"max_det={getattr(Config, 'CLASS_METRIC_MAX_DETECTIONS', getattr(Config, 'CLASS_MAX_DETECTIONS', 0))} | "
        f"centerness={getattr(Config, 'CLASS_METRIC_USE_CENTERNESS', False)}"
    )
    logger.info(
        f"  Inference Detect  : conf={getattr(Config, 'DET_CONF_THRESHOLD', getattr(Config, 'CONF_THRESHOLD', 0.0))} | "
        f"nms={getattr(Config, 'DET_NMS_IOU_THRESHOLD', getattr(Config, 'NMS_IOU_THRESHOLD', 0.0))} | "
        f"max_det={getattr(Config, 'DET_MAX_DETECTIONS', getattr(Config, 'MAX_DETECTIONS', 0))}"
    )
    logger.info(
        f"  Score Mode        : class_cm="
        f"{'class_x_centerness' if getattr(Config, 'USE_CENTERNESS_IN_SCORE', False) else 'class_only'} | "
        f"map={'class_x_centerness' if getattr(Config, 'USE_CENTERNESS_IN_SCORE', False) else 'class_only'}"
    )
    logger.info(f"  Augment           : {getattr(Config, 'AUGMENT', False)}")
    logger.info(f"  Aug Repeat        : {getattr(Config, 'AUGMENT_REPEAT_FACTOR', 1) if getattr(Config, 'AUGMENT', False) else 1}")
    logger.info(
        f"  Dataloader        : workers={getattr(Config, 'NUM_WORKERS', 0)} | "
        f"pin_memory={getattr(Config, 'PIN_MEMORY', False)} | "
        f"persistent={getattr(Config, 'PERSISTENT_WORKERS', False)}"
    )
    logger.info(
        f"  Backbone Config   : {getattr(Config, 'BACKBONE_NAME', 'unknown')} | "
        f"pretrained={getattr(Config, 'BACKBONE_PRETRAINED', False)}"
    )
    logger.info(
        f"  Transformer       : dim={getattr(Config, 'TRANSFORMER_DIM', 0)} | "
        f"heads={getattr(Config, 'TRANSFORMER_HEADS', 0)} | "
        f"layers={getattr(Config, 'TRANSFORMER_LAYERS', 0)} | "
        f"ff={getattr(Config, 'TRANSFORMER_FF_DIM', 0)} | "
        f"dropout={getattr(Config, 'TRANSFORMER_DROPOUT', 0.0)}"
    )
    logger.info(f"  Backbone          : {num_params(model.backbone)/1e6:.2f}M")
    logger.info(
        f"  Stage P3/P4/P5    : "
        f"{num_params(model.stage_p3)/1e6:.2f}M / "
        f"{num_params(model.stage_p4)/1e6:.2f}M / "
        f"{num_params(model.stage_p5)/1e6:.2f}M"
    )
    logger.info(
        f"  FPN               : "
        f"{(num_params(model.lat_p3) + num_params(model.lat_p4) + num_params(model.smooth_p3) + num_params(model.smooth_p4))/1e6:.2f}M"
    )
    logger.info(f"  Head              : {num_params(model.detection_head)/1e6:.2f}M")
    logger.info(f"  Total Parameters  : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    logger.info(sep + "\n")


def _savefig(filename: str):
    plt.tight_layout()
    plt.savefig(Config.GRAPHS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()


def _annotate_best_point(x_values, y_values, mode='max', text='Best Model'):
    if not x_values or not y_values or len(x_values) != len(y_values):
        return

    y_array = np.asarray(y_values, dtype=float)
    if y_array.size == 0 or np.all(~np.isfinite(y_array)):
        return

    best_idx = int(np.nanargmax(y_array) if mode == 'max' else np.nanargmin(y_array))
    best_x = x_values[best_idx]
    best_y = y_values[best_idx]

    plt.scatter([best_x], [best_y], s=18, c='black', zorder=6)

    x_offset = 10 if best_idx < max(1, len(x_values) // 2) else -46
    y_offset = -16 if best_y > float(np.nanmean(y_array)) else 12
    plt.annotate(
        text,
        xy=(best_x, best_y),
        xytext=(x_offset, y_offset),
        textcoords='offset points',
        fontsize=7,
        color='black',
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=0.7,
            shrinkA=2,
            shrinkB=2,
        ),
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.2),
        zorder=7,
    )


def save_single_metric_plot(x_val, y_val, title, ylabel, fname, color, marker, mode='max'):
    if not x_val or len(x_val) != len(y_val):
        return

    plt.figure(figsize=(10, 5))
    plt.plot(x_val, y_val, color=color, marker=marker, linewidth=1.8, markersize=4)
    _annotate_best_point(x_val, y_val, mode=mode)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.7)
    _savefig(fname)


def build_dense_epoch_series(x_sparse, y_sparse):
    if not x_sparse or len(x_sparse) != len(y_sparse):
        return [], []

    if len(x_sparse) == 1:
        return list(x_sparse), list(y_sparse)

    x_dense = list(range(int(x_sparse[0]), int(x_sparse[-1]) + 1))
    y_dense = np.interp(x_dense, x_sparse, y_sparse).tolist()
    return x_dense, y_dense


def plot_dense_train_series(x_sparse, y_sparse, label, color, marker):
    if not x_sparse or len(x_sparse) != len(y_sparse):
        return [], []

    x_dense, y_dense = build_dense_epoch_series(x_sparse, y_sparse)
    if x_dense and len(x_dense) == len(y_dense):
        plt.plot(x_dense, y_dense, label=label, color=color, linewidth=1.8)
        plt.scatter(x_sparse, y_sparse, color=color, marker=marker, s=28, zorder=5)
    return x_dense, y_dense


def save_dual_metric_plot(
    x_train,
    y_train,
    x_val,
    y_val,
    title,
    ylabel,
    fname,
    train_color='royalblue',
    val_color='tomato',
    train_marker='s',
    val_marker='o',
    mode='max',
    train_sparse_x=None,
    train_sparse_y=None,
):
    if (not x_train or len(x_train) != len(y_train)) and (not x_val or len(x_val) != len(y_val)):
        return

    plt.figure(figsize=(10, 5))
    if x_train and len(x_train) == len(y_train):
        sparse_x = train_sparse_x if train_sparse_x and len(train_sparse_x) == len(train_sparse_y or []) else x_train
        sparse_y = train_sparse_y if train_sparse_y and len(train_sparse_y or []) == len(sparse_x) else y_train
        plt.plot(x_train, y_train, label='Train', color=train_color, linewidth=1.8)
        if sparse_x and len(sparse_x) == len(sparse_y):
            plt.scatter(sparse_x, sparse_y, color=train_color, marker=train_marker, s=28, zorder=5)
            _annotate_best_point(sparse_x, sparse_y, mode=mode, text='Best Train')
        else:
            _annotate_best_point(x_train, y_train, mode=mode, text='Best Train')
    if x_val and len(x_val) == len(y_val):
        plt.plot(x_val, y_val, label='Val', color=val_color, marker=val_marker, linewidth=1.8, markersize=4)
        _annotate_best_point(x_val, y_val, mode=mode, text='Best Val')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.7)
    _savefig(fname)


def annotate_metric_series(series_list):
    for x_values, y_values, mode in series_list:
        _annotate_best_point(x_values, y_values, mode=mode)


def _safe_metric_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ('_', '-') else "_" for ch in name.lower())


def save_loss_plot(x_tr, y_tr, x_val, y_val, title, ylabel, fname):
    plt.figure(figsize=(10, 5))
    if x_tr and len(x_tr) == len(y_tr):
        plt.plot(x_tr, y_tr, label='Train', color='royalblue', linewidth=1.5)
        _annotate_best_point(x_tr, y_tr, mode='min')
    if x_val and len(x_val) == len(y_val):
        plt.plot(x_val, y_val, label='Val', color='tomato', linewidth=1.5, marker='o', markersize=3)
        _annotate_best_point(x_val, y_val, mode='min')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.5)
    _savefig(fname)


def update_all_plots(
    x_tr,
    x_val,
    h_tr_loss,
    h_val_loss,
    h_tr_bbox,
    h_val_bbox,
    h_tr_cls,
    h_val_cls,
    h_tr_obj,
    h_val_obj,
    x_tr_metrics,
    h_tr_acc,
    h_tr_acc_cls,
    h_tr_prec,
    h_tr_rec,
    h_tr_f1,
    h_tr_map50,
    h_tr_map5095,
    h_acc,
    h_acc_cls,
    h_prec,
    h_rec,
    h_f1,
    h_map50,
    h_map5095,
    class_names,
    num_classes,
):
    x_tr_acc_dense, h_tr_acc_dense = build_dense_epoch_series(x_tr_metrics, h_tr_acc)
    x_tr_prec_dense, h_tr_prec_dense = build_dense_epoch_series(x_tr_metrics, h_tr_prec)
    x_tr_rec_dense, h_tr_rec_dense = build_dense_epoch_series(x_tr_metrics, h_tr_rec)
    x_tr_f1_dense, h_tr_f1_dense = build_dense_epoch_series(x_tr_metrics, h_tr_f1)
    x_tr_map50_dense, h_tr_map50_dense = build_dense_epoch_series(x_tr_metrics, h_tr_map50)
    x_tr_map5095_dense, h_tr_map5095_dense = build_dense_epoch_series(x_tr_metrics, h_tr_map5095)

    save_loss_plot(x_tr, h_tr_loss, x_val, h_val_loss, 'Total Loss', 'Loss', 'loss_total.png')
    save_loss_plot(x_tr, h_tr_bbox, x_val, h_val_bbox, 'BBox Regression Loss', 'Loss', 'loss_bbox.png')
    save_loss_plot(x_tr, h_tr_cls, x_val, h_val_cls, 'Classification Loss', 'Loss', 'loss_cls.png')
    save_loss_plot(x_tr, h_tr_obj, x_val, h_val_obj, 'Objectness/Centerness', 'Loss', 'loss_obj.png')

    if x_val and len(x_val) == len(h_acc):
        plt.figure(figsize=(10, 5))
        if x_tr_acc_dense and len(x_tr_acc_dense) == len(h_tr_acc_dense):
            plot_dense_train_series(x_tr_metrics, h_tr_acc, 'Train Accuracy', 'green', 's')
        plt.plot(x_val, h_acc, label='Val Accuracy', color='limegreen', marker='s')
        if x_tr_prec_dense and len(x_tr_prec_dense) == len(h_tr_prec_dense):
            plot_dense_train_series(x_tr_metrics, h_tr_prec, 'Train Precision', 'royalblue', 'o')
        plt.plot(x_val, h_prec, label='Val Precision', color='cornflowerblue', marker='o')
        if x_tr_rec_dense and len(x_tr_rec_dense) == len(h_tr_rec_dense):
            plot_dense_train_series(x_tr_metrics, h_tr_rec, 'Train Recall', 'darkorange', 'D')
        plt.plot(x_val, h_rec, label='Val Recall', color='orange', marker='D')
        if x_tr_f1_dense and len(x_tr_f1_dense) == len(h_tr_f1_dense):
            plot_dense_train_series(x_tr_metrics, h_tr_f1, 'Train F1-Score', 'purple', '^')
        plt.plot(x_val, h_f1, label='Val F1-Score', color='mediumpurple', marker='^')
        annotate_metric_series([
            (x_tr_metrics, h_tr_acc, 'max'),
            (x_val, h_acc, 'max'),
            (x_tr_metrics, h_tr_prec, 'max'),
            (x_val, h_prec, 'max'),
            (x_tr_metrics, h_tr_rec, 'max'),
            (x_val, h_rec, 'max'),
            (x_tr_metrics, h_tr_f1, 'max'),
            (x_val, h_f1, 'max'),
        ])
        plt.title('Train vs Val Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.7)
        _savefig('classification_metrics_graph.png')

        save_dual_metric_plot(x_tr_acc_dense, h_tr_acc_dense, x_val, h_acc, 'Train vs Val Accuracy', 'Accuracy', 'accuracy_graph.png', 'green', 'limegreen', 's', 'o', train_sparse_x=x_tr_metrics, train_sparse_y=h_tr_acc)
        save_dual_metric_plot(x_tr_prec_dense, h_tr_prec_dense, x_val, h_prec, 'Train vs Val Precision', 'Precision', 'precision_graph.png', 'royalblue', 'cornflowerblue', 'o', 's', train_sparse_x=x_tr_metrics, train_sparse_y=h_tr_prec)
        save_dual_metric_plot(x_tr_rec_dense, h_tr_rec_dense, x_val, h_rec, 'Train vs Val Recall', 'Recall', 'recall_graph.png', 'darkorange', 'orange', 'D', 'o', train_sparse_x=x_tr_metrics, train_sparse_y=h_tr_rec)
        save_dual_metric_plot(x_tr_f1_dense, h_tr_f1_dense, x_val, h_f1, 'Train vs Val F1-Score', 'F1-Score', 'f1_graph.png', 'purple', 'mediumpurple', '^', 'o', train_sparse_x=x_tr_metrics, train_sparse_y=h_tr_f1)

    if x_val and h_acc_cls and len(h_acc_cls) == num_classes and len(x_val) == len(h_acc_cls[0]):
        colors = ['green', 'royalblue', 'darkorange', 'purple', 'teal', 'crimson']
        for c in range(num_classes):
            class_label = class_names[c] if c < len(class_names) else f"class_{c}"
            file_label = _safe_metric_filename(class_label)
            class_x_dense, class_y_dense = build_dense_epoch_series(
                x_tr_metrics,
                h_tr_acc_cls[c] if h_tr_acc_cls and len(h_tr_acc_cls) > c else [],
            )
            save_dual_metric_plot(
                class_x_dense,
                class_y_dense,
                x_val,
                h_acc_cls[c],
                f'Train vs Val Accuracy - {class_label}',
                'Accuracy',
                f'acc_{file_label}.png',
                colors[c % len(colors)],
                'black',
                'o',
                's',
                train_sparse_x=x_tr_metrics,
                train_sparse_y=h_tr_acc_cls[c] if h_tr_acc_cls and len(h_tr_acc_cls) > c else [],
            )

    if x_val and len(x_val) == len(h_map50):
        plt.figure(figsize=(10, 5))
        if x_tr_map50_dense and len(x_tr_map50_dense) == len(h_tr_map50_dense):
            plot_dense_train_series(x_tr_metrics, h_tr_map50, 'Train mAP@0.50', 'seagreen', 's')
        plt.plot(x_val, h_map50, label='Val mAP@0.50', color='mediumseagreen', marker='s')
        if x_tr_map5095_dense and len(x_tr_map5095_dense) == len(h_tr_map5095_dense):
            plot_dense_train_series(x_tr_metrics, h_tr_map5095, 'Train mAP@[0.50:0.95]', 'teal', '^')
        plt.plot(x_val, h_map5095, label='Val mAP@[0.50:0.95]', color='turquoise', marker='^')
        annotate_metric_series([
            (x_tr_metrics, h_tr_map50, 'max'),
            (x_val, h_map50, 'max'),
            (x_tr_metrics, h_tr_map5095, 'max'),
            (x_val, h_map5095, 'max'),
        ])
        plt.title('Train vs Val mAP')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.7)
        _savefig('map_graph.png')

        save_dual_metric_plot(x_tr_map50_dense, h_tr_map50_dense, x_val, h_map50, 'Train vs Val mAP@0.50', 'mAP@0.50', 'map50_graph.png', 'seagreen', 'mediumseagreen', 's', 'o', train_sparse_x=x_tr_metrics, train_sparse_y=h_tr_map50)
        save_dual_metric_plot(x_tr_map5095_dense, h_tr_map5095_dense, x_val, h_map5095, 'Train vs Val mAP@[0.50:0.95]', 'mAP@[0.50:0.95]', 'map5095_graph.png', 'teal', 'turquoise', '^', 'o', train_sparse_x=x_tr_metrics, train_sparse_y=h_tr_map5095)


def create_comparison_images(images, targets, predictions, epoch, class_names):
    save_dir = Config.TEST_RESULT_DIR
    n = min(Config.TEST_VIS_SAMPLES, len(images), len(targets), len(predictions))
    panel_gap = 2

    def add_title_bar(image, title, bar_height=42):
        h, w = image.shape[:2]
        canvas = np.full((h + bar_height, w, 3), 255, dtype=np.uint8)
        canvas[bar_height:, :] = image
        cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        return canvas

    def join_with_gap(panels, gap=2, gap_color=255):
        if not panels:
            return None
        if len(panels) == 1:
            return panels[0]

        panel_height = panels[0].shape[0]
        spacer = np.full((panel_height, gap, 3), gap_color, dtype=np.uint8)
        combined_panels = []
        for idx, panel in enumerate(panels):
            combined_panels.append(panel)
            if idx < len(panels) - 1:
                combined_panels.append(spacer.copy())
        return np.hstack(combined_panels)

    for i in range(n):
        img_tensor = images[i]
        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            img = np.asarray(img_tensor).transpose(1, 2, 0)
        img = (img * np.array(Config.STD) + np.array(Config.MEAN)) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        orig = img_bgr.copy()
        gt_img = img_bgr.copy()
        pred_img = img_bgr.copy()

        tgt = targets[i] if isinstance(targets, list) else {
            'boxes': targets['boxes'][i],
            'labels': targets['labels'][i],
        }

        if len(tgt['boxes']) > 0:
            gt_img = draw_bounding_boxes(
                gt_img,
                tgt['boxes'],
                tgt['labels'],
                class_names=class_names,
                box_format='cxywh',
            )

        if len(predictions[i]['boxes']) > 0:
            pred_img = draw_bounding_boxes(
                pred_img,
                predictions[i]['boxes'],
                predictions[i]['classes'],
                predictions[i]['scores'],
                class_names=class_names,
                box_format='xyxy',
            )

        orig = add_title_bar(orig, 'Original')
        gt_img = add_title_bar(gt_img, 'Ground Truth')
        pred_img = add_title_bar(pred_img, 'Prediction')

        combined = join_with_gap([orig, gt_img, pred_img], gap=panel_gap)
        cv2.imwrite(str(save_dir / f"compare_epoch{epoch}_{i}.jpg"), combined)


def log_per_class_metrics_dual(
    logger,
    epoch,
    class_names,
    val_current_bundle,
    val_best_bundle,
    tr_current_bundle=None,
    tr_best_bundle=None,
):
    sep = "=" * 90

    logger.info(sep)
    logger.info(f"  METRIK PER KELAS  |  Epoch {epoch:03d}")
    logger.info(sep)

    log_metric_table(logger, "VALIDASI (Epoch Ini)", class_names, val_current_bundle)
    log_metric_table(logger, "VALIDASI (Best Sejauh Ini)", class_names, val_best_bundle)

    if tr_current_bundle is not None:
        log_metric_table(logger, "TRAIN (Epoch Ini)", class_names, tr_current_bundle)
        log_metric_table(logger, "TRAIN (Best Sejauh Ini)", class_names, tr_best_bundle)

    logger.info(sep + "\n")


def print_final_summary(
    logger,
    class_names,
    best_val_map50_epoch,
    best_val_acc_epoch,
    best_val_bundle,
    best_tr_bundle,
    test_bundle,
    final_train_loss,
    final_val_loss,
    total_epochs,
    btt,
    btt_e,
    btb,
    btb_e,
    btc,
    btc_e,
    bto,
    bto_e,
    bvt,
    bvt_e,
    bvb,
    bvb_e,
    bvc,
    bvc_e,
    bvo,
    bvo_e,
):
    sep = "=" * 90
    sep2 = "-" * 90

    logger.info("\n" + sep)
    logger.info("  RINGKASAN AKHIR TRAINING")
    logger.info(sep)
    logger.info(f"  Best Val mAP@0.50        : {best_val_bundle['mAP@0.50']['global']:.4f}  (Epoch {best_val_map50_epoch})")
    logger.info(f"  Best Val mAP@[0.50:0.95] : {best_val_bundle['mAP@[0.50:0.95]']['global']:.4f}")
    logger.info(f"  Best Val Accuracy        : {best_val_bundle['Accuracy']['global']:.4f}  (Epoch {best_val_acc_epoch})")
    logger.info(f"  Best Val Precision       : {best_val_bundle['Precision']['global']:.4f}")
    logger.info(f"  Best Val Recall          : {best_val_bundle['Recall']['global']:.4f}")
    logger.info(f"  Best Val F1-Score        : {best_val_bundle['F1-Score']['global']:.4f}")
    logger.info(sep2)

    log_metric_table(logger, "BEST VALIDASI", class_names, best_val_bundle)
    logger.info(sep2)

    log_metric_table(logger, "BEST TRAIN", class_names, best_tr_bundle)
    logger.info(sep2)

    logger.info("  TEST GLOBAL")
    logger.info(
        f"  Test mAP@0.50={test_bundle['mAP@0.50']['global']:.4f} | "
        f"mAP@[0.50:0.95]={test_bundle['mAP@[0.50:0.95]']['global']:.4f} | "
        f"Accuracy={test_bundle['Accuracy']['global']:.4f} | "
        f"Prec={test_bundle['Precision']['global']:.4f} | "
        f"Rec={test_bundle['Recall']['global']:.4f} | "
        f"F1={test_bundle['F1-Score']['global']:.4f}"
    )
    log_metric_table(logger, "TEST PER KELAS", class_names, test_bundle)
    logger.info(sep2)

    logger.info("  BEST LOSS")
    logger.info(f"  {'Komponen':<28} {'Train':>12} {'Epoch':>8} {'Val':>12} {'Epoch':>8}")
    logger.info(f"  {'-'*28} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    logger.info(f"  {'Total':<28} {btt:>12.4f} {btt_e:>8} {bvt:>12.4f} {bvt_e:>8}")
    logger.info(f"  {'BBox/Regression':<28} {btb:>12.4f} {btb_e:>8} {bvb:>12.4f} {bvb_e:>8}")
    logger.info(f"  {'Classification':<28} {btc:>12.4f} {btc_e:>8} {bvc:>12.4f} {bvc_e:>8}")
    logger.info(f"  {'Objectness':<28} {bto:>12.4f} {bto_e:>8} {bvo:>12.4f} {bvo_e:>8}")
    logger.info(sep2)

    gap = final_train_loss - final_val_loss
    logger.info(f"  Final Train Loss : {final_train_loss:.4f}  |  Final Val Loss: {final_val_loss:.4f}")
    logger.info(
        f"  Gap (Train-Val)  : {gap:+.4f}  -> "
        + ("GOOD FIT" if abs(gap) < 0.05 else ("VAL > TRAIN (data shift?)" if gap < -0.05 else "TRAIN << VAL (overfitting?)"))
    )
    logger.info(sep + "\n")


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, logger=None):
    model.train()
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    tl = tb = tc = to = 0.0
    n = 0

    pbar = tqdm(
        dataloader,
        desc=f"Train E-{epoch:03d}",
        bar_format='{desc:<14} |{bar:28}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
    )

    for batch_idx, (images, targets) in enumerate(pbar):
        batch_name = f"Train E-{epoch:03d} B-{batch_idx + 1:04d}"
        if Config.STRICT_TARGET_VALIDATION:
            targets = sanitize_targets(
                targets,
                num_classes=Config.NUM_CLASSES,
                image_size=Config.IMAGE_SIZE,
                logger=logger,
                batch_name=batch_name,
            )

        try:
            images = images.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=Config.USE_AMP):
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = losses['total_loss']

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss terdeteksi: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(getattr(Config, 'GRAD_CLIP_NORM', 0.0)),
            )
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as exc:
            cleanup_cuda_after_error()
            if logger is not None:
                logger.error(
                    "%s gagal diproses | image_ids=%s | error=%s",
                    batch_name,
                    targets.get('image_ids', []),
                    exc,
                )
            raise

        tl += loss.item()
        tb += losses['bbox_loss'].item()
        tc += losses['class_loss'].item()
        to += losses['obj_loss'].item()
        n += 1

        pbar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            L_Tot=f"{loss.item():.3f}",
            L_Box=f"{losses['bbox_loss'].item():.3f}",
            L_Cls=f"{losses['class_loss'].item():.3f}",
            W_Box=f"{losses.get('bbox_weight', torch.tensor(0.0)).item():.2f}",
        )

    n = max(1, n)
    return {
        'total_loss': tl / n,
        'bbox_loss': tb / n,
        'class_loss': tc / n,
        'obj_loss': to / n,
    }


def _chunked_map(preds, tgts, num_classes, iou_thresholds, chunk=150):
    total = len(preds)
    if total == 0:
        results = {'mAP@0.50': 0.0, 'mAP@[0.5:0.95]': 0.0}
        for thr in iou_thresholds:
            for cls_id in range(num_classes):
                results[f'AP@{thr:.2f}_class_{cls_id}'] = 0.0
        return results

    if total <= chunk:
        try:
            return calculate_map(preds, tgts, num_classes=num_classes, iou_thresholds=iou_thresholds)
        except Exception:
            results = {'mAP@0.50': 0.0, 'mAP@[0.5:0.95]': 0.0}
            for thr in iou_thresholds:
                for cls_id in range(num_classes):
                    results[f'AP@{thr:.2f}_class_{cls_id}'] = 0.0
            return results

    per_threshold = {thr: [] for thr in iou_thresholds}
    per_class = {thr: {cls_id: [] for cls_id in range(num_classes)} for thr in iou_thresholds}

    for chunk_idx in range((total + chunk - 1) // chunk):
        start = chunk_idx * chunk
        end = min((chunk_idx + 1) * chunk, total)
        try:
            part = calculate_map(preds[start:end], tgts[start:end], num_classes=num_classes, iou_thresholds=iou_thresholds)
            for thr in iou_thresholds:
                per_threshold[thr].append(part.get(f'mAP@{thr:.2f}', 0.0))
                for cls_id in range(num_classes):
                    per_class[thr][cls_id].append(part.get(f'AP@{thr:.2f}_class_{cls_id}', 0.0))
        except Exception:
            for thr in iou_thresholds:
                per_threshold[thr].append(0.0)
                for cls_id in range(num_classes):
                    per_class[thr][cls_id].append(0.0)
        gc.collect()

    results = {}
    for thr in iou_thresholds:
        results[f'mAP@{thr:.2f}'] = float(np.mean(per_threshold[thr]))
        for cls_id in range(num_classes):
            results[f'AP@{thr:.2f}_class_{cls_id}'] = float(np.mean(per_class[thr][cls_id]))
    if len(iou_thresholds) > 1:
        results['mAP@[0.5:0.95]'] = float(np.mean([results[f'mAP@{thr:.2f}'] for thr in iou_thresholds]))
    return results


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    label_prefix="Val",
    collect_samples=False,
    logger=None,
    show_progress=True,
):
    model.eval()
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    tl = tb = tc = to = 0.0
    n = 0
    all_class_preds = []
    all_det_preds = []
    all_tgts = []
    sample_imgs = []
    sample_tgts = []

    pbar = tqdm(
        dataloader,
        desc=f"{label_prefix} E-{epoch:03d}",
        bar_format='{desc:<14} |{bar:28}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
        disable=not show_progress,
    )

    for idx, (images, targets) in enumerate(pbar):
        batch_name = f"{label_prefix} E-{epoch:03d} B-{idx + 1:04d}"
        if Config.STRICT_TARGET_VALIDATION:
            targets = sanitize_targets(
                targets,
                num_classes=Config.NUM_CLASSES,
                image_size=Config.IMAGE_SIZE,
                logger=logger,
                batch_name=batch_name,
            )

        images = images.to(device, non_blocking=True)
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            losses = criterion(outputs, targets)

        tl += losses['total_loss'].item()
        tb += losses['bbox_loss'].item()
        tc += losses['class_loss'].item()
        to += losses['obj_loss'].item()
        n += 1

        class_detections = model.get_class_oriented_detections(
            images,
            conf_threshold=getattr(Config, 'CLASS_METRIC_CONF_THRESHOLD', Config.CLASS_CONF_THRESHOLD),
            nms_iou_threshold=getattr(Config, 'CLASS_METRIC_NMS_IOU_THRESHOLD', Config.CLASS_NMS_IOU_THRESHOLD),
            max_detections=getattr(Config, 'CLASS_METRIC_MAX_DETECTIONS', Config.CLASS_MAX_DETECTIONS),
            use_centerness_in_score=getattr(Config, 'CLASS_METRIC_USE_CENTERNESS', False),
            outputs=outputs,
        )
        det_detections = model.get_detections(
            images,
            conf_threshold=Config.DET_CONF_THRESHOLD,
            nms_iou_threshold=Config.DET_NMS_IOU_THRESHOLD,
            max_detections=Config.DET_MAX_DETECTIONS,
            outputs=outputs,
        )

        for class_det, det_det, boxes, labels in zip(class_detections, det_detections, targets['boxes'], targets['labels']):
            all_class_preds.append({
                'boxes': class_det['boxes'].cpu(),
                'scores': class_det['scores'].cpu(),
                'classes': class_det['classes'].cpu(),
            })
            all_det_preds.append({
                'boxes': det_det['boxes'].cpu(),
                'scores': det_det['scores'].cpu(),
                'classes': det_det['classes'].cpu(),
            })
            all_tgts.append({
                'boxes': boxes.cpu(),
                'labels': labels.cpu(),
            })

        if collect_samples and len(sample_imgs) < Config.TEST_VIS_SAMPLES:
            remaining = Config.TEST_VIS_SAMPLES - len(sample_imgs)
            batch_take = min(remaining, len(targets['boxes']))
            sample_imgs.extend([img.cpu().clone() for img in images[:batch_take]])
            sample_tgts.extend(
                [{'boxes': b.cpu(), 'labels': l.cpu()} for b, l in zip(targets['boxes'][:batch_take], targets['labels'][:batch_take])]
            )

        del images, outputs, class_detections, det_detections
        if getattr(Config, 'EMPTY_CACHE_PER_EVAL_BATCH', False):
            torch.cuda.empty_cache()
            gc.collect()
        if show_progress:
            pbar.set_postfix(L_Tot=f"{losses['total_loss'].item():.3f}")

    n = max(1, n)
    avg_losses = {
        'total_loss': tl / n,
        'bbox_loss': tb / n,
        'class_loss': tc / n,
        'obj_loss': to / n,
    }

    class_metrics = calculate_classification_metrics(all_class_preds, all_tgts, Config.NUM_CLASSES)
    map_metrics = _chunked_map(all_det_preds, all_tgts, Config.NUM_CLASSES, MAP_IOU_THRESHOLDS)
    metrics = {
        'val_loss': avg_losses['class_loss'],
        'val_loss_total': avg_losses['total_loss'],
        'val_loss_cls': avg_losses['class_loss'],
        'val_loss_bbox': avg_losses['bbox_loss'],
        'val_loss_obj': avg_losses['obj_loss'],
        **class_metrics,
        **map_metrics,
    }
    return metrics, avg_losses, all_class_preds, all_det_preds, all_tgts, sample_imgs, sample_tgts


def _extract_per_class(class_metrics, num_classes):
    acc = list(class_metrics.get('accuracy_per_class', np.zeros(num_classes))[:num_classes])
    prec = list(class_metrics.get('precision_per_class', np.zeros(num_classes))[:num_classes])
    rec = list(class_metrics.get('recall_per_class', np.zeros(num_classes))[:num_classes])
    f1 = list(class_metrics.get('f1_per_class', np.zeros(num_classes))[:num_classes])
    return acc, prec, rec, f1


def save_checkpoint(model, optimizer, epoch, metrics, fname=None, scheduler=None, scaler=None, train_state=None):
    fname = fname or f'checkpoint_epoch_{epoch}.pth'
    path = Config.CHECKPOINT_DIR / fname
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        payload['scaler_state_dict'] = scaler.state_dict()
    if train_state is not None:
        payload['train_state'] = train_state
    torch.save(payload, path)
    return path


def run_test_phase(model, criterion, device, class_names, val_tf, logger, epoch_label, checkpoint_path: Path | None = None):
    test_bundle = init_metric_bundle(Config.NUM_CLASSES)

    try:
        from torch.utils.data import DataLoader
        from data.utils import collate_fn

        test_ds_final = ObjectDetectionDataset(
            Config.TEST_IMAGES,
            Config.TEST_ANNOTATIONS,
            transform=val_tf,
            image_size=Config.IMAGE_SIZE,
            repeat_factor=1,
        )
        test_loader = DataLoader(
            test_ds_final,
            batch_size=min(Config.BATCH_SIZE, 4),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        logger.info(
            f"\n  Mengevaluasi pada test set "
            f"(checkpoint: {checkpoint_path if checkpoint_path is not None else 'model saat ini'})..."
        )
        test_metrics, _, test_class_preds, test_det_preds, test_tgts, sample_imgs, sample_tgts_list = evaluate(
            model,
            test_loader,
            criterion,
            device,
            epoch_label,
            label_prefix="Test",
            collect_samples=True,
            logger=logger,
        )

        test_bundle = extract_metric_bundle(test_metrics, Config.NUM_CLASSES)

        logger.info(
            f"  Test mAP@0.50={test_bundle['mAP@0.50']['global']:.4f} | "
            f"mAP@[0.50:0.95]={test_bundle['mAP@[0.50:0.95]']['global']:.4f} | "
            f"Acc={test_bundle['Accuracy']['global']:.4f} | "
            f"Prec={test_bundle['Precision']['global']:.4f} | "
            f"Rec={test_bundle['Recall']['global']:.4f} | "
            f"F1={test_bundle['F1-Score']['global']:.4f}"
        )

        generate_confusion_matrix(
            test_class_preds,
            test_tgts,
            Config.NUM_CLASSES,
            class_names=class_names,
            fname=Config.GRAPHS_DIR / 'confusion_matrix_class_test.png',
        )
        generate_detection_confusion_matrix(
            test_det_preds,
            test_tgts,
            Config.NUM_CLASSES,
            class_names=class_names,
            iou_threshold=0.5,
            fname=Config.GRAPHS_DIR / 'confusion_matrix_detection_test.png',
        )

        if len(sample_imgs) > 0:
            create_comparison_images(
                sample_imgs,
                sample_tgts_list,
                test_class_preds[:len(sample_imgs)],
                epoch_label,
                class_names,
            )
            logger.info(f"  Gambar perbandingan disimpan di: {Config.TEST_RESULT_DIR}")

    except Exception as exc:
        import traceback
        logger.error(f"Error fase Testing: {exc}\n{traceback.format_exc()}")

    return test_bundle


def train(args):
    if getattr(args, 'vis_samples', None) is not None:
        Config.TEST_VIS_SAMPLES = int(args.vis_samples)

    test_only_checkpoint = None
    if getattr(args, 'test_only', False):
        test_only_source = getattr(args, 'checkpoint', None) or getattr(args, 'resume', None)
        test_only_checkpoint = resolve_evaluation_checkpoint(test_only_source) if test_only_source else None
        if test_only_checkpoint is None:
            raise ValueError("Mode --test-only membutuhkan --checkpoint atau --resume yang mengarah ke run/checkpoint.")

    resume_checkpoint = resolve_resume_checkpoint(getattr(args, 'resume', None)) if not getattr(args, 'test_only', False) else None
    active_checkpoint = test_only_checkpoint or resume_checkpoint
    run_dir = setup_run(active_checkpoint)
    logger = setup_logging()

    logger.info("=" * 70)
    if getattr(args, 'test_only', False):
        logger.info(f"  TEST ONLY       |  Run: {run_dir.name}")
        logger.info(f"  Checkpoint      : {test_only_checkpoint}")
    elif resume_checkpoint is not None:
        logger.info(f"  RESUME TRAINING |  Run: {run_dir.name}")
        logger.info(f"  Resume From     : {resume_checkpoint}")
    else:
        logger.info(f"  MULAI TRAINING  |  Run: {run_dir.name}")
    logger.info(f"  Output Dir: {run_dir.resolve()}")
    logger.info("=" * 70)

    device = Config.DEVICE
    if device.type == 'cuda' and Config.CUDA_DEBUG_SYNC:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logger.warning("CUDA debug sync aktif: CUDA_LAUNCH_BLOCKING=1. Training akan lebih lambat tetapi stacktrace lebih akurat.")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = bool(getattr(Config, 'CUDA_BENCHMARK', True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(Config, 'ALLOW_TF32', True))
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = bool(getattr(Config, 'ALLOW_TF32', True))
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    tr_tf = get_train_transforms(
        Config.IMAGE_SIZE,
        Config.MEAN,
        Config.STD,
        augment=Config.AUGMENT,
        median_blur_prob=Config.MEDIAN_BLUR_PROB,
        median_blur_limit=Config.MEDIAN_BLUR_LIMIT,
        horizontal_flip_prob=Config.HORIZONTAL_FLIP_PROB,
        vertical_flip_prob=Config.VERTICAL_FLIP_PROB,
        rotate_limit=Config.ROTATE_LIMIT,
        rotate_prob=Config.ROTATE_PROB,
        random_resized_crop_prob=Config.RANDOM_RESIZED_CROP_PROB,
        random_resized_crop_scale=Config.RANDOM_RESIZED_CROP_SCALE,
        shift_scale_rotate_prob=Config.SHIFT_SCALE_ROTATE_PROB,
        shift_limit=Config.SHIFT_LIMIT,
        scale_limit=Config.SCALE_LIMIT,
        color_jitter_prob=Config.COLOR_JITTER_PROB,
        color_jitter_brightness=Config.COLOR_JITTER_BRIGHTNESS,
        color_jitter_contrast=Config.COLOR_JITTER_CONTRAST,
        color_jitter_saturation=Config.COLOR_JITTER_SATURATION,
        color_jitter_hue=Config.COLOR_JITTER_HUE,
        random_brightness_contrast_prob=Config.RANDOM_BRIGHTNESS_CONTRAST_PROB,
        clahe_prob=Config.CLAHE_PROB,
    )
    val_tf = get_val_transforms(Config.IMAGE_SIZE, Config.MEAN, Config.STD)

    train_ds = ObjectDetectionDataset(
        Config.TRAIN_IMAGES,
        Config.TRAIN_ANNOTATIONS,
        transform=tr_tf,
        image_size=Config.IMAGE_SIZE,
        repeat_factor=Config.AUGMENT_REPEAT_FACTOR if Config.AUGMENT else 1,
    )
    val_ds = ObjectDetectionDataset(
        Config.VAL_IMAGES,
        Config.VAL_ANNOTATIONS,
        transform=val_tf,
        image_size=Config.IMAGE_SIZE,
        repeat_factor=1,
    )

    test_ds = None
    try:
        test_ds = ObjectDetectionDataset(
            Config.TEST_IMAGES,
            Config.TEST_ANNOTATIONS,
            transform=val_tf,
            image_size=Config.IMAGE_SIZE,
            repeat_factor=1,
        )
    except Exception:
        pass

    class_names = Config.COCO_CLASSES if hasattr(Config, 'COCO_CLASSES') else [f"Class_{i}" for i in range(Config.NUM_CLASSES)]
    print_dataset_summary(logger, train_ds, val_ds, test_ds, Config.NUM_CLASSES, class_names)

    train_loader, val_loader = create_dataloaders(
        train_ds,
        val_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    model = HybridDetector(
        num_classes=Config.NUM_CLASSES,
        image_size=Config.IMAGE_SIZE,
        transformer_dim=Config.TRANSFORMER_DIM,
        transformer_heads=Config.TRANSFORMER_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS,
    ).to(device)
    print_model_config(
        logger,
        model,
        Config.NUM_CLASSES,
        Config.IMAGE_SIZE,
        Config.BATCH_SIZE,
        Config.LEARNING_RATE,
        Config.EPOCHS,
        device,
        Config.USE_AMP,
    )

    criterion = AnchorFreeLoss(num_classes=Config.NUM_CLASSES)
    optimizer = optim.AdamW(
        [
            {
                'params': [p for n, p in model.named_parameters() if 'backbone' in n],
                'lr': Config.LEARNING_RATE * 0.1,
            },
            {
                'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
                'lr': Config.LEARNING_RATE,
            },
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=Config.LEARNING_RATE * 0.01,
    )
    scaler = GradScaler('cuda', enabled=Config.USE_AMP)
    start_epoch = 0

    best_val_map50 = 0.0
    best_val_map5095 = 0.0
    best_val_map50_epoch = 0
    best_val_acc = 0.0
    best_val_acc_epoch = 0
    best_val_bundle = init_metric_bundle(Config.NUM_CLASSES)
    best_tr_bundle = init_metric_bundle(Config.NUM_CLASSES)

    h_tr_loss = []
    h_val_loss = []
    h_tr_bbox = []
    h_val_bbox = []
    h_tr_cls = []
    h_val_cls = []
    h_tr_obj = []
    h_val_obj = []
    x_tr_metrics = []
    h_tr_acc = []
    h_tr_acc_cls = [[] for _ in range(Config.NUM_CLASSES)]
    h_tr_prec = []
    h_tr_rec = []
    h_tr_f1 = []
    h_tr_map50 = []
    h_tr_map5095 = []
    h_acc = []
    h_acc_cls = [[] for _ in range(Config.NUM_CLASSES)]
    h_prec = []
    h_rec = []
    h_f1 = []
    h_map50 = []
    h_map5095 = []

    btt = float('inf')
    btb = float('inf')
    btc = float('inf')
    bto = float('inf')
    bvt = float('inf')
    bvb = float('inf')
    bvc = float('inf')
    bvo = float('inf')
    btt_e = btb_e = btc_e = bto_e = 0
    bvt_e = bvb_e = bvc_e = bvo_e = 0

    final_train_loss = 0.0
    final_val_loss = 0.0
    elapsed_time_offset = 0.0
    training_session_start = time.time()

    train_eval_freq = max(1, int(getattr(Config, 'TRAIN_EVAL_FREQUENCY', 10)))
    train_graph_eval_freq = max(1, int(getattr(Config, 'TRAIN_GRAPH_EVAL_FREQUENCY', 1)))

    def current_train_state():
        return {
            'best_val_map50': best_val_map50,
            'best_val_map5095': best_val_map5095,
            'best_val_map50_epoch': best_val_map50_epoch,
            'best_val_acc': best_val_acc,
            'best_val_acc_epoch': best_val_acc_epoch,
            'best_val_bundle': best_val_bundle,
            'best_tr_bundle': best_tr_bundle,
            'h_tr_loss': h_tr_loss,
            'h_val_loss': h_val_loss,
            'h_tr_bbox': h_tr_bbox,
            'h_val_bbox': h_val_bbox,
            'h_tr_cls': h_tr_cls,
            'h_val_cls': h_val_cls,
            'h_tr_obj': h_tr_obj,
            'h_val_obj': h_val_obj,
            'x_tr_metrics': x_tr_metrics,
            'h_tr_acc': h_tr_acc,
            'h_tr_acc_cls': h_tr_acc_cls,
            'h_tr_prec': h_tr_prec,
            'h_tr_rec': h_tr_rec,
            'h_tr_f1': h_tr_f1,
            'h_tr_map50': h_tr_map50,
            'h_tr_map5095': h_tr_map5095,
            'h_acc': h_acc,
            'h_acc_cls': h_acc_cls,
            'h_prec': h_prec,
            'h_rec': h_rec,
            'h_f1': h_f1,
            'h_map50': h_map50,
            'h_map5095': h_map5095,
            'btt': btt,
            'btb': btb,
            'btc': btc,
            'bto': bto,
            'bvt': bvt,
            'bvb': bvb,
            'bvc': bvc,
            'bvo': bvo,
            'btt_e': btt_e,
            'btb_e': btb_e,
            'btc_e': btc_e,
            'bto_e': bto_e,
            'bvt_e': bvt_e,
            'bvb_e': bvb_e,
            'bvc_e': bvc_e,
            'bvo_e': bvo_e,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'total_train_time': get_realtime_elapsed(training_session_start, elapsed_time_offset),
        }

    if getattr(args, 'test_only', False):
        run_test_phase(
            model=model,
            criterion=criterion,
            device=device,
            class_names=class_names,
            val_tf=val_tf,
            logger=logger,
            epoch_label=0,
            checkpoint_path=test_only_checkpoint,
        )
        logger.info(f"\n  Test-only selesai. Output disimpan di: {Config.RUN_DIR.resolve()}")
        logger.info(f"  test_results/  ({Config.TEST_VIS_SAMPLES} gambar prediksi)")
        return

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception:
                logger.warning("State GradScaler dari checkpoint tidak dapat dimuat. Training tetap dilanjutkan.")

        start_epoch = int(ckpt.get('epoch', 0))
        resume_state = ckpt.get('train_state', {})
        best_val_map50 = float(resume_state.get('best_val_map50', best_val_map50))
        best_val_map5095 = float(resume_state.get('best_val_map5095', best_val_map5095))
        best_val_map50_epoch = int(resume_state.get('best_val_map50_epoch', best_val_map50_epoch))
        best_val_acc = float(resume_state.get('best_val_acc', best_val_acc))
        best_val_acc_epoch = int(resume_state.get('best_val_acc_epoch', best_val_acc_epoch))
        best_val_bundle = resume_state.get('best_val_bundle', best_val_bundle)
        best_tr_bundle = resume_state.get('best_tr_bundle', best_tr_bundle)

        h_tr_loss = resume_state.get('h_tr_loss', h_tr_loss)
        h_val_loss = resume_state.get('h_val_loss', h_val_loss)
        h_tr_bbox = resume_state.get('h_tr_bbox', h_tr_bbox)
        h_val_bbox = resume_state.get('h_val_bbox', h_val_bbox)
        h_tr_cls = resume_state.get('h_tr_cls', h_tr_cls)
        h_val_cls = resume_state.get('h_val_cls', h_val_cls)
        h_tr_obj = resume_state.get('h_tr_obj', h_tr_obj)
        h_val_obj = resume_state.get('h_val_obj', h_val_obj)
        x_tr_metrics = resume_state.get('x_tr_metrics', x_tr_metrics)
        h_tr_acc = resume_state.get('h_tr_acc', h_tr_acc)
        h_tr_acc_cls = resume_state.get('h_tr_acc_cls', h_tr_acc_cls)
        h_tr_prec = resume_state.get('h_tr_prec', h_tr_prec)
        h_tr_rec = resume_state.get('h_tr_rec', h_tr_rec)
        h_tr_f1 = resume_state.get('h_tr_f1', h_tr_f1)
        h_tr_map50 = resume_state.get('h_tr_map50', h_tr_map50)
        h_tr_map5095 = resume_state.get('h_tr_map5095', h_tr_map5095)
        h_acc = resume_state.get('h_acc', h_acc)
        h_acc_cls = resume_state.get('h_acc_cls', h_acc_cls)
        h_prec = resume_state.get('h_prec', h_prec)
        h_rec = resume_state.get('h_rec', h_rec)
        h_f1 = resume_state.get('h_f1', h_f1)
        h_map50 = resume_state.get('h_map50', h_map50)
        h_map5095 = resume_state.get('h_map5095', h_map5095)

        btt = float(resume_state.get('btt', btt))
        btb = float(resume_state.get('btb', btb))
        btc = float(resume_state.get('btc', btc))
        bto = float(resume_state.get('bto', bto))
        bvt = float(resume_state.get('bvt', bvt))
        bvb = float(resume_state.get('bvb', bvb))
        bvc = float(resume_state.get('bvc', bvc))
        bvo = float(resume_state.get('bvo', bvo))
        btt_e = int(resume_state.get('btt_e', btt_e))
        btb_e = int(resume_state.get('btb_e', btb_e))
        btc_e = int(resume_state.get('btc_e', btc_e))
        bto_e = int(resume_state.get('bto_e', bto_e))
        bvt_e = int(resume_state.get('bvt_e', bvt_e))
        bvb_e = int(resume_state.get('bvb_e', bvb_e))
        bvc_e = int(resume_state.get('bvc_e', bvc_e))
        bvo_e = int(resume_state.get('bvo_e', bvo_e))
        final_train_loss = float(resume_state.get('final_train_loss', final_train_loss))
        final_val_loss = float(resume_state.get('final_val_loss', final_val_loss))
        elapsed_time_offset = float(resume_state.get('total_train_time', elapsed_time_offset))
        training_session_start = time.time()

        logger.info(f"  Resume Epoch     : {start_epoch}")
        logger.info(f"  Next Epoch       : {start_epoch + 1}")
        logger.info(f"  Akumulasi Waktu  : {format_time(elapsed_time_offset)}")

    for epoch in range(start_epoch, Config.EPOCHS):
        cur_epoch = epoch + 1
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cur_epoch, logger=logger)
        train_time = time.time() - t0
        final_train_loss = train_losses['total_loss']

        h_tr_loss.append(train_losses['total_loss'])
        h_tr_bbox.append(train_losses['bbox_loss'])
        h_tr_cls.append(train_losses['class_loss'])
        h_tr_obj.append(train_losses['obj_loss'])

        if train_losses['total_loss'] < btt:
            btt, btt_e = train_losses['total_loss'], cur_epoch
        if train_losses['bbox_loss'] < btb:
            btb, btb_e = train_losses['bbox_loss'], cur_epoch
        if train_losses['class_loss'] < btc:
            btc, btc_e = train_losses['class_loss'], cur_epoch
        if train_losses['obj_loss'] < bto:
            bto, bto_e = train_losses['obj_loss'], cur_epoch

        if cur_epoch % Config.EVAL_FREQUENCY != 0:
            logger.info(f"Epoch {cur_epoch:03d}/{Config.EPOCHS} | {format_time(train_time)} | TrainLoss={train_losses['total_loss']:.4f}")
            save_checkpoint(
                model,
                optimizer,
                cur_epoch,
                {'train_loss': train_losses['total_loss']},
                'latest_checkpoint.pth',
                scheduler=scheduler,
                scaler=scaler,
                train_state=current_train_state(),
            )
            if cur_epoch % getattr(Config, 'SAVE_FREQUENCY', 5) == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    cur_epoch,
                    {'train_loss': train_losses['total_loss']},
                    scheduler=scheduler,
                    scaler=scaler,
                    train_state=current_train_state(),
                )
            scheduler.step()
            continue

        t1 = time.time()
        val_metrics, val_losses, val_class_preds, val_det_preds, val_tgts, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cur_epoch,
            logger=logger,
        )
        val_time = time.time() - t1
        final_val_loss = val_losses['total_loss']

        h_val_loss.append(val_losses['total_loss'])
        h_val_bbox.append(val_losses['bbox_loss'])
        h_val_cls.append(val_losses['class_loss'])
        h_val_obj.append(val_losses['obj_loss'])

        if val_losses['total_loss'] < bvt:
            bvt, bvt_e = val_losses['total_loss'], cur_epoch
        if val_losses['bbox_loss'] < bvb:
            bvb, bvb_e = val_losses['bbox_loss'], cur_epoch
        if val_losses['class_loss'] < bvc:
            bvc, bvc_e = val_losses['class_loss'], cur_epoch
        if val_losses['obj_loss'] < bvo:
            bvo, bvo_e = val_losses['obj_loss'], cur_epoch

        val_bundle = extract_metric_bundle(val_metrics, Config.NUM_CLASSES)
        update_best_metric_bundle(best_val_bundle, val_bundle)

        cur_acc = val_bundle['Accuracy']['global']
        cur_prec = val_bundle['Precision']['global']
        cur_rec = val_bundle['Recall']['global']
        cur_f1 = val_bundle['F1-Score']['global']
        cur_map50 = val_bundle['mAP@0.50']['global']
        cur_map5095 = val_bundle['mAP@[0.50:0.95]']['global']

        h_acc.append(cur_acc)
        h_prec.append(cur_prec)
        h_rec.append(cur_rec)
        h_f1.append(cur_f1)
        h_map50.append(cur_map50)
        h_map5095.append(cur_map5095)

        for class_id in range(Config.NUM_CLASSES):
            h_acc_cls[class_id].append(val_bundle['Accuracy']['per_class'][class_id])

        prev_best_map50 = best_val_map50
        prev_best_acc = best_val_acc

        if cur_map50 > best_val_map50:
            best_val_map50 = cur_map50
            best_val_map50_epoch = cur_epoch
        best_val_map5095 = max(best_val_map5095, cur_map5095)
        if cur_acc > best_val_acc:
            best_val_acc = cur_acc
            best_val_acc_epoch = cur_epoch

        epoch_time = train_time + val_time
        logger.info(
            f"Epoch {cur_epoch:03d}/{Config.EPOCHS} | {format_time(epoch_time)} | "
            f"TrainLossCls={train_losses['class_loss']:.4f} | ValLossCls={val_losses['class_loss']:.4f} | "
            f"TrainLossTot={train_losses['total_loss']:.4f} | ValLossTot={val_losses['total_loss']:.4f} | "
            f"mAP50={cur_map50:.4f} | mAP5095={cur_map5095:.4f} | "
            f"Acc={cur_acc:.4f} | Prec={cur_prec:.4f} | Rec={cur_rec:.4f} | F1={cur_f1:.4f}"
        )

        run_train_table_eval = cur_epoch % train_eval_freq == 0
        run_train_graph_eval = cur_epoch % train_graph_eval_freq == 0

        if run_train_graph_eval:
            if run_train_table_eval:
                logger.info(f"  [E{cur_epoch}] Menjalankan evaluasi train set untuk per-class metrics...")

            eval_label = "TrainEval" if run_train_table_eval else "TrainGraph"
            tr_metrics, _, tr_class_preds, tr_det_preds, tr_tgts, _, _ = evaluate(
                model,
                train_loader,
                criterion,
                device,
                cur_epoch,
                label_prefix=eval_label,
                logger=logger if run_train_table_eval else None,
                show_progress=run_train_table_eval,
            )
            tr_bundle = extract_metric_bundle(tr_metrics, Config.NUM_CLASSES)
            update_best_metric_bundle(best_tr_bundle, tr_bundle)

            x_tr_metrics.append(cur_epoch)
            h_tr_acc.append(tr_bundle['Accuracy']['global'])
            h_tr_prec.append(tr_bundle['Precision']['global'])
            h_tr_rec.append(tr_bundle['Recall']['global'])
            h_tr_f1.append(tr_bundle['F1-Score']['global'])
            h_tr_map50.append(tr_bundle['mAP@0.50']['global'])
            h_tr_map5095.append(tr_bundle['mAP@[0.50:0.95]']['global'])
            for class_id in range(Config.NUM_CLASSES):
                h_tr_acc_cls[class_id].append(tr_bundle['Accuracy']['per_class'][class_id])

            if run_train_table_eval:
                log_per_class_metrics_dual(
                    logger,
                    cur_epoch,
                    class_names,
                    val_bundle,
                    best_val_bundle,
                    tr_bundle,
                    best_tr_bundle,
                )

            del tr_class_preds, tr_det_preds, tr_tgts
            gc.collect()
            torch.cuda.empty_cache()

        x_tr = list(range(1, len(h_tr_loss) + 1))
        x_val = [i for i in range(1, len(h_tr_loss) + 1) if i % Config.EVAL_FREQUENCY == 0][:len(h_val_loss)]
        update_all_plots(
            x_tr,
            x_val,
            h_tr_loss,
            h_val_loss,
            h_tr_bbox,
            h_val_bbox,
            h_tr_cls,
            h_val_cls,
            h_tr_obj,
            h_val_obj,
            x_tr_metrics,
            h_tr_acc,
            h_tr_acc_cls,
            h_tr_prec,
            h_tr_rec,
            h_tr_f1,
            h_tr_map50,
            h_tr_map5095,
            h_acc,
            h_acc_cls,
            h_prec,
            h_rec,
            h_f1,
            h_map50,
            h_map5095,
            class_names,
            Config.NUM_CLASSES,
        )

        generate_confusion_matrix(
            val_class_preds,
            val_tgts,
            Config.NUM_CLASSES,
            class_names=class_names,
            fname=Config.GRAPHS_DIR / 'confusion_matrix_class_val.png',
        )
        generate_detection_confusion_matrix(
            val_det_preds,
            val_tgts,
            Config.NUM_CLASSES,
            class_names=class_names,
            iou_threshold=0.5,
            fname=Config.GRAPHS_DIR / 'confusion_matrix_detection_val.png',
        )

        checkpoint_metric = getattr(Config, 'CHECKPOINT_METRIC', 'mAP@0.50')
        if checkpoint_metric == 'Accuracy':
            is_primary_best = cur_acc >= prev_best_acc
        else:
            is_primary_best = cur_map50 >= prev_best_map50

        if is_primary_best:
            save_checkpoint(
                model,
                optimizer,
                cur_epoch,
                val_metrics,
                'best_model.pth',
                scheduler=scheduler,
                scaler=scaler,
                train_state=current_train_state(),
            )
        if cur_map50 >= prev_best_map50:
            save_checkpoint(
                model,
                optimizer,
                cur_epoch,
                val_metrics,
                'best_map_model.pth',
                scheduler=scheduler,
                scaler=scaler,
                train_state=current_train_state(),
            )
        save_checkpoint(
            model,
            optimizer,
            cur_epoch,
            val_metrics,
            'latest_checkpoint.pth',
            scheduler=scheduler,
            scaler=scaler,
            train_state=current_train_state(),
        )
        if cur_epoch % getattr(Config, 'SAVE_FREQUENCY', 5) == 0:
            save_checkpoint(
                model,
                optimizer,
                cur_epoch,
                val_metrics,
                scheduler=scheduler,
                scaler=scaler,
                train_state=current_train_state(),
            )

        del val_class_preds, val_det_preds, val_tgts
        gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()

    logger.info(
        f"\n  TRAINING SELESAI | Waktu total: "
        f"{format_time(get_realtime_elapsed(training_session_start, elapsed_time_offset))}"
    )
    logger.info(f"  Best Val mAP@0.50 : {best_val_map50:.4f}  (Epoch {best_val_map50_epoch})")
    logger.info(
        f"  Best Val Accuracy : {best_val_acc:.4f}  (Epoch {best_val_acc_epoch})"
        f"{' [checkpoint utama]' if getattr(Config, 'CHECKPOINT_METRIC', 'mAP@0.50') == 'Accuracy' else ''}"
    )

    test_bundle = run_test_phase(
        model=model,
        criterion=criterion,
        device=device,
        class_names=class_names,
        val_tf=val_tf,
        logger=logger,
        epoch_label=Config.EPOCHS,
        checkpoint_path=Config.CHECKPOINT_DIR / 'best_model.pth',
    )

    print_final_summary(
        logger,
        class_names,
        best_val_map50_epoch,
        best_val_acc_epoch,
        best_val_bundle,
        best_tr_bundle,
        test_bundle,
        final_train_loss,
        final_val_loss,
        Config.EPOCHS,
        btt,
        btt_e,
        btb,
        btb_e,
        btc,
        btc_e,
        bto,
        bto_e,
        bvt,
        bvt_e,
        bvb,
        bvb_e,
        bvc,
        bvc_e,
        bvo,
        bvo_e,
    )

    logger.info(f"\n  Semua output tersimpan di: {Config.RUN_DIR.resolve()}")
    logger.info("  checkpoints/   (model .pth)")
    logger.info("  logs/          (file .log)")
    logger.info(f"  graphs/        ({len(list(Config.GRAPHS_DIR.glob('*.png')))} grafik .png)")
    logger.info("  test_results/  (gambar prediksi)")

    stage2_classifier_enabled = bool(getattr(Config, "ENABLE_STAGE2_CLASSIFIER", True))
    if not getattr(args, "detector_only", False) and stage2_classifier_enabled:
        logger.info("\n" + "=" * 70)
        logger.info("  LANJUT TRAINING STAGE-2 CLASSIFIER")
        logger.info("=" * 70)
        try:
            cls_result = train_stage2_classifier()
            logger.info(
                f"  Stage-2 classifier selesai | "
                f"best_safety_score={cls_result['best_score']:.4f}"
            )
            logger.info(f"  Best classifier checkpoint   : {cls_result['best_path']}")
            logger.info(f"  Latest classifier checkpoint : {cls_result['latest_path']}")
        except Exception as exc:
            import traceback
            logger.error(
                "Training stage-2 classifier gagal: %s\n%s",
                exc,
                traceback.format_exc(),
            )
    elif getattr(args, "detector_only", False):
        logger.info("\n  Stage-2 classifier dilewati karena --detector-only aktif.")
    else:
        logger.info("\n  Stage-2 classifier dilewati karena Config.ENABLE_STAGE2_CLASSIFIER=False.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--detector-only', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--vis-samples', type=int, default=None)
    args = parser.parse_args()
    train(args)
