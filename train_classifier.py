"""
Training stage-2 classifier HTEM untuk keputusan kelas yang lebih aman.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import cv2
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import (
    DiseaseCropDataset,
    get_classifier_train_transforms,
    get_classifier_val_transforms,
)
from models import PaperDiseaseClassifier, hierarchical_classifier_loss
from utils.visualization import draw_bounding_boxes


def _resolve_classifier_resume_checkpoint(resume_arg: str | None) -> Path | None:
    if not resume_arg:
        return None

    path = Path(resume_arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_file():
        return path

    ckpt_dir = path / "checkpoints" if (path / "checkpoints").exists() else path
    latest_ckpt = ckpt_dir / "latest_classifier_checkpoint.pth"
    if latest_ckpt.exists():
        return latest_ckpt

    best_ckpt = ckpt_dir / Config.CLASSIFIER_CHECKPOINT_NAME
    if best_ckpt.exists():
        return best_ckpt

    raise FileNotFoundError(f"Tidak menemukan checkpoint classifier untuk resume di: {path}")


def _setup_run_dir(resume_checkpoint: Path | None = None) -> Path:
    if resume_checkpoint is not None:
        run_dir = resume_checkpoint.resolve().parent.parent
    else:
        run_dir = Config.BASE_OUTPUT_DIR / f"classifier_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (run_dir / "test_results").mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("stage2_classifier")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "logs" / f"classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _format_time(seconds: float) -> str:
    return f"{int(seconds) // 60}m {int(seconds) % 60:02d}s"


def _get_realtime_elapsed(session_start_time: float | None, elapsed_offset: float = 0.0) -> float:
    if session_start_time is None:
        return float(elapsed_offset)
    return max(0.0, float(elapsed_offset) + (time.time() - session_start_time))


def _make_loaders():
    classifier_workers = int(getattr(Config, "CLASSIFIER_NUM_WORKERS", Config.NUM_WORKERS))
    train_tf = get_classifier_train_transforms(
        image_size=Config.CLASSIFIER_IMAGE_SIZE,
        mean=Config.MEAN,
        std=Config.STD,
        horizontal_flip_prob=Config.CLASSIFIER_HORIZONTAL_FLIP_PROB,
        vertical_flip_prob=Config.CLASSIFIER_VERTICAL_FLIP_PROB,
        rotate_limit=Config.CLASSIFIER_ROTATE_LIMIT,
        rotate_prob=Config.CLASSIFIER_ROTATE_PROB,
        color_jitter_prob=Config.CLASSIFIER_COLOR_JITTER_PROB,
        random_brightness_contrast_prob=Config.CLASSIFIER_RANDOM_BRIGHTNESS_CONTRAST_PROB,
        clahe_prob=Config.CLASSIFIER_CLAHE_PROB,
    )
    val_tf = get_classifier_val_transforms(
        image_size=Config.CLASSIFIER_IMAGE_SIZE,
        mean=Config.MEAN,
        std=Config.STD,
    )

    train_ds = DiseaseCropDataset(
        Config.TRAIN_IMAGES,
        Config.TRAIN_ANNOTATIONS,
        transform=train_tf,
        crop_padding=Config.CLASSIFIER_CROP_PADDING,
        min_crop_size=Config.CLASSIFIER_MIN_CROP_SIZE,
        repeat_factor=Config.AUGMENT_REPEAT_FACTOR if Config.AUGMENT else 1,
    )
    val_ds = DiseaseCropDataset(
        Config.VAL_IMAGES,
        Config.VAL_ANNOTATIONS,
        transform=val_tf,
        crop_padding=Config.CLASSIFIER_CROP_PADDING,
        min_crop_size=Config.CLASSIFIER_MIN_CROP_SIZE,
        repeat_factor=1,
    )

    test_ds = DiseaseCropDataset(
        Config.TEST_IMAGES,
        Config.TEST_ANNOTATIONS,
        transform=val_tf,
        crop_padding=Config.CLASSIFIER_CROP_PADDING,
        min_crop_size=Config.CLASSIFIER_MIN_CROP_SIZE,
        repeat_factor=1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=classifier_workers,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS if classifier_workers > 0 else False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=classifier_workers,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS if classifier_workers > 0 else False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=classifier_workers,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS if classifier_workers > 0 else False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def _build_model(device: torch.device) -> PaperDiseaseClassifier:
    model = PaperDiseaseClassifier(
        num_classes=Config.NUM_CLASSES,
        dropout=Config.CLASSIFIER_DROPOUT,
        stage_dims=list(Config.PAPER_STAGE_DIMS),
        stage_layout=list(Config.PAPER_STAGE_LAYOUT),
        stage_heads=list(Config.PAPER_STAGE_HEADS),
        stage_reductions=list(Config.PAPER_STAGE_REDUCTIONS),
        cte_channels=Config.PAPER_CTE_CHANNELS,
        expansion_ratio=Config.PAPER_LFFN_EXPANSION_RATIO,
        kernel_size=Config.PAPER_LFFN_KERNEL_SIZE,
        embed_kernel_size=Config.PAPER_EMBED_KERNEL_SIZE,
    )
    return model.to(device)


def _safe_predictions(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_probs, top_classes = probs.max(dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    margins = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
    thresholds = torch.tensor(Config.STAGE2_CLASS_THRESHOLDS, device=probs.device, dtype=probs.dtype)
    accept = (top_probs >= thresholds[top_classes]) & (margins >= float(Config.STAGE2_MIN_MARGIN))
    safe_pred = top_classes.clone()
    safe_pred[~accept] = Config.NUM_CLASSES
    return safe_pred, top_probs, margins


def _compute_classifier_metrics_from_cm(cm: np.ndarray) -> dict:
    precisions = []
    recalls = []
    f1s = []
    total_entries = float(max(1, cm.sum()))

    for cls_id in range(Config.NUM_CLASSES):
        tp = float(cm[cls_id, cls_id])
        fp = float(cm[:, cls_id].sum() - tp)
        fn = float(cm[cls_id, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    accuracy_per_class = []
    for cls_id in range(Config.NUM_CLASSES):
        tp = float(cm[cls_id, cls_id])
        fp = float(cm[:, cls_id].sum() - tp)
        fn = float(cm[cls_id, :].sum() - tp)
        tn = float(total_entries - tp - fp - fn)
        accuracy_per_class.append((tp + tn) / total_entries)

    accepted = cm[:, :Config.NUM_CLASSES].sum()
    total_samples = max(1, int(cm.sum()))
    reject_rate = 1.0 - (accepted / total_samples)
    safety_score = 0.50 * precisions[0] + 0.25 * precisions[1] + 0.25 * precisions[2]

    return {
        "accuracy_per_class": accuracy_per_class,
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1s,
        "accuracy_total": float(sum(cm[c, c] for c in range(Config.NUM_CLASSES)) / total_entries),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "moler_precision": float(precisions[0]),
        "reject_rate": float(reject_rate),
        "safety_score": float(safety_score),
    }


def _evaluate_classifier(model, loader, device, desc: str = "ValCls"):
    model.eval()
    loss_sum = 0.0
    total = 0
    cm = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES + 1), dtype=np.int64)

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=True)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(images)
            loss, _ = hierarchical_classifier_loss(
                outputs,
                labels,
                label_smoothing=Config.CLASSIFIER_LABEL_SMOOTHING,
            )
            probs = model.combined_probabilities(outputs)
            safe_pred, _, _ = _safe_predictions(probs)

            loss_sum += float(loss.item()) * labels.size(0)
            total += labels.size(0)

            for gt, pred in zip(labels.detach().cpu().tolist(), safe_pred.detach().cpu().tolist()):
                cm[int(gt), int(pred)] += 1

            running_metrics = _compute_classifier_metrics_from_cm(cm)
            pbar.set_postfix(
                loss=f"{(loss_sum / max(1, total)):.4f}",
                acc=f"{running_metrics['accuracy_total']:.4f}",
                prec=f"{running_metrics['macro_precision']:.4f}",
                rec=f"{running_metrics['macro_recall']:.4f}",
                f1=f"{running_metrics['macro_f1']:.4f}",
            )

    summary = _compute_classifier_metrics_from_cm(cm)

    return {
        "loss": loss_sum / max(1, total),
        "accuracy_per_class": summary["accuracy_per_class"],
        "precision_per_class": summary["precision_per_class"],
        "recall_per_class": summary["recall_per_class"],
        "f1_per_class": summary["f1_per_class"],
        "accuracy_total": summary["accuracy_total"],
        "macro_precision": summary["macro_precision"],
        "macro_recall": summary["macro_recall"],
        "macro_f1": summary["macro_f1"],
        "moler_precision": summary["moler_precision"],
        "reject_rate": summary["reject_rate"],
        "safety_score": summary["safety_score"],
        "confusion_matrix": cm,
    }


def _bbox_xywh_to_xyxy(bbox_xywh: list[float] | np.ndarray) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def _classifier_label_from_pred(pred_id: int, class_names: list[str]) -> str:
    if 0 <= pred_id < len(class_names):
        return class_names[pred_id]
    return str(getattr(Config, "STAGE2_UNKNOWN_NAME", "unknown"))


@torch.no_grad()
def _save_classifier_visualizations(
    model: PaperDiseaseClassifier,
    dataset: DiseaseCropDataset,
    device: torch.device,
    output_dir: Path,
    class_names: list[str],
    max_images: int = 100,
    prefix: str = "test",
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_to_samples: dict[int, list[dict]] = {}
    for sample in dataset.samples:
        image_to_samples.setdefault(int(sample["image_id"]), []).append(sample)

    saved = 0
    for image_id, samples in image_to_samples.items():
        if saved >= max_images:
            break

        image_rgb = dataset._load_image(image_id)
        vis_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        boxes = []
        labels = []
        scores = []

        for sample in samples:
            crop = dataset._crop_with_context(image_rgb, sample["bbox_xywh"])
            if dataset.transform is not None:
                transformed = dataset.transform(image=crop)
                crop_tensor = transformed["image"]
            else:
                crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

            if isinstance(crop_tensor, np.ndarray):
                crop_tensor = torch.from_numpy(crop_tensor).permute(2, 0, 1).float()
            else:
                crop_tensor = crop_tensor.float()

            crop_tensor = crop_tensor.unsqueeze(0).to(device, non_blocking=True)
            outputs = model(crop_tensor)
            probs = model.combined_probabilities(outputs)
            safe_pred, top_probs, _ = _safe_predictions(probs)

            pred_id = int(safe_pred[0].item())
            boxes.append(_bbox_xywh_to_xyxy(sample["bbox_xywh"]))
            labels.append(_classifier_label_from_pred(pred_id, class_names))
            scores.append(float(top_probs[0].item()))

        if boxes:
            vis_image = draw_bounding_boxes(
                vis_image,
                boxes,
                labels,
                scores=scores,
                class_names=class_names,
                thickness=2,
                box_format="xyxy",
            )

        image_info = dataset.images_info[image_id]
        image_name = Path(image_info["file_name"]).stem
        save_path = output_dir / f"{prefix}_{saved + 1:03d}_{image_name}.jpg"
        cv2.imwrite(str(save_path), vis_image)
        saved += 1

    return saved


def _metric_bundle_from_eval(metrics: dict) -> dict:
    return {
        "Accuracy": {
            "per_class": list(metrics["accuracy_per_class"]),
            "global": float(metrics["accuracy_total"]),
        },
        "Precision": {
            "per_class": list(metrics["precision_per_class"]),
            "global": float(metrics["macro_precision"]),
        },
        "Recall": {
            "per_class": list(metrics["recall_per_class"]),
            "global": float(metrics["macro_recall"]),
        },
        "F1-Score": {
            "per_class": list(metrics["f1_per_class"]),
            "global": float(metrics["macro_f1"]),
        },
        "Safety Score": {
            "per_class": [float("nan")] * Config.NUM_CLASSES,
            "global": float(metrics["safety_score"]),
        },
        "Reject Rate": {
            "per_class": [float("nan")] * Config.NUM_CLASSES,
            "global": float(metrics["reject_rate"]),
        },
        "Loss": {
            "per_class": [float("nan")] * Config.NUM_CLASSES,
            "global": float(metrics["loss"]),
        },
    }


def _format_metric_value(value: float) -> str:
    if value != value:
        return "      -"
    return f"{value:>10.4f}"


def _log_metric_table(logger: logging.Logger, title: str, class_names: list[str], bundle: dict) -> None:
    logger.info("  +============================+===============+===============+===============+===============+")
    logger.info(f"  |{title:^92}|")
    logger.info("  +============================+===============+===============+===============+===============+")
    logger.info(
        f"  | {'METRIK':<26} | {class_names[0].upper():>13} | {class_names[1].upper():>13} | "
        f"{class_names[2].upper():>13} | {'GLOBAL':>13} |"
    )
    logger.info("  +----------------------------+---------------+---------------+---------------+---------------+")
    for metric_name, metric_values in bundle.items():
        per_class = metric_values["per_class"]
        global_value = metric_values["global"]
        logger.info(
            f"  | {metric_name:<26} |"
            f" {_format_metric_value(per_class[0])} |"
            f" {_format_metric_value(per_class[1])} |"
            f" {_format_metric_value(per_class[2])} |"
            f" {_format_metric_value(global_value)} |"
        )
    logger.info("  +============================+===============+===============+===============+===============+")


def _savefig(graph_dir: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(graph_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_train_val_loss(graph_dir: Path, epochs, train_loss_hist, val_loss_hist) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_hist, label="Train", color="royalblue", linewidth=1.8)
    plt.plot(epochs, val_loss_hist, label="Val", color="tomato", linewidth=1.8, marker="o", markersize=3)
    plt.title("Classifier Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    _savefig(graph_dir, "classifier_loss.png")


def _plot_dual_metric(
    graph_dir: Path,
    x_train,
    y_train,
    x_val,
    y_val,
    title: str,
    ylabel: str,
    filename: str,
    train_color: str,
    val_color: str,
) -> None:
    plt.figure(figsize=(8, 5))
    if x_train and len(x_train) == len(y_train):
        plt.plot(x_train, y_train, label="Train", color=train_color, linewidth=1.8, marker="o", markersize=3)
    if x_val and len(x_val) == len(y_val):
        plt.plot(x_val, y_val, label="Val", color=val_color, linewidth=1.8, marker="s", markersize=3)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    _savefig(graph_dir, filename)


def _plot_single_metric(graph_dir: Path, epochs, values, title: str, ylabel: str, filename: str, color: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, color=color, linewidth=1.8, marker="o", markersize=3)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    _savefig(graph_dir, filename)


def _plot_per_class_metric(
    graph_dir: Path,
    x_train,
    train_per_class_hist,
    x_val,
    val_per_class_hist,
    class_names,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    colors = ["crimson", "forestgreen", "royalblue"]
    val_colors = ["lightcoral", "limegreen", "cornflowerblue"]
    markers = ["o", "s", "^"]
    plt.figure(figsize=(8, 5))
    for idx, class_name in enumerate(class_names):
        if x_train and len(x_train) == len(train_per_class_hist[idx]):
            plt.plot(
                x_train,
                train_per_class_hist[idx],
                label=f"Train {class_name}",
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linewidth=1.6,
                markersize=3,
            )
        if x_val and len(x_val) == len(val_per_class_hist[idx]):
            plt.plot(
                x_val,
                val_per_class_hist[idx],
                label=f"Val {class_name}",
                color=val_colors[idx % len(val_colors)],
                marker=markers[idx % len(markers)],
                linewidth=1.6,
                markersize=3,
            )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    _savefig(graph_dir, filename)


def _plot_confusion_matrix(
    graph_dir: Path,
    cm: np.ndarray,
    class_names: list[str],
    filename: str = "classifier_confusion_matrix.png",
    title: str = "Classifier Confusion Matrix",
) -> None:
    labels = list(class_names) + [Config.STAGE2_UNKNOWN_NAME]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks[:-1], labels[:-1])

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    _savefig(graph_dir, filename)


def _update_classifier_plots(
    graph_dir: Path,
    train_loss_hist,
    val_loss_hist,
    train_acc_hist,
    val_acc_hist,
    train_safety_score_hist,
    safety_score_hist,
    train_reject_rate_hist,
    reject_rate_hist,
    train_macro_prec_hist,
    macro_prec_hist,
    train_macro_rec_hist,
    macro_rec_hist,
    train_macro_f1_hist,
    macro_f1_hist,
    train_per_class_acc_hist,
    val_per_class_acc_hist,
    train_per_class_prec_hist,
    per_class_prec_hist,
    train_per_class_rec_hist,
    per_class_rec_hist,
    train_per_class_f1_hist,
    per_class_f1_hist,
    class_names,
    confusion_matrix,
) -> None:
    epochs = list(range(1, len(train_loss_hist) + 1))
    _plot_train_val_loss(graph_dir, epochs, train_loss_hist, val_loss_hist)
    _plot_dual_metric(graph_dir, epochs, train_acc_hist, epochs, val_acc_hist, "Train vs Val Accuracy", "Accuracy", "classifier_accuracy.png", "green", "limegreen")
    _plot_dual_metric(graph_dir, epochs, train_safety_score_hist, epochs, safety_score_hist, "Train vs Val Safety Score", "Score", "classifier_safety_score.png", "darkgreen", "seagreen")
    _plot_dual_metric(graph_dir, epochs, train_reject_rate_hist, epochs, reject_rate_hist, "Train vs Val Reject Rate", "Rate", "classifier_reject_rate.png", "darkorange", "goldenrod")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_macro_prec_hist, label="Train Macro Precision", color="royalblue", marker="o", linewidth=1.6, markersize=3)
    plt.plot(epochs, macro_prec_hist, label="Val Macro Precision", color="cornflowerblue", marker="o", linewidth=1.6, markersize=3)
    plt.plot(epochs, train_macro_rec_hist, label="Train Macro Recall", color="darkorange", marker="s", linewidth=1.6, markersize=3)
    plt.plot(epochs, macro_rec_hist, label="Val Macro Recall", color="orange", marker="s", linewidth=1.6, markersize=3)
    plt.plot(epochs, train_macro_f1_hist, label="Train Macro F1", color="purple", marker="^", linewidth=1.6, markersize=3)
    plt.plot(epochs, macro_f1_hist, label="Val Macro F1", color="mediumpurple", marker="^", linewidth=1.6, markersize=3)
    plt.title("Classifier Train vs Val Macro Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.legend()
    _savefig(graph_dir, "classifier_macro_metrics.png")

    _plot_per_class_metric(
        graph_dir, epochs, train_per_class_acc_hist, epochs, val_per_class_acc_hist, class_names,
        "Train vs Val Accuracy Per Class", "Accuracy", "classifier_accuracy_per_class.png",
    )
    _plot_per_class_metric(
        graph_dir, epochs, train_per_class_prec_hist, epochs, per_class_prec_hist, class_names,
        "Precision Per Class", "Precision", "classifier_precision_per_class.png",
    )
    _plot_per_class_metric(
        graph_dir, epochs, train_per_class_rec_hist, epochs, per_class_rec_hist, class_names,
        "Recall Per Class", "Recall", "classifier_recall_per_class.png",
    )
    _plot_per_class_metric(
        graph_dir, epochs, train_per_class_f1_hist, epochs, per_class_f1_hist, class_names,
        "F1 Per Class", "F1-Score", "classifier_f1_per_class.png",
    )
    _plot_confusion_matrix(graph_dir, confusion_matrix, class_names)


def train_classifier(args: argparse.Namespace | None = None) -> dict:
    device = Config.DEVICE
    resume_checkpoint = _resolve_classifier_resume_checkpoint(getattr(args, "resume", None))
    run_dir = _setup_run_dir(resume_checkpoint)
    graph_dir = run_dir / "graphs"
    vis_dir = run_dir / "test_results" / "classifier_predictions"
    logger = _setup_logger(run_dir)
    class_names = list(Config.COCO_CLASSES[:Config.NUM_CLASSES])

    torch.set_num_threads(int(getattr(Config, "CLASSIFIER_BLAS_NUM_THREADS", 1)))

    logger.info("=" * 70)
    logger.info("MULAI TRAINING CLASSIFIER STAGE-2")
    logger.info(f"Run Dir            : {run_dir}")
    logger.info(f"Device             : {device}")
    logger.info(f"Image Size         : {Config.CLASSIFIER_IMAGE_SIZE}")
    logger.info(f"Batch Size         : {Config.CLASSIFIER_BATCH_SIZE}")
    logger.info(f"Learning Rate      : {Config.CLASSIFIER_LEARNING_RATE}")
    logger.info(f"Epochs             : {Config.CLASSIFIER_EPOCHS}")
    logger.info(f"Crop Padding       : {Config.CLASSIFIER_CROP_PADDING}")
    logger.info(f"Safe Thresholds    : {Config.STAGE2_CLASS_THRESHOLDS}")
    logger.info(f"Safe Margin        : {Config.STAGE2_MIN_MARGIN}")
    logger.info(f"Classifier Workers : {getattr(Config, 'CLASSIFIER_NUM_WORKERS', Config.NUM_WORKERS)}")
    logger.info(f"Resume Checkpoint  : {resume_checkpoint if resume_checkpoint is not None else '-'}")
    logger.info("=" * 70)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = _make_loaders()
    logger.info(f"Train Crops        : {len(train_ds):,}")
    logger.info(f"Val Crops          : {len(val_ds):,}")
    logger.info(f"Test Crops         : {len(test_ds):,}")
    logger.info("=" * 70)

    model = _build_model(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.CLASSIFIER_LEARNING_RATE,
        weight_decay=Config.CLASSIFIER_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.CLASSIFIER_EPOCHS)
    scaler = GradScaler(enabled=bool(Config.USE_AMP and device.type == "cuda"))

    start_epoch = 0
    best_score = -1.0
    best_epoch = 0
    best_path = run_dir / "checkpoints" / Config.CLASSIFIER_CHECKPOINT_NAME
    latest_run_path = run_dir / "checkpoints" / "latest_classifier_checkpoint.pth"
    latest_path = Path(Config.CLASSIFIER_CHECKPOINT_PATH)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    train_safety_score_hist = []
    safety_score_hist = []
    train_reject_rate_hist = []
    reject_rate_hist = []
    train_macro_prec_hist = []
    macro_prec_hist = []
    train_macro_rec_hist = []
    macro_rec_hist = []
    train_macro_f1_hist = []
    macro_f1_hist = []
    train_per_class_acc_hist = [[] for _ in range(Config.NUM_CLASSES)]
    val_per_class_acc_hist = [[] for _ in range(Config.NUM_CLASSES)]
    train_per_class_prec_hist = [[] for _ in range(Config.NUM_CLASSES)]
    per_class_prec_hist = [[] for _ in range(Config.NUM_CLASSES)]
    train_per_class_rec_hist = [[] for _ in range(Config.NUM_CLASSES)]
    per_class_rec_hist = [[] for _ in range(Config.NUM_CLASSES)]
    train_per_class_f1_hist = [[] for _ in range(Config.NUM_CLASSES)]
    per_class_f1_hist = [[] for _ in range(Config.NUM_CLASSES)]
    latest_confusion_matrix = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES + 1), dtype=np.int64)
    elapsed_time_offset = 0.0
    training_session_start = time.time()

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                logger.warning("State GradScaler classifier tidak dapat dimuat. Resume tetap dilanjutkan.")

        start_epoch = int(ckpt.get("epoch", 0))
        best_score = float(ckpt.get("best_score", best_score))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        train_loss_hist = ckpt.get("train_loss_hist", train_loss_hist)
        val_loss_hist = ckpt.get("val_loss_hist", val_loss_hist)
        train_acc_hist = ckpt.get("train_acc_hist", train_acc_hist)
        val_acc_hist = ckpt.get("val_acc_hist", val_acc_hist)
        train_safety_score_hist = ckpt.get("train_safety_score_hist", train_safety_score_hist)
        safety_score_hist = ckpt.get("safety_score_hist", safety_score_hist)
        train_reject_rate_hist = ckpt.get("train_reject_rate_hist", train_reject_rate_hist)
        reject_rate_hist = ckpt.get("reject_rate_hist", reject_rate_hist)
        train_macro_prec_hist = ckpt.get("train_macro_prec_hist", train_macro_prec_hist)
        macro_prec_hist = ckpt.get("macro_prec_hist", macro_prec_hist)
        train_macro_rec_hist = ckpt.get("train_macro_rec_hist", train_macro_rec_hist)
        macro_rec_hist = ckpt.get("macro_rec_hist", macro_rec_hist)
        train_macro_f1_hist = ckpt.get("train_macro_f1_hist", train_macro_f1_hist)
        macro_f1_hist = ckpt.get("macro_f1_hist", macro_f1_hist)
        train_per_class_acc_hist = ckpt.get("train_per_class_acc_hist", train_per_class_acc_hist)
        val_per_class_acc_hist = ckpt.get("val_per_class_acc_hist", val_per_class_acc_hist)
        train_per_class_prec_hist = ckpt.get("train_per_class_prec_hist", train_per_class_prec_hist)
        per_class_prec_hist = ckpt.get("per_class_prec_hist", per_class_prec_hist)
        train_per_class_rec_hist = ckpt.get("train_per_class_rec_hist", train_per_class_rec_hist)
        per_class_rec_hist = ckpt.get("per_class_rec_hist", per_class_rec_hist)
        train_per_class_f1_hist = ckpt.get("train_per_class_f1_hist", train_per_class_f1_hist)
        per_class_f1_hist = ckpt.get("per_class_f1_hist", per_class_f1_hist)
        latest_confusion_matrix = ckpt.get("latest_confusion_matrix", latest_confusion_matrix)
        elapsed_time_offset = float(ckpt.get("total_train_time", elapsed_time_offset))
        training_session_start = time.time()
        logger.info(f"Resume classifier dari epoch {start_epoch}, best_score={best_score:.4f}")
        logger.info(f"Akumulasi waktu classifier: {_format_time(elapsed_time_offset)}")

    for epoch in range(start_epoch + 1, Config.CLASSIFIER_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        start = time.time()
        train_cm_running = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES + 1), dtype=np.int64)

        pbar = tqdm(train_loader, desc=f"TrainCls {epoch:03d}", leave=True)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=bool(Config.USE_AMP and device.type == "cuda")):
                outputs = model(images)
                loss, _ = hierarchical_classifier_loss(
                    outputs,
                    labels,
                    label_smoothing=Config.CLASSIFIER_LABEL_SMOOTHING,
                )
                probs = model.combined_probabilities(outputs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item()) * labels.size(0)
            num_samples += labels.size(0)
            safe_pred, _, _ = _safe_predictions(probs.detach())
            for gt, pred in zip(labels.detach().cpu().tolist(), safe_pred.detach().cpu().tolist()):
                train_cm_running[int(gt), int(pred)] += 1
            running_metrics = _compute_classifier_metrics_from_cm(train_cm_running)
            pbar.set_postfix(
                loss=f"{(epoch_loss / max(1, num_samples)):.4f}",
                acc=f"{running_metrics['accuracy_total']:.4f}",
                prec=f"{running_metrics['macro_precision']:.4f}",
                rec=f"{running_metrics['macro_recall']:.4f}",
                f1=f"{running_metrics['macro_f1']:.4f}",
            )

        scheduler.step()
        train_loss = epoch_loss / max(1, num_samples)
        train_metrics = _evaluate_classifier(model, train_loader, device, desc=f"TrainMetricCls {epoch:03d}")
        val_metrics = _evaluate_classifier(model, val_loader, device, desc=f"ValCls {epoch:03d}")
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_metrics["loss"])
        train_acc_hist.append(train_metrics["accuracy_total"])
        val_acc_hist.append(val_metrics["accuracy_total"])
        train_safety_score_hist.append(train_metrics["safety_score"])
        safety_score_hist.append(val_metrics["safety_score"])
        train_reject_rate_hist.append(train_metrics["reject_rate"])
        reject_rate_hist.append(val_metrics["reject_rate"])
        train_macro_prec_hist.append(train_metrics["macro_precision"])
        macro_prec_hist.append(val_metrics["macro_precision"])
        train_macro_rec_hist.append(train_metrics["macro_recall"])
        macro_rec_hist.append(val_metrics["macro_recall"])
        train_macro_f1_hist.append(train_metrics["macro_f1"])
        macro_f1_hist.append(val_metrics["macro_f1"])
        latest_confusion_matrix = val_metrics["confusion_matrix"]
        for class_idx in range(Config.NUM_CLASSES):
            train_per_class_acc_hist[class_idx].append(train_metrics["accuracy_per_class"][class_idx])
            val_per_class_acc_hist[class_idx].append(val_metrics["accuracy_per_class"][class_idx])
            train_per_class_prec_hist[class_idx].append(train_metrics["precision_per_class"][class_idx])
            per_class_prec_hist[class_idx].append(val_metrics["precision_per_class"][class_idx])
            train_per_class_rec_hist[class_idx].append(train_metrics["recall_per_class"][class_idx])
            per_class_rec_hist[class_idx].append(val_metrics["recall_per_class"][class_idx])
            train_per_class_f1_hist[class_idx].append(train_metrics["f1_per_class"][class_idx])
            per_class_f1_hist[class_idx].append(val_metrics["f1_per_class"][class_idx])

        _update_classifier_plots(
            graph_dir=graph_dir,
            train_loss_hist=train_loss_hist,
            val_loss_hist=val_loss_hist,
            train_acc_hist=train_acc_hist,
            val_acc_hist=val_acc_hist,
            train_safety_score_hist=train_safety_score_hist,
            safety_score_hist=safety_score_hist,
            train_reject_rate_hist=train_reject_rate_hist,
            reject_rate_hist=reject_rate_hist,
            train_macro_prec_hist=train_macro_prec_hist,
            macro_prec_hist=macro_prec_hist,
            train_macro_rec_hist=train_macro_rec_hist,
            macro_rec_hist=macro_rec_hist,
            train_macro_f1_hist=train_macro_f1_hist,
            macro_f1_hist=macro_f1_hist,
            train_per_class_acc_hist=train_per_class_acc_hist,
            val_per_class_acc_hist=val_per_class_acc_hist,
            train_per_class_prec_hist=train_per_class_prec_hist,
            per_class_prec_hist=per_class_prec_hist,
            train_per_class_rec_hist=train_per_class_rec_hist,
            per_class_rec_hist=per_class_rec_hist,
            train_per_class_f1_hist=train_per_class_f1_hist,
            per_class_f1_hist=per_class_f1_hist,
            class_names=class_names,
            confusion_matrix=latest_confusion_matrix,
        )

        logger.info(
            f"Epoch {epoch:03d}/{Config.CLASSIFIER_EPOCHS} | "
            f"time={time.time() - start:.1f}s | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_metrics['loss']:.4f} | "
            f"TrainAcc={train_metrics['accuracy_total']:.4f} | ValAcc={val_metrics['accuracy_total']:.4f} | "
            f"TrainPrec={train_metrics['macro_precision']:.4f} | ValPrec={val_metrics['macro_precision']:.4f} | "
            f"TrainRec={train_metrics['macro_recall']:.4f} | ValRec={val_metrics['macro_recall']:.4f} | "
            f"TrainF1={train_metrics['macro_f1']:.4f} | ValF1={val_metrics['macro_f1']:.4f} | "
            f"SafetyScore={val_metrics['safety_score']:.4f} | "
            f"MolerPrec={val_metrics['moler_precision']:.4f} | "
            f"RejectRate={val_metrics['reject_rate']:.4f}"
        )

        if val_metrics["safety_score"] > best_score:
            best_score = val_metrics["safety_score"]
            best_epoch = epoch
            payload = {
                "epoch": epoch,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "total_train_time": _get_realtime_elapsed(training_session_start, elapsed_time_offset),
                "val_metrics": val_metrics,
            }
            torch.save(payload, best_path)
            torch.save(payload, latest_path)
            logger.info(f"Best classifier checkpoint diperbarui: {best_path}")
            logger.info(f"Latest classifier checkpoint diperbarui: {latest_path}")

        latest_payload = {
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss_hist": train_loss_hist,
            "val_loss_hist": val_loss_hist,
            "train_acc_hist": train_acc_hist,
            "val_acc_hist": val_acc_hist,
            "train_safety_score_hist": train_safety_score_hist,
            "safety_score_hist": safety_score_hist,
            "train_reject_rate_hist": train_reject_rate_hist,
            "reject_rate_hist": reject_rate_hist,
            "train_macro_prec_hist": train_macro_prec_hist,
            "macro_prec_hist": macro_prec_hist,
            "train_macro_rec_hist": train_macro_rec_hist,
            "macro_rec_hist": macro_rec_hist,
            "train_macro_f1_hist": train_macro_f1_hist,
            "macro_f1_hist": macro_f1_hist,
            "train_per_class_acc_hist": train_per_class_acc_hist,
            "val_per_class_acc_hist": val_per_class_acc_hist,
            "train_per_class_prec_hist": train_per_class_prec_hist,
            "per_class_prec_hist": per_class_prec_hist,
            "train_per_class_rec_hist": train_per_class_rec_hist,
            "per_class_rec_hist": per_class_rec_hist,
            "train_per_class_f1_hist": train_per_class_f1_hist,
            "per_class_f1_hist": per_class_f1_hist,
            "latest_confusion_matrix": latest_confusion_matrix,
            "total_train_time": _get_realtime_elapsed(training_session_start, elapsed_time_offset),
        }
        torch.save(latest_payload, latest_run_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_train_metrics = _evaluate_classifier(model, train_loader, device, desc="TrainBestCls")
    best_val_metrics = _evaluate_classifier(model, val_loader, device, desc="ValBestCls")
    test_metrics = _evaluate_classifier(model, test_loader, device, desc="TestCls")

    best_train_bundle = _metric_bundle_from_eval(best_train_metrics)
    best_val_bundle = _metric_bundle_from_eval(best_val_metrics)
    test_bundle = _metric_bundle_from_eval(test_metrics)

    _plot_confusion_matrix(
        graph_dir,
        best_val_metrics["confusion_matrix"],
        class_names,
        filename="classifier_confusion_matrix_val_best.png",
        title="Classifier Confusion Matrix (Best Val)",
    )
    _plot_confusion_matrix(
        graph_dir,
        test_metrics["confusion_matrix"],
        class_names,
        filename="classifier_confusion_matrix_test.png",
        title="Classifier Confusion Matrix (Test)",
    )

    saved_vis_count = 0
    if bool(getattr(Config, "CLASSIFIER_SAVE_VISUALIZATIONS", True)):
        saved_vis_count = _save_classifier_visualizations(
            model=model,
            dataset=test_ds,
            device=device,
            output_dir=vis_dir,
            class_names=class_names,
            max_images=int(getattr(Config, "CLASSIFIER_VIS_MAX_IMAGES", 100)),
            prefix="test",
        )

    logger.info("")
    logger.info("=" * 90)
    logger.info("  RINGKASAN AKHIR TRAINING CLASSIFIER")
    logger.info("=" * 90)
    logger.info(
        f"  Total Waktu Training  : "
        f"{_format_time(_get_realtime_elapsed(training_session_start, elapsed_time_offset))}"
    )
    logger.info(f"  Best Val Safety Score : {best_score:.4f}  (Epoch {best_epoch})")
    logger.info(f"  Best Val Macro Prec   : {best_val_metrics['macro_precision']:.4f}")
    logger.info(f"  Best Val Macro Recall : {best_val_metrics['macro_recall']:.4f}")
    logger.info(f"  Best Val Macro F1     : {best_val_metrics['macro_f1']:.4f}")
    logger.info(f"  Best Val Reject Rate  : {best_val_metrics['reject_rate']:.4f}")
    logger.info("-" * 90)
    _log_metric_table(logger, "BEST VALIDASI", class_names, best_val_bundle)
    logger.info("-" * 90)
    _log_metric_table(logger, "BEST TRAIN", class_names, best_train_bundle)
    logger.info("-" * 90)
    _log_metric_table(logger, "TEST", class_names, test_bundle)
    logger.info("-" * 90)
    logger.info(f"  Test Safety Score     : {test_metrics['safety_score']:.4f}")
    logger.info(f"  Test Macro Prec       : {test_metrics['macro_precision']:.4f}")
    logger.info(f"  Test Macro Recall     : {test_metrics['macro_recall']:.4f}")
    logger.info(f"  Test Macro F1         : {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Test Reject Rate      : {test_metrics['reject_rate']:.4f}")
    logger.info(f"  Best classifier checkpoint : {best_path}")
    logger.info(f"  Latest classifier checkpoint: {latest_path}")
    logger.info(f"  Grafik classifier tersimpan di: {graph_dir}")
    if bool(getattr(Config, "CLASSIFIER_SAVE_VISUALIZATIONS", True)):
        logger.info(f"  Visualisasi classifier tersimpan di: {vis_dir}  ({saved_vis_count} gambar)")
    logger.info("=" * 90)
    return {
        "run_dir": run_dir,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_path": best_path,
        "latest_path": latest_path,
        "graph_dir": graph_dir,
        "vis_dir": vis_dir,
        "saved_vis_count": saved_vis_count,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path file/folder checkpoint classifier untuk resume.")
    train_classifier(parser.parse_args())
