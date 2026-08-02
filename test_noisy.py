"""
test_noisy.py — Evaluasi model terbaik pada dataset test yang sudah diberi noise.

Script ini me-load best model dari run_20260612_204115_Data100% dan
mengevaluasi performanya pada semua dataset noise (3 jenis x 3 level)
yang digenerate oleh generate_noisy_test.py.

Hasil (metrik, confusion matrix, comparison images) disimpan di folder
output terpisah untuk setiap konfigurasi noise.

TIDAK mengubah config.py, train.py, atau file model lainnya.

Cara pakai:
    1. Jalankan dulu: python generate_noisy_test.py
    2. Lalu:          python test_noisy.py

Output (9 folder):
    outputs/noise_test_gaussian_rendah/
    outputs/noise_test_gaussian_sedang/
    outputs/noise_test_gaussian_tinggi/
    outputs/noise_test_salt_pepper_rendah/
    outputs/noise_test_salt_pepper_sedang/
    outputs/noise_test_salt_pepper_tinggi/
    outputs/noise_test_poisson_rendah/
    outputs/noise_test_poisson_sedang/
    outputs/noise_test_poisson_tinggi/
"""
from __future__ import annotations

import gc
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import dari proyek yang sudah ada (tanpa mengubahnya)
from config import Config
from data import ObjectDetectionDataset, get_val_transforms
from data.utils import collate_fn
from models import HybridDetector, PlainCNNDetector, ResNetDetector, VGGDetector
from utils import AnchorFreeLoss, calculate_map
from utils.metrics_fixed import (
    calculate_multiclass_metrics,
    generate_confusion_matrix,
    generate_detection_confusion_matrix,
)
from utils.visualization import draw_bounding_boxes
from train import (
    MAP_IOU_THRESHOLDS,
    MULTI_PER_CLASS_ROW_ORDER,
    build_detector,
    create_comparison_images,
    evaluate,
    extract_multiclass_bundle,
    extract_multiclass_global_metrics,
    init_best_global_metrics,
    init_metric_bundle,
    log_metric_table,
    log_global_metric_table,
    sanitize_targets,
    _chunked_map,
)

# ======================== KONFIGURASI ========================
CHECKPOINT_PATH = Path("outputs") / "run_20260731_125110" / "checkpoints" / "best_model.pth"

# Definisi 9 dataset noise (3 jenis x 3 level)
NOISE_TYPES = ["gaussian", "salt_pepper", "poisson"]
NOISE_LEVELS = ["rendah", "sedang", "tinggi"]

NOISE_LABELS = {
    "gaussian":     "Gaussian",
    "salt_pepper":  "Salt-and-Pepper",
    "poisson":      "Poisson",
}
LEVEL_LABELS = {
    "rendah": "Rendah",
    "sedang": "Sedang",
    "tinggi": "Tinggi",
}

# Parameter untuk label tabel
NOISE_PARAMS = {
    "gaussian":    {"rendah": "sigma=10", "sedang": "sigma=25", "tinggi": "sigma=50"},
    "salt_pepper": {"rendah": "1%",       "sedang": "5%",       "tinggi": "10%"},
    "poisson":     {"rendah": "scale=60", "sedang": "scale=25", "tinggi": "scale=8"},
}


def build_noise_datasets():
    """Bangun daftar semua dataset noise yang akan dievaluasi."""
    datasets = {}
    for noise_type in NOISE_TYPES:
        for level in NOISE_LEVELS:
            key = f"{noise_type}_{level}"
            noise_label = NOISE_LABELS[noise_type]
            level_label = LEVEL_LABELS[level]
            param_label = NOISE_PARAMS[noise_type][level]
            datasets[key] = {
                "noise_type": noise_type,
                "level": level,
                "label": f"{noise_label} {level_label} ({param_label})",
                "short_label": f"{noise_label} {level_label}",
                "data_dir": Path("data") / f"noisy_test_{noise_type}_{level}",
                "output_dir": Path("outputs") / f"noise_test_{noise_type}_{level}",
            }
    return datasets


def setup_logger(log_dir: Path, noise_key: str) -> logging.Logger:
    """Buat logger khusus per noise test."""
    logger = logging.getLogger(f'noise_test_{noise_key}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = log_dir / f"test_noise_{noise_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def run_single_noise_test(
    noise_key: str,
    noise_config: dict,
    model: torch.nn.Module,
    criterion,
    device: torch.device,
    class_names: list,
    val_tf,
) -> dict:
    """Jalankan evaluasi pada satu dataset noise."""

    data_dir = noise_config["data_dir"]
    output_dir = noise_config["output_dir"]
    label = noise_config["label"]

    # Setup dirs
    test_result_dir = output_dir / "test_results"
    graphs_dir = output_dir / "graphs" / "multi_label"
    log_dir = output_dir / "logs"
    for d in [test_result_dir, graphs_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Override Config paths sementara
    original_test_result_dir = Config.TEST_RESULT_DIR
    original_graphs_dir = Config.GRAPHS_DIR
    Config.TEST_RESULT_DIR = test_result_dir
    Config.GRAPHS_DIR = output_dir / "graphs"

    logger = setup_logger(log_dir, noise_key)

    test_images_dir = data_dir / "test2017"
    test_annotations = data_dir / "annotations_coco" / "instances_test2017.json"

    if not test_images_dir.exists():
        logger.error(f"Folder gambar noise tidak ditemukan: {test_images_dir}")
        logger.error("Jalankan generate_noisy_test.py terlebih dahulu!")
        Config.TEST_RESULT_DIR = original_test_result_dir
        Config.GRAPHS_DIR = original_graphs_dir
        return None

    if not test_annotations.exists():
        logger.error(f"File anotasi noise tidak ditemukan: {test_annotations}")
        Config.TEST_RESULT_DIR = original_test_result_dir
        Config.GRAPHS_DIR = original_graphs_dir
        return None

    sep = "=" * 70
    logger.info(sep)
    logger.info(f"  EVALUASI NOISE TEST: {label}")
    logger.info(sep)
    logger.info(f"  Checkpoint  : {CHECKPOINT_PATH}")
    logger.info(f"  Test Images : {test_images_dir}")
    logger.info(f"  Annotations : {test_annotations}")
    logger.info(f"  Output Dir  : {output_dir}")
    logger.info(sep)

    try:
        test_ds = ObjectDetectionDataset(
            test_images_dir,
            test_annotations,
            transform=val_tf,
            image_size=Config.IMAGE_SIZE,
            repeat_factor=1,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=min(Config.BATCH_SIZE, 4),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        logger.info(f"  Jumlah gambar test: {len(test_ds)}")
        logger.info(f"  Memulai evaluasi pada {label}...")

        test_metrics, _, test_class_preds, test_det_preds, test_tgts, sample_imgs, sample_tgts_list = evaluate(
            model,
            test_loader,
            criterion,
            device,
            0,
            label_prefix=f"Test-{noise_key}",
            collect_samples=True,
            logger=logger,
        )

        test_multi_bundle = extract_multiclass_bundle(test_metrics, Config.NUM_CLASSES)
        test_multi_globals = extract_multiclass_global_metrics(test_metrics)

        # Log hasil
        logger.info(sep)
        logger.info(f"  HASIL: {label}")
        logger.info(sep)
        logger.info(f"  mAP@0.50           = {test_multi_bundle['mAP@0.50']['average']:.4f}")
        logger.info(f"  mAP@[0.50:0.95]    = {test_multi_bundle['mAP@[0.50:0.95]']['average']:.4f}")
        logger.info(f"  Average Accuracy   = {test_multi_globals['Average Accuracy']:.4f}")
        logger.info(f"  System Accuracy    = {test_multi_globals['System Accuracy']:.4f}")
        logger.info(f"  Average Precision  = {test_multi_globals['Average Precision']:.4f}")
        logger.info(f"  System Precision   = {test_multi_globals['System Precision']:.4f}")
        logger.info(f"  Average Recall     = {test_multi_globals['Average Recall']:.4f}")
        logger.info(f"  System Recall      = {test_multi_globals['System Recall']:.4f}")
        logger.info(f"  Average F1         = {test_multi_globals['Average F1']:.4f}")
        logger.info(f"  System F1          = {test_multi_globals['System F1']:.4f}")
        logger.info(sep)

        log_metric_table(logger, f"TEST {label.upper()}", class_names, test_multi_bundle)

        # Confusion matrix
        generate_confusion_matrix(
            test_class_preds, test_tgts, Config.NUM_CLASSES,
            class_names=class_names,
            fname=graphs_dir / f'confusion_matrix_class_{noise_key}.png',
        )
        generate_detection_confusion_matrix(
            test_det_preds, test_tgts, Config.NUM_CLASSES,
            class_names=class_names, iou_threshold=0.5,
            fname=graphs_dir / f'confusion_matrix_detection_{noise_key}.png',
        )
        logger.info(f"  Confusion matrix disimpan di: {graphs_dir}")

        # Comparison images
        if len(sample_imgs) > 0:
            create_comparison_images(
                sample_imgs, sample_tgts_list,
                test_class_preds[:len(sample_imgs)], 0, class_names,
            )
            logger.info(f"  Gambar perbandingan disimpan di: {test_result_dir}")

        result = {
            'noise_key': noise_key,
            'noise_type': noise_config['noise_type'],
            'level': noise_config['level'],
            'label': label,
            'short_label': noise_config['short_label'],
            'multi_bundle': test_multi_bundle,
            'multi_globals': test_multi_globals,
        }

    except Exception as exc:
        logger.error(f"Error saat evaluasi {label}: {exc}")
        logger.error(traceback.format_exc())
        result = None

    finally:
        Config.TEST_RESULT_DIR = original_test_result_dir
        Config.GRAPHS_DIR = original_graphs_dir
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def print_comparison_table(results: list, class_names: list):
    """Cetak tabel perbandingan metrik semua noise, dikelompokkan per level."""
    sep = "=" * 100

    # ===== TABEL 1: Per Level Intensitas =====
    for level in NOISE_LEVELS:
        level_results = [r for r in results if r['level'] == level]
        if not level_results:
            continue

        level_label = LEVEL_LABELS[level]
        print(f"\n{sep}")
        print(f"  PERBANDINGAN NOISE -- INTENSITAS {level_label.upper()}")
        print(sep)

        header = f"  {'Metrik':<28}"
        for r in level_results:
            header += f" | {NOISE_LABELS[r['noise_type']]:>18}"
        print(header)
        print("  " + "-" * (28 + (21 * len(level_results))))

        metric_keys = [
            ('mAP@0.50', 'mAP@0.50', 'average'),
            ('mAP@[0.50:0.95]', 'mAP@[0.50:0.95]', 'average'),
            ('Average Accuracy', None, 'Average Accuracy'),
            ('Average Precision', None, 'Average Precision'),
            ('Average Recall', None, 'Average Recall'),
            ('Average F1', None, 'Average F1'),
        ]

        for display_name, bundle_key, global_key in metric_keys:
            row = f"  {display_name:<28}"
            for r in level_results:
                if bundle_key and bundle_key in r['multi_bundle']:
                    val = r['multi_bundle'][bundle_key]['average']
                elif global_key in r['multi_globals']:
                    val = r['multi_globals'][global_key]
                else:
                    val = 0.0
                row += f" | {val:>18.4f}"
            print(row)
        print(sep)

    # ===== TABEL 2: Per Jenis Noise (efek meningkatkan intensitas) =====
    for noise_type in NOISE_TYPES:
        noise_results = [r for r in results if r['noise_type'] == noise_type]
        if not noise_results:
            continue

        noise_label = NOISE_LABELS[noise_type]
        print(f"\n{sep}")
        print(f"  PENGARUH INTENSITAS -- {noise_label.upper()}")
        print(sep)

        header = f"  {'Metrik':<28}"
        for r in noise_results:
            param = NOISE_PARAMS[noise_type][r['level']]
            header += f" | {LEVEL_LABELS[r['level']]+' ('+param+')':>22}"
        print(header)
        print("  " + "-" * (28 + (25 * len(noise_results))))

        metric_keys = [
            ('mAP@0.50', 'mAP@0.50', 'average'),
            ('mAP@[0.50:0.95]', 'mAP@[0.50:0.95]', 'average'),
            ('Average Accuracy', None, 'Average Accuracy'),
            ('Average Precision', None, 'Average Precision'),
            ('Average Recall', None, 'Average Recall'),
            ('Average F1', None, 'Average F1'),
        ]

        for display_name, bundle_key, global_key in metric_keys:
            row = f"  {display_name:<28}"
            for r in noise_results:
                if bundle_key and bundle_key in r['multi_bundle']:
                    val = r['multi_bundle'][bundle_key]['average']
                elif global_key in r['multi_globals']:
                    val = r['multi_globals'][global_key]
                else:
                    val = 0.0
                row += f" | {val:>22.4f}"
            print(row)
        print(sep)

    # ===== TABEL 3: mAP@0.50 Per Kelas - Semua Konfigurasi =====
    print(f"\n{sep}")
    print("  BREAKDOWN PER KELAS: mAP@0.50 (SEMUA KONFIGURASI)")
    print(sep)

    header = f"  {'Konfigurasi':<35}"
    for cls_name in class_names:
        header += f" | {cls_name:>14}"
    header += f" | {'RATA-RATA':>14}"
    print(header)
    print("  " + "-" * (35 + (17 * (len(class_names) + 1))))

    for r in results:
        row = f"  {r['short_label']:<35}"
        for cls_idx in range(len(class_names)):
            val = r['multi_bundle']['mAP@0.50']['per_class'][cls_idx]
            row += f" | {val:>14.4f}"
        avg = r['multi_bundle']['mAP@0.50']['average']
        row += f" | {avg:>14.4f}"
        print(row)
    print(sep)


def main():
    # Validasi checkpoint
    if not CHECKPOINT_PATH.exists():
        print(f"[X] Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")
        print("  Pastikan training run_20260612_204115_Data100% sudah selesai.")
        sys.exit(1)

    # Build noise datasets
    noise_datasets = build_noise_datasets()

    # Validasi folder noise
    missing = []
    for key, cfg in noise_datasets.items():
        if not (cfg["data_dir"] / "test2017").exists():
            missing.append(key)
    if missing:
        print("[X] Folder noise belum digenerate. Jalankan dulu:")
        print("  python generate_noisy_test.py")
        print(f"  Yang belum ada: {', '.join(missing)}")
        sys.exit(1)

    print("=" * 70)
    print("  EVALUASI MODEL TERBAIK PADA DATA TEST DENGAN NOISE")
    print("  (3 Jenis Noise x 3 Level Intensitas = 9 Eksperimen)")
    print("=" * 70)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Total konfigurasi: {len(noise_datasets)}")
    print("=" * 70)

    # Setup
    device = Config.DEVICE
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = bool(getattr(Config, 'CUDA_BENCHMARK', True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(Config, 'ALLOW_TF32', True))
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = bool(getattr(Config, 'ALLOW_TF32', True))

    # Build model & load checkpoint
    model = build_detector().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  [OK] Model berhasil dimuat (Epoch {ckpt.get('epoch', '?')})")

    criterion = AnchorFreeLoss(num_classes=Config.NUM_CLASSES)
    val_tf = get_val_transforms(Config.IMAGE_SIZE, Config.MEAN, Config.STD)
    class_names = Config.COCO_CLASSES

    # Jalankan evaluasi untuk setiap konfigurasi noise
    all_results = []
    total = len(noise_datasets)
    for idx, (noise_key, noise_config) in enumerate(noise_datasets.items(), 1):
        print(f"\n{'-' * 70}")
        print(f"  [{idx}/{total}] Mengevaluasi: {noise_config['label']}")
        print(f"{'-' * 70}")

        result = run_single_noise_test(
            noise_key=noise_key,
            noise_config=noise_config,
            model=model,
            criterion=criterion,
            device=device,
            class_names=class_names,
            val_tf=val_tf,
        )

        if result is not None:
            all_results.append(result)
            print(f"  [OK] {noise_config['label']} selesai")
        else:
            print(f"  [X] {noise_config['label']} gagal")

    # Cetak tabel perbandingan
    if all_results:
        print_comparison_table(all_results, class_names)

    # Ringkasan
    print(f"\n{'=' * 70}")
    print(f"  SELESAI! {len(all_results)}/{total} evaluasi berhasil.")
    print(f"  Output tersimpan di folder outputs/noise_test_*/")
    print("=" * 70)


if __name__ == "__main__":
    main()
