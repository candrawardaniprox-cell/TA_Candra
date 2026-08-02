from __future__ import annotations
"""
test_illumination.py — Evaluasi model terbaik pada dataset test dengan pencahayaan buatan.

Script ini me-load best model dari run_20260612_204115_Data100% dan
mengevaluasi performanya pada 3 dataset illumination (Terang, Normal, Gelap).

Hasil (metrik, confusion matrix, comparison images) disimpan di folder output terpisah.
"""

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

from config import Config
from data import ObjectDetectionDataset, get_val_transforms
from data.utils import collate_fn
from utils import AnchorFreeLoss
from utils.metrics_fixed import (
    generate_confusion_matrix,
    generate_detection_confusion_matrix,
)
from train import (
    build_detector,
    create_comparison_images,
    evaluate,
    extract_multiclass_bundle,
    extract_multiclass_global_metrics,
    log_metric_table,
)

# ======================== KONFIGURASI ========================
CHECKPOINT_PATH = Path("outputs") / "run_20260731_125110" / "checkpoints" / "best_model.pth" 

ILLUMINATION_DATASETS = {
    "terang": {
        "label": "Siang Terik (Gamma=2.0)",
        "data_dir": Path("data") / "illumination_test_terang",
        "output_dir": Path("outputs") / "illumination_test_terang",
    },
    "normal": {
        "label": "Baseline Asli (Gamma=1.0)",
        "data_dir": Path("data") / "illumination_test_normal",
        "output_dir": Path("outputs") / "illumination_test_normal",
    },
    "gelap": {
        "label": "Mendung/Malam (Gamma=0.3)",
        "data_dir": Path("data") / "illumination_test_gelap",
        "output_dir": Path("outputs") / "illumination_test_gelap",
    },
}

def setup_logger(log_dir: Path, key: str) -> logging.Logger:
    logger = logging.getLogger(f'illum_test_{key}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = log_dir / f"test_illumination_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_single_test(
    key: str,
    config_dict: dict,
    model: torch.nn.Module,
    criterion,
    device: torch.device,
    class_names: list,
    val_tf,
) -> dict:

    data_dir = config_dict["data_dir"]
    output_dir = config_dict["output_dir"]
    label = config_dict["label"]

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

    logger = setup_logger(log_dir, key)

    test_images_dir = data_dir / "test2017"
    test_annotations = data_dir / "annotations_coco" / "instances_test2017.json"

    if not test_images_dir.exists() or not test_annotations.exists():
        logger.error(f"[X] Dataset {label} tidak ditemukan di {data_dir}")
        return None

    try:
        test_ds = ObjectDetectionDataset(
            test_images_dir, test_annotations,
            transform=val_tf, image_size=Config.IMAGE_SIZE, repeat_factor=1,
        )
        test_loader = DataLoader(
            test_ds, batch_size=min(Config.BATCH_SIZE, 4),
            shuffle=False, num_workers=0, collate_fn=collate_fn,
        )

        logger.info(f"  Memulai evaluasi {label} ({len(test_ds)} gambar)...")

        test_metrics, _, test_class_preds, test_det_preds, test_tgts, sample_imgs, sample_tgts_list = evaluate(
            model, test_loader, criterion, device, 0,
            label_prefix=f"Illum-{key}", collect_samples=True, logger=logger,
        )

        test_multi_bundle = extract_multiclass_bundle(test_metrics, Config.NUM_CLASSES)
        test_multi_globals = extract_multiclass_global_metrics(test_metrics)

        log_metric_table(logger, f"TEST {label.upper()}", class_names, test_multi_bundle)

        generate_confusion_matrix(
            test_class_preds, test_tgts, Config.NUM_CLASSES,
            class_names=class_names, fname=graphs_dir / f'confusion_matrix_class_{key}.png',
        )
        if len(sample_imgs) > 0:
            create_comparison_images(
                sample_imgs, sample_tgts_list, test_class_preds[:len(sample_imgs)], 0, class_names,
            )

        result = {
            'key': key,
            'label': label,
            'multi_bundle': test_multi_bundle,
            'multi_globals': test_multi_globals,
        }

    except Exception as exc:
        logger.error(f"Error: {exc}")
        result = None

    finally:
        Config.TEST_RESULT_DIR = original_test_result_dir
        Config.GRAPHS_DIR = original_graphs_dir
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result

def print_comparison_table(results: list, class_names: list):
    sep = "=" * 80
    print(f"\n{sep}")
    print("  PERBANDINGAN KINERJA MODEL PADA KONDISI PENCAHAYAAN BERBEDA")
    print(sep)

    header = f"  {'Metrik':<20}"
    for r in results:
        header += f" | {r['label']:>25}"
    print(header)
    print("  " + "-" * (20 + (28 * len(results))))

    metric_keys = [
        ('mAP@0.50', 'mAP@0.50', 'average'),
        ('mAP@[0.50:0.95]', 'mAP@[0.50:0.95]', 'average'),
        ('Average Accuracy', None, 'Average Accuracy'),
        ('Average F1', None, 'Average F1'),
    ]

    for display_name, bundle_key, global_key in metric_keys:
        row = f"  {display_name:<20}"
        for r in results:
            if bundle_key and bundle_key in r['multi_bundle']:
                val = r['multi_bundle'][bundle_key]['average']
            elif global_key in r['multi_globals']:
                val = r['multi_globals'][global_key]
            else:
                val = 0.0
            row += f" | {val:>25.4f}"
        print(row)
    print(sep)

    print(f"\n{sep}")
    print("  BREAKDOWN PER KELAS: mAP@0.50")
    print(sep)
    header_cls = f"  {'Kelas':<20}"
    for r in results:
        header_cls += f" | {r['key'].upper():>25}"
    print(header_cls)
    print("  " + "-" * (20 + (28 * len(results))))

    for cls_idx, cls_name in enumerate(class_names):
        row = f"  {cls_name:<20}"
        for r in results:
            val = r['multi_bundle']['mAP@0.50']['per_class'][cls_idx]
            row += f" | {val:>25.4f}"
        print(row)
    print(sep)


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"[X] Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")
        sys.exit(1)

    print("=" * 80)
    print("  EVALUASI MODEL PADA DATA TEST DENGAN ILLUMINATION BERBEDA")
    print("=" * 80)

    device = Config.DEVICE
    model = build_detector().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  [OK] Model dimuat dari Epoch {ckpt.get('epoch', '?')}")

    criterion = AnchorFreeLoss(num_classes=Config.NUM_CLASSES)
    val_tf = get_val_transforms(Config.IMAGE_SIZE, Config.MEAN, Config.STD)
    class_names = Config.COCO_CLASSES

    all_results = []
    # Run in specific order: Normal (Baseline) -> Terang -> Gelap
    order = ["normal", "terang", "gelap"]
    
    for key in order:
        config = ILLUMINATION_DATASETS[key]
        print(f"\n{'-' * 80}")
        print(f"  Mengevaluasi: {config['label']}")
        print(f"{'-' * 80}")
        
        result = run_single_test(
            key, config, model, criterion, device, class_names, val_tf
        )
        if result:
            all_results.append(result)
            print(f"  [OK] Selesai")

    if all_results:
        print_comparison_table(all_results, class_names)

if __name__ == "__main__":
    main()
