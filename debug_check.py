"""
debug_check.py — Jalankan SEBELUM training untuk memverifikasi semua fix benar.

Usage:
    python debug_check.py

Pastikan script ini dijalankan dari root folder proyek Anda.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import ObjectDetectionDataset, get_val_transforms, collate_fn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 0 — Verifikasi jumlah kategori dataset vs NUM_CLASSES di config
# ─────────────────────────────────────────────────────────────────────────────
def check_num_classes():
    print("\n" + "=" * 60)
    print("CHECK 0: Jumlah Kelas Dataset vs Config.NUM_CLASSES")
    print("=" * 60)

    transform = get_val_transforms(image_size=Config.IMAGE_SIZE)
    dataset   = ObjectDetectionDataset(
        Config.VAL_IMAGES, Config.VAL_ANNOTATIONS,
        transform=transform, image_size=Config.IMAGE_SIZE,
    )

    num_cats_in_dataset = len(dataset.category_id_to_idx)
    cat_names = [dataset.categories[dataset.idx_to_category_id[i]]['name']
                 for i in range(num_cats_in_dataset)]

    print(f"  Jumlah kategori dalam dataset    : {num_cats_in_dataset}")
    print(f"  Nama kategori (indeks → nama)    :")
    for i, name in enumerate(cat_names):
        print(f"    [{i}] {name}")
    print(f"  Config.NUM_CLASSES               : {Config.NUM_CLASSES}")
    print(f"  Config.COCO_CLASSES              : {Config.COCO_CLASSES}")

    if num_cats_in_dataset != Config.NUM_CLASSES:
        print(f"\n  ✗ MISMATCH! Dataset punya {num_cats_in_dataset} kategori, "
              f"tapi Config.NUM_CLASSES = {Config.NUM_CLASSES}")
        print(f"\n  ══ SOLUSI: Edit config.py ══")
        print(f"  Ubah:")
        print(f"    NUM_CLASSES  = {num_cats_in_dataset}")
        print(f"    COCO_CLASSES = {cat_names}")
        print(f"  ════════════════════════════")
    else:
        print(f"\n  ✓ Jumlah kelas SESUAI ({num_cats_in_dataset} kategori)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Format GT Bounding Boxes
# ─────────────────────────────────────────────────────────────────────────────
def check_bbox_format():
    print("\n" + "=" * 60)
    print("CHECK 1: Format GT Bounding Boxes")
    print("=" * 60)

    transform = get_val_transforms(image_size=Config.IMAGE_SIZE)
    dataset   = ObjectDetectionDataset(
        Config.VAL_IMAGES, Config.VAL_ANNOTATIONS,
        transform=transform, image_size=Config.IMAGE_SIZE,
    )

    sample = dataset[0]
    boxes  = sample['boxes']
    labels = sample['labels']

    print(f"  Jumlah bbox pada sample pertama : {len(boxes)}")
    if len(boxes) > 0:
        print(f"  Format boxes (harus cx,cy,w,h)  :")
        for i, (b, l) in enumerate(zip(boxes, labels)):
            cx, cy, w, h = b.tolist()
            label_idx = l.item()
            print(f"    [{i}] cx={cx:.1f}  cy={cy:.1f}  w={w:.1f}  h={h:.1f}  label={label_idx}")

            # Sanity check: label harus dalam range [0, NUM_CLASSES-1]
            assert 0 <= label_idx < Config.NUM_CLASSES, (
                f"label={label_idx} di luar range [0, {Config.NUM_CLASSES-1}]! "
                f"Pastikan NUM_CLASSES di config.py sudah benar."
            )
            assert 0 < cx < Config.IMAGE_SIZE, f"cx={cx} di luar range gambar!"
            assert 0 < cy < Config.IMAGE_SIZE, f"cy={cy} di luar range gambar!"
            assert w  > 0,                     f"width={w} harus > 0!"
            assert h  > 0,                     f"height={h} harus > 0!"

    print("  ✓ Format GT boxes BENAR (cx,cy,w,h dalam piksel)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Jumlah Anchor Positif di Loss Function
# ─────────────────────────────────────────────────────────────────────────────
def check_num_positives():
    print("\n" + "=" * 60)
    print("CHECK 2: Jumlah Anchor Positif di Loss Function")
    print("=" * 60)

    # ── PERBAIKAN: Import dari nama file yang benar ──────────────────────────
    from models import HybridDetector
    from utils.loss_fixed import AnchorFreeLoss          # ← WAS: utils.loss

    device = torch.device('cpu')

    model = HybridDetector(
        num_classes        = Config.NUM_CLASSES,
        image_size         = Config.IMAGE_SIZE,
        transformer_dim    = Config.TRANSFORMER_DIM,
        transformer_heads  = Config.TRANSFORMER_HEADS,
        transformer_layers = Config.TRANSFORMER_LAYERS,
    ).to(device)
    model.eval()

    criterion = AnchorFreeLoss(num_classes=Config.NUM_CLASSES)

    # Dummy batch: 1 gambar, 3 objek — satu per rentang FPN
    dummy_images  = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    dummy_targets = {
        'boxes': [
            torch.tensor([
                [Config.IMAGE_SIZE * 0.42, Config.IMAGE_SIZE * 0.42,  80.0,  80.0],  # P3 (kecil)
                [Config.IMAGE_SIZE * 0.50, Config.IMAGE_SIZE * 0.50, 100.0, 100.0],  # P4 (sedang)
                [Config.IMAGE_SIZE * 0.58, Config.IMAGE_SIZE * 0.25, 160.0, 160.0],  # P5 (besar)
            ])
        ],
        'labels': [torch.tensor([0, min(1, Config.NUM_CLASSES - 1),
                                  min(2, Config.NUM_CLASSES - 1)])],
    }

    with torch.no_grad():
        outputs = model(dummy_images)

    losses = criterion(outputs, dummy_targets)

    total  = losses['total_loss'].item()
    cls    = losses['class_loss'].item()
    bbox   = losses['bbox_loss'].item()
    obj    = losses['obj_loss'].item()

    print(f"  Losses:")
    print(f"    total_loss  : {total:.4f}")
    print(f"    class_loss  : {cls:.4f}")
    print(f"    bbox_loss   : {bbox:.4f}")
    print(f"    obj_loss    : {obj:.4f}")

    if bbox > 0 and obj > 0:
        print(f"  ✓ Loss function BENAR — ada anchor positif (bbox_loss & obj_loss > 0)")
    else:
        print(f"  ✗ MASALAH! bbox_loss atau obj_loss = 0 → tidak ada anchor positif.")
        print(f"    Cek ukuran objek dummy vs FPN limits di loss_fixed.py")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Konsistensi Format IoU di Metrics
# ─────────────────────────────────────────────────────────────────────────────
def check_iou_format():
    print("\n" + "=" * 60)
    print("CHECK 3: Konsistensi Format IoU di Metrics")
    print("=" * 60)

    # ── PERBAIKAN: Import dari nama file yang benar ──────────────────────────
    from utils.metrics_fixed import calculate_iou_batch   # ← WAS: utils.metrics

    # Prediksi dari model: format xyxy (absolute pixels)
    pred_xyxy = torch.tensor([[100.0, 100.0, 200.0, 200.0]])

    # GT dari dataset: format cxywh (center format)
    # Kotak yang sama: cx=150, cy=150, w=100, h=100
    gt_cxywh  = torch.tensor([[150.0, 150.0, 100.0, 100.0]])

    iou = calculate_iou_batch(pred_xyxy, gt_cxywh)
    print(f"  Pred (xyxy)  : {pred_xyxy[0].tolist()}")
    print(f"  GT   (cxywh) : {gt_cxywh[0].tolist()}")
    print(f"  IoU           : {iou[0, 0].item():.4f}  (harus = 1.0 untuk kotak yang sama)")

    if abs(iou[0, 0].item() - 1.0) < 0.01:
        print("  ✓ Fungsi IoU BENAR")
    else:
        print(f"  ✗ MASALAH! IoU = {iou[0,0].item():.4f}, seharusnya 1.0. "
              f"Cek metrics_fixed.py")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — DataLoader Batch Test
# ─────────────────────────────────────────────────────────────────────────────
def check_dataloader():
    print("\n" + "=" * 60)
    print("CHECK 4: DataLoader — Batch Pertama")
    print("=" * 60)

    transform = get_val_transforms(image_size=Config.IMAGE_SIZE)
    dataset   = ObjectDetectionDataset(
        Config.VAL_IMAGES, Config.VAL_ANNOTATIONS,
        transform=transform, image_size=Config.IMAGE_SIZE,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    images, targets = next(iter(loader))
    print(f"  images.shape   : {images.shape}")
    print(f"  images.dtype   : {images.dtype}")
    print(f"  Batch size     : {len(targets['boxes'])}")
    for i in range(len(targets['boxes'])):
        b = targets['boxes'][i]
        l = targets['labels'][i]
        print(f"  Sampel [{i}]  boxes={b.shape}  labels={l.tolist()}")
        if len(l) > 0:
            max_label = l.max().item()
            assert max_label < Config.NUM_CLASSES, (
                f"label maks={max_label} >= NUM_CLASSES={Config.NUM_CLASSES}! "
                f"Update config.py!"
            )

    print("  ✓ DataLoader OK — batch dapat dimuat dengan benar")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC SCRIPT — Jalankan sebelum training")
    print("=" * 60)

    try:
        check_num_classes()
    except Exception as e:
        print(f"  ✗ ERROR di check_num_classes: {e}")

    try:
        check_bbox_format()
    except Exception as e:
        print(f"  ✗ ERROR di check_bbox_format: {e}")

    try:
        check_num_positives()
    except Exception as e:
        print(f"  ✗ ERROR di check_num_positives: {e}")

    try:
        check_iou_format()
    except Exception as e:
        print(f"  ✗ ERROR di check_iou_format: {e}")

    try:
        check_dataloader()
    except Exception as e:
        print(f"  ✗ ERROR di check_dataloader: {e}")

    print("\n" + "=" * 60)
    print("  Diagnostic selesai.")
    print("  Semua ✓ = siap training. Ada ✗ = perbaiki dulu sebelum train.")
    print("=" * 60 + "\n")