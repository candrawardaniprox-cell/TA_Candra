"""
augment_dataset2025_splitdulu.py

Script augmentasi dataset offline untuk Dataset2025_splitdulu.

Pipeline:
1. SEMUA gambar (train, val, test) akan dikenakan enhancement:
   a. CLAHE  - untuk meningkatkan kontras secara adaptif dan merata.
   b. Median Filter - untuk mengurangi noise tanpa mengaburkan tepi objek.
2. Augmentasi HANYA menggunakan transformasi spasial (Geometri):
   - Horizontal Flip
   - Vertical Flip
   - Rotate (±15 derajat)
   - Zooming / Crop (Scale In)
   (Kombinasi 1-4 teknik ini diterapkan secara acak per gambar augmentasi)
3. Target penyeimbangan:
   - TRAIN : 3500 bbox per kelas (moler, slabung, ulat_grayak)
   - VAL   : 1000 bbox per kelas (moler, slabung, ulat_grayak)
   - TEST  : HANYA enhancement CLAHE + Median Filter (tanpa augmentasi)

Input:
  data/Dataset2025_splitdulu/
    ├── train2017/
    ├── val2017/
    ├── test2017/
    └── annotations_coco/
          ├── instances_train2017.json
          ├── instances_val2017.json
          └── instances_test2017.json

Output:
  data/Dataset2025_splitdulu_augmented/
    ├── train2017/
    ├── val2017/
    ├── test2017/
    └── annotations_coco/
          ├── instances_train2017.json
          ├── instances_val2017.json
          └── instances_test2017.json

Cara pakai:
  python augment_dataset2025_splitdulu.py --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ======================== KONSTANTA ========================
SEED = 42

# Category IDs yang digunakan untuk augmentasi (bukan parent class)
# ID 0 = "TA_Candra-1gDx" (parent / supercategory, DIABAIKAN)
# ID 1 = moler, ID 2 = slabung, ID 3 = ulat_grayak
AUGMENT_CAT_IDS = {1, 2, 3}

# ======================== TEKNIK ENHANCEMENT ========================


def apply_clahe(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Terapkan CLAHE ke gambar.
    """
    clip_limit = rng.uniform(2.0, 4.0)
    tile_size = rng.choice([4, 8, 8, 8, 16])  # Bias ke 8

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def apply_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Terapkan Median Filter untuk mengurangi noise pada gambar.

    Median Filter sangat efektif untuk menghilangkan salt-and-pepper noise
    sambil tetap mempertahankan tepi (edge) objek pada gambar.

    Parameter:
        image  : Gambar input (BGR).
        ksize  : Ukuran kernel median filter (harus bilangan ganjil: 3, 5, 7, ...).
                 Default = 3 (kernel 3x3).

    Return:
        Gambar hasil filtering.
    """
    result = cv2.medianBlur(image, ksize)
    return result


def apply_enhancement(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Terapkan pipeline enhancement lengkap ke gambar:
      1. CLAHE   (peningkatan kontras adaptif)
      2. Median Filter (pengurangan noise)

    Urutan ini dipilih agar CLAHE meningkatkan kontras terlebih dahulu,
    kemudian Median Filter membersihkan noise yang mungkin muncul atau
    teramplifikasi oleh proses CLAHE.
    """
    # Langkah 1: CLAHE
    result = apply_clahe(image, rng)

    # Langkah 2: Median Filter
    result = apply_median_filter(result, ksize=3)

    return result


# ======================== TEKNIK AUGMENTASI GEOMETRI ========================


def aug_spatial(
    image: np.ndarray,
    boxes_xywh: List[List[float]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Teknik Geometri: Flip (H/V), Rotasi, dan Zooming.
    Diterapkan kombinasi acak. Dijamin minimal 1 teknik selalu diterapkan.
    """
    h, w = image.shape[:2]
    result = image.copy()
    result_boxes = [list(b) for b in boxes_xywh]

    # Pilih teknik secara acak
    techniques = []
    if rng.random() < 0.5: techniques.append("hflip")
    if rng.random() < 0.3: techniques.append("vflip")
    if rng.random() < 0.4: techniques.append("rotate")
    if rng.random() < 0.4: techniques.append("zoom")

    # Jika kebetulan tidak ada yang terpilih, paksa pilih 1 secara acak
    if not techniques:
        techniques.append(rng.choice(["hflip", "vflip", "rotate", "zoom"]))

    # 1. Horizontal Flip
    if "hflip" in techniques:
        result = cv2.flip(result, 1)
        for i, (bx, by, bw, bh) in enumerate(result_boxes):
            result_boxes[i] = [w - bx - bw, by, bw, bh]

    # 2. Vertical Flip
    if "vflip" in techniques:
        result = cv2.flip(result, 0)
        for i, (bx, by, bw, bh) in enumerate(result_boxes):
            result_boxes[i] = [bx, h - by - bh, bw, bh]

    # 3. Rotasi
    if "rotate" in techniques:
        angle = rng.uniform(-15, 15)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        result = cv2.warpAffine(result, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT_101)

        new_boxes = []
        for bx, by, bw, bh in result_boxes:
            corners = np.array([
                [bx, by],
                [bx + bw, by],
                [bx + bw, by + bh],
                [bx, by + bh],
            ], dtype=np.float64)
            ones = np.ones((4, 1), dtype=np.float64)
            corners_h = np.hstack([corners, ones])
            transformed = (M @ corners_h.T).T
            x_min = max(0, transformed[:, 0].min())
            y_min = max(0, transformed[:, 1].min())
            x_max = min(new_w, transformed[:, 0].max())
            y_max = min(new_h, transformed[:, 1].max())
            new_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
        result_boxes = new_boxes
        h, w = result.shape[:2]

    # 4. Zooming / Crop
    valid_indices = list(range(len(result_boxes)))

    if "zoom" in techniques:
        scale = rng.uniform(0.75, 0.95)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        top = rng.randint(0, h - crop_h)
        left = rng.randint(0, w - crop_w)

        result = result[top:top + crop_h, left:left + crop_w]

        new_boxes = []
        new_valid_indices = []
        for idx_order, i in enumerate(valid_indices):
            bx, by, bw, bh = result_boxes[i]
            nx = max(0, bx - left)
            ny = max(0, by - top)
            nx2 = min(crop_w, bx + bw - left)
            ny2 = min(crop_h, by + bh - top)
            nw = nx2 - nx
            nh = ny2 - ny

            # Pastikan bbox masih ada di area crop
            if nw > 2 and nh > 2:
                scale_x = w / crop_w
                scale_y = h / crop_h
                new_boxes.append([nx * scale_x, ny * scale_y, nw * scale_x, nh * scale_y])
                new_valid_indices.append(i)

        result_boxes = new_boxes
        valid_indices = new_valid_indices
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result_boxes = [result_boxes[i] for i in valid_indices]

    return result, result_boxes, valid_indices


# ======================== UTILITAS ========================


def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def category_counter(annotations, cat_ids_filter: set = None) -> Counter:
    """Hitung jumlah bbox per category_id (hanya yang ada di filter)."""
    if cat_ids_filter:
        return Counter(
            int(ann["category_id"])
            for ann in annotations
            if int(ann["category_id"]) in cat_ids_filter
        )
    return Counter(int(ann["category_id"]) for ann in annotations)


# ======================== PROSES PER SPLIT ========================


def process_enhancement_only(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    image_subdir: str,
    annotation_file: str,
    rng: random.Random,
):
    """
    Proses split TEST: hanya terapkan CLAHE + Median Filter tanpa augmentasi.
    """
    print(f"\n{'=' * 60}")
    print(f"  [{split_name.upper()}] Enhancement Only (CLAHE + Median Filter)")
    print(f"{'=' * 60}")

    # Load annotation
    ann_path = input_dir / "annotations_coco" / annotation_file
    coco = load_coco(ann_path)

    # Buat folder output
    out_img_dir = output_dir / image_subdir
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir = output_dir / "annotations_coco"
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    # Proses semua gambar dengan CLAHE + Median Filter
    src_img_dir = input_dir / image_subdir
    processed = 0
    for img_info in coco["images"]:
        src = src_img_dir / img_info["file_name"]
        dst = out_img_dir / img_info["file_name"]
        if src.exists():
            img = cv2.imread(str(src))
            if img is not None:
                img_enhanced = apply_enhancement(img, rng)
                cv2.imwrite(str(dst), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                processed += 1

    print(f"  [OK] {processed} gambar diproses dengan CLAHE + Median Filter.")

    # Simpan annotation tanpa perubahan (gambar sama, hanya kualitas meningkat)
    out_ann_path = out_ann_dir / annotation_file
    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"  [OK] Annotation disimpan: {out_ann_path}")


def process_augmentation(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    image_subdir: str,
    annotation_file: str,
    targets: Dict[int, int],
    rng: random.Random,
):
    """
    Proses split TRAIN / VAL:
      1. Terapkan CLAHE + Median Filter ke semua gambar asli
      2. Augmentasi Geometri hingga mencapai target bbox per kelas
    """
    print(f"\n{'=' * 60}")
    print(f"  [{split_name.upper()}] CLAHE + Median + Augmentasi Geometri")
    print(f"{'=' * 60}")

    # Load annotation
    ann_path = input_dir / "annotations_coco" / annotation_file
    coco = load_coco(ann_path)
    cats = {int(c["id"]): c["name"] for c in coco["categories"]}

    # Bangun index anotasi per image
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[int(ann["image_id"])].append(ann)

    original_counts = category_counter(coco["annotations"], AUGMENT_CAT_IDS)

    # Buat folder output
    out_img_dir = output_dir / image_subdir
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir = output_dir / "annotations_coco"
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    src_img_dir = input_dir / image_subdir

    # ---- Langkah 1: Terapkan CLAHE + Median Filter ke semua data asli ----
    print(f"\n  [1/3] Menerapkan CLAHE + Median Filter ke {split_name} asli...")
    processed = 0
    for img_info in coco["images"]:
        src = src_img_dir / img_info["file_name"]
        dst = out_img_dir / img_info["file_name"]
        if src.exists():
            img = cv2.imread(str(src))
            if img is not None:
                img_enhanced = apply_enhancement(img, rng)
                cv2.imwrite(str(dst), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                processed += 1
    print(f"  [OK] {processed} gambar asli telah diproses CLAHE + Median Filter.")

    # ---- Langkah 2: Augmentasi Geometri ----
    print(f"\n  [2/3] Menerapkan Augmentasi Geometri ({split_name})...")
    print(f"  Target per kelas:")
    for cat_id in sorted(targets):
        cat_name = cats.get(cat_id, f"ID_{cat_id}")
        orig = original_counts.get(cat_id, 0)
        target = targets[cat_id]
        print(f"    - {cat_name:<15} : {orig:>5} -> {target:>5} bbox")

    # Bangun daftar gambar per kategori
    cat_images = defaultdict(list)
    for img_info in coco["images"]:
        img_id = int(img_info["id"])
        seen_cats = set()
        for ann in ann_by_img.get(img_id, []):
            cat_id = int(ann["category_id"])
            if cat_id in AUGMENT_CAT_IDS and cat_id not in seen_cats:
                cat_images[cat_id].append(img_info)
                seen_cats.add(cat_id)

    # Siapkan COCO output baru
    new_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": copy.deepcopy(coco["categories"]),
        "images": copy.deepcopy(coco["images"]),
        "annotations": copy.deepcopy(coco["annotations"]),
    }

    next_img_id = max(int(img["id"]) for img in coco["images"]) + 1
    next_ann_id = (
        max(int(ann["id"]) for ann in coco["annotations"]) + 1
        if coco["annotations"]
        else 0
    )
    current_counts = Counter(original_counts)

    total_augmented = 0
    attempts = 0
    max_attempts = 500000

    while attempts < max_attempts:
        remaining = {
            cid: target - current_counts.get(cid, 0)
            for cid, target in targets.items()
        }
        needed_cats = [cid for cid, need in remaining.items() if need > 0]
        if not needed_cats:
            break

        attempts += 1
        weights = [remaining[cid] for cid in needed_cats]
        selected_cat = rng.choices(needed_cats, weights=weights, k=1)[0]

        candidates = cat_images.get(selected_cat, [])
        if not candidates:
            continue

        img_info = rng.choice(candidates)
        img_id = int(img_info["id"])
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue

        # Baca citra yang SUDAH di-enhance dari output_dir
        src_path = out_img_dir / img_info["file_name"]
        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        boxes = []
        cat_ids_list = []
        for ann in anns:
            boxes.append(ann["bbox"])
            cat_ids_list.append(int(ann["category_id"]))

        # Lakukan transformasi Geometri
        aug_img, aug_boxes, valid_indices = aug_spatial(img, boxes, rng)
        if not aug_boxes:
            continue

        # Filter categories sesuai valid_indices
        aug_cats = [cat_ids_list[i] for i in valid_indices]

        aug_cat_counter = Counter(aug_cats)
        if aug_cat_counter.get(selected_cat, 0) <= 0:
            continue

        # Pengecekan overshoot target
        overshoots = False
        for cat_id, count in aug_cat_counter.items():
            if cat_id in targets:
                if current_counts.get(cat_id, 0) + count > targets[cat_id] + 50:
                    overshoots = True
                    break
        if overshoots:
            continue

        # Simpan gambar augmentasi
        original_stem = Path(img_info["file_name"]).stem
        aug_file_name = f"{original_stem}_geo{total_augmented:05d}.jpg"
        aug_path = out_img_dir / aug_file_name
        cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Update COCO record
        aug_h, aug_w = aug_img.shape[:2]
        new_coco["images"].append({
            "id": next_img_id,
            "file_name": aug_file_name,
            "width": aug_w,
            "height": aug_h,
        })

        for bbox, cat_id in zip(aug_boxes, aug_cats):
            bx, by, bw, bh = bbox
            new_coco["annotations"].append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": cat_id,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            next_ann_id += 1

        current_counts.update(aug_cat_counter)
        next_img_id += 1
        total_augmented += 1

        if total_augmented % 500 == 0:
            rem_str = ", ".join(
                f"{cats.get(c, c)}:{targets[c] - current_counts.get(c, 0)}"
                for c in sorted(targets)
            )
            print(f"    [{total_augmented:>5} augmented] remaining bbox: {rem_str}")

    # ---- Langkah 3: Simpan JSON COCO ----
    print(f"\n  [3/3] Menyimpan annotation {split_name}...")
    out_ann_path = out_ann_dir / annotation_file
    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False)

    print(f"\n  --- HASIL AKHIR {split_name.upper()} ---")
    for cat_id in sorted(targets):
        cat_name = cats.get(cat_id, f"ID_{cat_id}")
        orig = original_counts.get(cat_id, 0)
        final = current_counts.get(cat_id, 0)
        print(f"    {cat_name:<15} : {orig:>5} -> {final:>5} bbox")
    print(f"    Total gambar akhir : {len(new_coco['images'])}")
    print(f"    Total augmentasi   : {total_augmented}")


# ======================== MAIN ========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmentasi Dataset2025_splitdulu (CLAHE + Median + Geometri)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/Dataset2025_splitdulu"),
        help="Folder dataset input (default: data/Dataset2025_splitdulu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Dataset2025_splitdulu_augmented"),
        help="Folder dataset output (default: data/Dataset2025_splitdulu_augmented)",
    )
    parser.add_argument(
        "--target-train", type=int, default=3500,
        help="Target bbox per kelas untuk TRAIN (default: 3500)",
    )
    parser.add_argument(
        "--target-val", type=int, default=1000,
        help="Target bbox per kelas untuk VAL (default: 1000)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Hapus folder output jika sudah ada",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  AUGMENTASI Dataset2025_splitdulu")
    print("  CLAHE + MEDIAN FILTER + GEOMETRI")
    print("=" * 60)
    print(f"  Input  : {args.input_dir}")
    print(f"  Output : {args.output_dir}")
    print(f"  Target TRAIN : {args.target_train} bbox per kelas")
    print(f"  Target VAL   : {args.target_val} bbox per kelas")
    print(f"  TEST         : Enhancement saja (tanpa augmentasi)")
    print("=" * 60)

    # Cek input dir
    if not args.input_dir.exists():
        print(f"[X] Folder input tidak ditemukan: {args.input_dir}")
        sys.exit(1)

    # Handle output dir
    if args.output_dir.exists():
        if not args.overwrite:
            print(f"[X] Folder output {args.output_dir} sudah ada. Gunakan --overwrite.")
            sys.exit(1)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Target bbox per kelas (cat_id 1=moler, 2=slabung, 3=ulat_grayak)
    train_targets = {
        1: args.target_train,  # moler
        2: args.target_train,  # slabung
        3: args.target_train,  # ulat_grayak
    }
    val_targets = {
        1: args.target_val,  # moler
        2: args.target_val,  # slabung
        3: args.target_val,  # ulat_grayak
    }

    # ====== PROSES TRAIN ======
    process_augmentation(
        split_name="Train",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_subdir="train2017",
        annotation_file="instances_train2017.json",
        targets=train_targets,
        rng=rng,
    )

    # ====== PROSES VAL ======
    process_augmentation(
        split_name="Val",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_subdir="val2017",
        annotation_file="instances_val2017.json",
        targets=val_targets,
        rng=rng,
    )

    # ====== PROSES TEST (Enhancement Only) ======
    process_enhancement_only(
        split_name="Test",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_subdir="test2017",
        annotation_file="instances_test2017.json",
        rng=rng,
    )

    # ====== RINGKASAN AKHIR ======
    print(f"\n{'=' * 60}")
    print("  SELESAI! RINGKASAN AUGMENTASI")
    print(f"{'=' * 60}")
    print(f"  Folder output : {args.output_dir}")
    print(f"  TRAIN : Augmentasi -> {args.target_train} bbox/kelas")
    print(f"  VAL   : Augmentasi -> {args.target_val} bbox/kelas")
    print(f"  TEST  : Enhancement CLAHE + Median Filter saja")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
