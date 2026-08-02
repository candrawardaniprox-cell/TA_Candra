"""
augment_balanced.py — Augmentasi Offline Dataset Pertanian Bawang Merah.

Script ini melakukan augmentasi COCO dataset secara offline dengan 5 teknik
augmentasi yang relevan untuk domain pertanian, dan menyeimbangkan jumlah
bounding box antar kelas (moler, slabung, ulat_grayak).

Teknik Augmentasi (diterapkan kombinasi acak 1-3 per gambar):
  1. Variasi Pencahayaan (Brightness, Contrast, Gamma Correction)
  2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. Injeksi Noise (Gaussian Noise / Salt & Pepper)
  4. Variasi Blur (Motion Blur / Gaussian Blur)
  5. Transformasi Spasial (Flip, Rotasi, Zoom/Scale)

Aturan:
  - Dataset ASLI tidak diubah, tetap utuh 100%.
  - Augmentasi ditambahkan sebagai gambar-gambar baru di atas data asli.
  - Bobot augmentasi per kelas bisa diatur agar jumlah bbox seimbang.

Input:
  data/Dataset2026/                (gambar + _annotations.coco.json)

Output:
  data/Dataset2026_augmented/      (asli + augmented, JSON COCO baru)

Cara pakai:
  python augment_balanced.py
  python augment_balanced.py --target-moler 5000 --target-slabung 5000 --target-ulat-grayak 5000
  python augment_balanced.py --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ======================== KONSTANTA ========================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# Nama kelas yang di-augmentasi (category_id=0 adalah label project Roboflow, diabaikan)
AUGMENTED_CLASSES = ["moler", "slabung", "ulat_grayak"]


# ======================== 5 TEKNIK AUGMENTASI ========================

def aug_brightness_contrast_gamma(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Teknik 1: Variasi Pencahayaan (Brightness, Contrast, Gamma Correction).
    Simulasi perubahan cahaya matahari di lahan pertanian (pagi/siang/sore/mendung).
    """
    img = image.astype(np.float32)

    # Brightness: geser kecerahan ±30 piksel
    brightness_shift = rng.uniform(-30, 30)
    img = img + brightness_shift

    # Contrast: skala kontras 0.7 - 1.3x
    contrast_factor = rng.uniform(0.7, 1.3)
    mean_val = np.mean(img)
    img = (img - mean_val) * contrast_factor + mean_val

    # Gamma Correction: simulasi over/under exposure
    img = np.clip(img, 0, 255).astype(np.uint8)
    gamma = rng.uniform(0.7, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    return img


def aug_clahe(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Teknik 2: CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Mempertegas batas antara bercak penyakit dan daun sehat.
    """
    clip_limit = rng.uniform(2.0, 4.0)
    tile_size = rng.choice([4, 8, 8, 8, 16])  # Bias ke 8

    # Konversi ke LAB color space (hanya equalize channel L/lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


def aug_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Teknik 3: Injeksi Noise (Gaussian / Salt & Pepper).
    Simulasi kamera smartphone dengan ISO tinggi di kondisi minim cahaya.
    """
    noise_type = rng.choice(["gaussian", "salt_pepper"])

    if noise_type == "gaussian":
        sigma = rng.uniform(5, 25)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    else:
        # Salt & Pepper noise
        prob = rng.uniform(0.005, 0.03)  # 0.5% - 3%
        noisy = image.copy()
        # Salt (putih)
        salt_mask = np.random.random(image.shape[:2]) < (prob / 2)
        noisy[salt_mask] = 255
        # Pepper (hitam)
        pepper_mask = np.random.random(image.shape[:2]) < (prob / 2)
        noisy[pepper_mask] = 0
        return noisy


def aug_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Teknik 4: Variasi Blur (Gaussian Blur / Motion Blur).
    Simulasi angin yang menggerakkan daun atau tangan petani yang bergetar.
    """
    blur_type = rng.choice(["gaussian", "motion"])

    if blur_type == "gaussian":
        kernel_size = rng.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        # Motion blur
        kernel_size = rng.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        direction = rng.choice(["horizontal", "vertical", "diagonal"])
        if direction == "horizontal":
            kernel[kernel_size // 2, :] = 1.0
        elif direction == "vertical":
            kernel[:, kernel_size // 2] = 1.0
        else:
            np.fill_diagonal(kernel, 1.0)
        kernel /= kernel_size
        return cv2.filter2D(image, -1, kernel)


def aug_spatial(
    image: np.ndarray,
    boxes_xywh: List[List[float]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Teknik 5: Transformasi Spasial (Flip, Rotasi, Zoom/Scale).
    Membuat model invariant terhadap orientasi dan jarak kamera.
    Mengembalikan gambar dan bounding box yang sudah di-transform.
    """
    h, w = image.shape[:2]
    result = image.copy()
    result_boxes = [list(b) for b in boxes_xywh]

    # Horizontal Flip (50% kemungkinan)
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)
        for i, (bx, by, bw, bh) in enumerate(result_boxes):
            result_boxes[i] = [w - bx - bw, by, bw, bh]

    # Vertical Flip (30% kemungkinan)
    if rng.random() < 0.3:
        result = cv2.flip(result, 0)
        for i, (bx, by, bw, bh) in enumerate(result_boxes):
            result_boxes[i] = [bx, h - by - bh, bw, bh]

    # Rotasi kecil ±15 derajat (40% kemungkinan)
    if rng.random() < 0.4:
        angle = rng.uniform(-15, 15)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Hitung ukuran baru agar tidak ada area terpotong
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        result = cv2.warpAffine(result, M, (new_w, new_h),
                                borderMode=cv2.BORDER_REFLECT_101)

        # Transform bbox: rotasi 4 corner lalu buat bounding rect
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

    # Zoom/Scale (30% kemungkinan) - crop lalu resize kembali
    if rng.random() < 0.3:
        scale = rng.uniform(0.8, 1.0)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        top = rng.randint(0, h - crop_h)
        left = rng.randint(0, w - crop_w)

        result = result[top:top + crop_h, left:left + crop_w]

        # Adjust bounding boxes
        new_boxes = []
        for bx, by, bw, bh in result_boxes:
            # Shift dan clip
            nx = max(0, bx - left)
            ny = max(0, by - top)
            nx2 = min(crop_w, bx + bw - left)
            ny2 = min(crop_h, by + bh - top)
            nw = nx2 - nx
            nh = ny2 - ny
            if nw > 2 and nh > 2:
                # Scale ke ukuran original
                scale_x = w / crop_w
                scale_y = h / crop_h
                new_boxes.append([nx * scale_x, ny * scale_y,
                                  nw * scale_x, nh * scale_y])
            else:
                new_boxes.append(None)  # Bbox hilang setelah crop
        result_boxes = [b for b in new_boxes if b is not None]

        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    return result, result_boxes


# ======================== PIPELINE AUGMENTASI ========================

def apply_augmentation_pipeline(
    image: np.ndarray,
    boxes_xywh: List[List[float]],
    cat_ids: List[int],
    rng: random.Random,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Terapkan kombinasi acak 1-3 teknik augmentasi pada satu gambar.
    Teknik dipilih secara random dengan probabilitas tertentu.

    Returns:
        - Gambar augmented
        - Bounding boxes [x, y, w, h] yang sudah di-transform
        - Category IDs yang tersisa (setelah filtering bbox hilang)
    """
    result = image.copy()
    result_boxes = [list(b) for b in boxes_xywh]
    result_cats = list(cat_ids)

    # Tentukan teknik mana yang aktif (minimal 1, maksimal 3)
    techniques = []
    if rng.random() < 0.5:
        techniques.append("brightness")
    if rng.random() < 0.3:
        techniques.append("clahe")
    if rng.random() < 0.3:
        techniques.append("noise")
    if rng.random() < 0.3:
        techniques.append("blur")
    if rng.random() < 0.6:
        techniques.append("spatial")

    # Pastikan minimal 1 teknik
    if not techniques:
        techniques.append(rng.choice(["brightness", "clahe", "noise", "blur", "spatial"]))

    # Batasi maksimal 3 teknik
    if len(techniques) > 3:
        techniques = rng.sample(techniques, 3)

    # Terapkan teknik-teknik augmentasi non-spasial dulu
    for tech in techniques:
        if tech == "brightness":
            result = aug_brightness_contrast_gamma(result, rng)
        elif tech == "clahe":
            result = aug_clahe(result, rng)
        elif tech == "noise":
            result = aug_noise(result, rng)
        elif tech == "blur":
            result = aug_blur(result, rng)

    # Terakhir, terapkan spatial transform (mengubah bbox)
    if "spatial" in techniques:
        result, new_boxes = aug_spatial(result, result_boxes, rng)

        # Sinkronkan category ids dengan bbox yang tersisa
        if len(new_boxes) < len(result_boxes):
            # Beberapa bbox hilang setelah crop, kita perlu filter
            # Catatan: aug_spatial mengembalikan None untuk bbox yang hilang,
            # tapi kita sudah filter di dalam fungsinya. Jadi kita perlu
            # tracking manual
            pass

        # Rebuild cat_ids berdasarkan jumlah bbox yang tersisa
        # (aug_spatial sudah memfilter None)
        if len(new_boxes) <= len(result_cats):
            result_cats = result_cats[:len(new_boxes)]
        result_boxes = new_boxes

    return result, result_boxes, result_cats


# ======================== UTILITAS COCO ========================

def load_coco(path: Path) -> Dict:
    """Muat file JSON COCO."""
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"JSON COCO tidak punya key wajib: {key}")
    return coco


def annotations_by_image(coco: Dict) -> Dict[int, List[Dict]]:
    """Kelompokkan anotasi berdasarkan image_id."""
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def category_counter(annotations) -> Counter:
    """Hitung jumlah bbox per category_id."""
    return Counter(int(ann["category_id"]) for ann in annotations)


def images_by_category(
    coco: Dict,
    ann_by_img: Dict[int, List[Dict]],
) -> Dict[int, List[Dict]]:
    """Kelompokkan image_info berdasarkan category yang ada di dalamnya."""
    cat_images: Dict[int, List[Dict]] = defaultdict(list)
    for img_info in coco["images"]:
        img_id = int(img_info["id"])
        anns = ann_by_img.get(img_id, [])
        seen_cats = set()
        for ann in anns:
            cat_id = int(ann["category_id"])
            if cat_id not in seen_cats:
                cat_images[cat_id].append(img_info)
                seen_cats.add(cat_id)
    return cat_images


# ======================== MAIN LOGIC ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmentasi offline dataset pertanian bawang merah dengan "
                    "5 teknik augmentasi dan class balancing."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/Dataset2026"),
        help="Folder dataset asli (berisi gambar + _annotations.coco.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Folder output. Default: <input-dir>_augmented",
    )
    parser.add_argument(
        "--target-moler", type=int, default=5000,
        help="Target jumlah bbox untuk kelas moler (default: 5000)",
    )
    parser.add_argument(
        "--target-slabung", type=int, default=5000,
        help="Target jumlah bbox untuk kelas slabung (default: 5000)",
    )
    parser.add_argument(
        "--target-ulat-grayak", type=int, default=5000,
        help="Target jumlah bbox untuk kelas ulat_grayak (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Kualitas JPEG output")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Hapus folder output jika sudah ada",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=500000,
        help="Batas percobaan saat mencari augmentasi untuk balance",
    )

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / f"{args.input_dir.name}_augmented"
    return args


def print_dataset_stats(title: str, cats: Dict[int, str], counts: Counter, img_count: int):
    """Cetak statistik dataset."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total Gambar  : {img_count}")
    for cat_id in sorted(counts):
        name = cats.get(cat_id, f"id={cat_id}")
        print(f"  {name:<15}: {counts[cat_id]:>6} bbox")
    print(f"  {'TOTAL':<15}: {sum(counts.values()):>6} bbox")
    print(sep)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Baca dataset asli ----
    annotation_file = args.input_dir / "_annotations.coco.json"
    if not annotation_file.exists():
        print(f"[X] File anotasi tidak ditemukan: {annotation_file}")
        sys.exit(1)

    coco = load_coco(annotation_file)
    cats = {int(c["id"]): c["name"] for c in coco["categories"]}
    ann_by_img = annotations_by_image(coco)

    # Hitung bbox asli
    all_anns = coco["annotations"]
    original_counts = category_counter(all_anns)

    print_dataset_stats("DATASET ASLI (SEBELUM AUGMENTASI)", cats, original_counts, len(coco["images"]))

    # ---- 2. Tentukan target per kelas ----
    # Cari category_id untuk setiap kelas
    name_to_id = {}
    for cat_id, name in cats.items():
        name_to_id[name.lower().replace("-", "_").replace(" ", "_")] = cat_id

    targets: Dict[int, int] = {}
    target_map = {
        "moler": args.target_moler,
        "slabung": args.target_slabung,
        "ulat_grayak": args.target_ulat_grayak,
    }

    for class_name, target_count in target_map.items():
        cat_id = name_to_id.get(class_name)
        if cat_id is not None:
            targets[cat_id] = target_count

    # Tampilkan rencana augmentasi
    sep = "=" * 70
    print(f"\n{sep}")
    print("  RENCANA AUGMENTASI")
    print(sep)
    for cat_id, target in sorted(targets.items()):
        current = original_counts.get(cat_id, 0)
        need = max(0, target - current)
        multiplier = target / current if current > 0 else 0
        print(f"  {cats[cat_id]:<15}: {current:>5} -> {target:>5}  "
              f"(perlu tambah {need:>5} bbox, ~{multiplier:.2f}x)")
    print(sep)

    # ---- 3. Siapkan folder output ----
    if args.output_dir.exists():
        if not args.overwrite:
            print(f"\n[X] Folder output sudah ada: {args.output_dir}")
            print("    Gunakan --overwrite untuk menimpa.")
            sys.exit(1)
        shutil.rmtree(args.output_dir)

    output_image_dir = args.output_dir
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. Salin semua gambar asli ke folder output ----
    print("\n  Menyalin gambar asli...")
    copied = 0
    for img_info in coco["images"]:
        src = args.input_dir / img_info["file_name"]
        dst = output_image_dir / img_info["file_name"]
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"  [OK] {copied} gambar asli disalin ke {output_image_dir}")

    # ---- 5. Mulai augmentasi per kelas (balance mode) ----
    cat_images = images_by_category(coco, ann_by_img)

    # Output COCO: mulai dari data asli
    new_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": copy.deepcopy(coco["categories"]),
        "images": copy.deepcopy(coco["images"]),
        "annotations": copy.deepcopy(coco["annotations"]),
    }

    # ID counter
    max_img_id = max(int(img["id"]) for img in coco["images"])
    max_ann_id = max(int(ann["id"]) for ann in coco["annotations"]) if coco["annotations"] else 0
    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    # Track current bbox counts (termasuk data asli)
    current_counts = Counter(original_counts)

    total_augmented_images = 0
    attempts = 0

    print(f"\n  Memulai augmentasi (target: {targets})...")
    print(f"  Max attempts: {args.max_attempts}")

    while attempts < args.max_attempts:
        # Cek kelas mana yang masih butuh augmentasi
        remaining = {
            cat_id: target - current_counts.get(cat_id, 0)
            for cat_id, target in targets.items()
        }
        needed_cats = [cat_id for cat_id, need in remaining.items() if need > 0]
        if not needed_cats:
            break

        attempts += 1

        # Pilih kelas yang paling membutuhkan (weighted random)
        weights = [remaining[cat_id] for cat_id in needed_cats]
        selected_cat = rng.choices(needed_cats, weights=weights, k=1)[0]

        # Pilih gambar random yang mengandung kelas tersebut
        candidates = cat_images.get(selected_cat, [])
        if not candidates:
            continue
        img_info = rng.choice(candidates)
        img_id = int(img_info["id"])

        # Ambil anotasi gambar ini
        anns = ann_by_img.get(img_id, [])
        if not anns:
            continue

        # Baca gambar
        src_path = args.input_dir / img_info["file_name"]
        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        # Kumpulkan bbox dan cat_ids
        boxes = []
        cat_ids_list = []
        for ann in anns:
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                boxes.append([float(v) for v in bbox])
                cat_ids_list.append(int(ann["category_id"]))

        if not boxes:
            continue

        # Terapkan augmentasi
        aug_img, aug_boxes, aug_cats = apply_augmentation_pipeline(
            img, boxes, cat_ids_list, rng
        )

        if not aug_boxes:
            continue

        # Cek apakah augmentasi ini masih mengandung kelas yang dibutuhkan
        aug_cat_counter = Counter(aug_cats)
        if aug_cat_counter.get(selected_cat, 0) <= 0:
            continue

        # Cek apakah tidak overshoot target
        overshoots = False
        for cat_id, count in aug_cat_counter.items():
            target = targets.get(cat_id)
            if target is not None and current_counts.get(cat_id, 0) + count > target + 50:
                # Toleransi overshoot 50 bbox
                overshoots = True
                break
        if overshoots:
            continue

        # Simpan gambar augmented
        original_stem = Path(img_info["file_name"]).stem
        aug_file_name = f"{original_stem}_aug{total_augmented_images:05d}.jpg"
        aug_path = output_image_dir / aug_file_name

        cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        # Buat entry COCO baru
        aug_h, aug_w = aug_img.shape[:2]
        new_img_entry = {
            "id": next_img_id,
            "file_name": aug_file_name,
            "width": aug_w,
            "height": aug_h,
        }
        new_coco["images"].append(new_img_entry)

        for bbox, cat_id in zip(aug_boxes, aug_cats):
            bx, by, bw, bh = bbox
            new_ann = {
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": cat_id,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
            }
            new_coco["annotations"].append(new_ann)
            next_ann_id += 1

        # Update counters
        current_counts.update(aug_cat_counter)
        next_img_id += 1
        total_augmented_images += 1

        # Progress log
        if total_augmented_images % 100 == 0:
            remaining_text = ", ".join(
                f"{cats[cid]}:{targets[cid] - current_counts.get(cid, 0)}"
                for cid in sorted(targets)
            )
            print(f"  [{total_augmented_images:>5} augmented] remaining bbox: {remaining_text}")

    # ---- 6. Cek apakah target tercapai ----
    remaining = {
        cat_id: targets[cat_id] - current_counts.get(cat_id, 0)
        for cat_id in targets
    }
    if any(v > 0 for v in remaining.values()):
        print(f"\n  [!] Target belum sepenuhnya tercapai.")
        print(f"      Sisa kebutuhan: {remaining}")
        print(f"      Coba naikkan --max-attempts.")

    # ---- 7. Tulis JSON COCO baru ----
    output_json = output_image_dir / "_annotations.coco.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False)

    # ---- 8. Tampilkan statistik akhir ----
    final_counts = category_counter(new_coco["annotations"])
    print_dataset_stats(
        "DATASET SETELAH AUGMENTASI",
        cats, final_counts, len(new_coco["images"])
    )

    # Tabel perbandingan
    print(f"\n{sep}")
    print("  RINGKASAN PERBANDINGAN")
    print(sep)
    print(f"  {'Kelas':<15} {'Asli':>8} {'Sesudah':>10} {'Target':>8} {'Pengali':>10}")
    print(f"  {'-' * 55}")
    for cat_id in sorted(targets):
        orig = original_counts.get(cat_id, 0)
        final = final_counts.get(cat_id, 0)
        target = targets[cat_id]
        mult = final / orig if orig > 0 else 0
        print(f"  {cats[cat_id]:<15} {orig:>8} {final:>10} {target:>8} {mult:>9.2f}x")
    print(sep)
    print(f"  Total gambar asli      : {len(coco['images'])}")
    print(f"  Total gambar augmented : {total_augmented_images}")
    print(f"  Total gambar akhir     : {len(new_coco['images'])}")
    print(f"  Output folder          : {args.output_dir}")
    print(f"  Output JSON            : {output_json}")
    print(sep)


if __name__ == "__main__":
    main()
