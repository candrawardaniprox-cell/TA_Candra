"""
split_augment_balanced.py

Pipeline lengkap: SPLIT → CLAHE + Median Filter → AUGMENTASI GEOMETRI (BALANCING)

Script ini melakukan SEMUA proses dalam satu file:
1. Membaca dataset mentah (citra + _annotations.coco.json) dari satu folder.
2. SPLIT data berdasarkan jumlah BBOX per kelas (BUKAN per citra):
   - Train : 2100 bbox per kelas (sebelum augmentasi)
   - Val   :  600 bbox per kelas (sebelum augmentasi)
   - Test  :  300 bbox per kelas (tanpa augmentasi)
3. Menerapkan CLAHE + Median Filter ke SEMUA gambar (train/val/test).
4. Augmentasi Geometri HANYA untuk train & val:
   - Train : 2100 → 3500 bbox per kelas
   - Val   :  600 → 1000 bbox per kelas
   - Test  :  300 bbox per kelas (TIDAK diaugmentasi)
5. Output berformat COCO:
   output_dir/
   ├── annotations_coco/
   │   ├── instances_train2017.json
   │   ├── instances_val2017.json
   │   └── instances_test2017.json
   ├── train2017/   [3500 bbox per kelas]
   ├── val2017/     [1000 bbox per kelas]
   └── test2017/    [300 bbox per kelas]

ATURAN PENTING:
- Split dilakukan per GAMBAR, tapi dihitung per BBOX.
- Gambar yang masuk test TIDAK BOLEH ada di train maupun val (no data leakage).
- Augmentasi geometri: Horizontal Flip, Vertical Flip, Rotasi (±15°), Zooming/Crop.
- CLAHE & Median Filter diterapkan ke SEMUA gambar (asli maupun augmentasi).

Cara pakai:
  python split_augment_balanced.py
  python split_augment_balanced.py --input data/Dataset2026 --output data/Dataset2026_split_aug_balanced
  python split_augment_balanced.py --overwrite
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
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

# ======================== KONSTANTA ========================
SEED = 42

# Target bbox PER KELAS sebelum augmentasi (split awal)
SPLIT_TARGET_TRAIN = 2100  # bbox per kelas di train
SPLIT_TARGET_VAL = 600     # bbox per kelas di val
SPLIT_TARGET_TEST = 300    # bbox per kelas di test

# Target bbox PER KELAS setelah augmentasi
AUG_TARGET_TRAIN = 3500    # bbox per kelas di train setelah augmentasi
AUG_TARGET_VAL = 1000      # bbox per kelas di val setelah augmentasi
# Test TIDAK diaugmentasi, tetap 300 bbox per kelas


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
    """
    result = cv2.medianBlur(image, ksize)
    return result


def apply_enhancement(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Pipeline enhancement: CLAHE → Median Filter.
    """
    result = apply_clahe(image, rng)
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
    """Muat file JSON COCO."""
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"JSON COCO tidak punya key wajib: {key}")
    return coco


def category_counter(annotations) -> Counter:
    return Counter(int(ann["category_id"]) for ann in annotations)


def print_separator(char="=", length=70):
    print(char * length)


# ======================== SPLIT BERDASARKAN BBOX ========================

def bbox_balanced_split(
    images: List[Dict],
    ann_by_img: Dict[int, List[Dict]],
    all_cat_ids: List[int],
    target_train: int,
    target_val: int,
    target_test: int,
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split gambar ke train/val/test sehingga jumlah BBOX per KELAS mendekati target.

    Algoritma:
    1. Hitung total bbox per kelas → tentukan rasio target.
    2. Prioritaskan pengisian test dulu (paling kecil), lalu val, lalu train.
    3. Gunakan greedy assignment: untuk setiap gambar (diacak),
       assign ke split yang paling membutuhkan bbox dari kelas yang ada di gambar tersebut.
    4. Gambar yang sudah di-assign ke test DIJAMIN tidak ada di train/val.
    """

    # Hitung bbox per kelas per gambar
    img_bbox_counts: Dict[int, Counter] = {}
    for img_info in images:
        img_id = int(img_info["id"])
        cat_counter = Counter()
        for ann in ann_by_img.get(img_id, []):
            cat_counter[int(ann["category_id"])] += 1
        img_bbox_counts[img_id] = cat_counter

    # Target per split per kelas
    targets = {
        "test": {cid: target_test for cid in all_cat_ids},
        "val": {cid: target_val for cid in all_cat_ids},
        "train": {cid: target_train for cid in all_cat_ids},
    }

    # Current counts
    current = {
        "test": Counter(),
        "val": Counter(),
        "train": Counter(),
    }

    # Assignments
    assignments: Dict[str, List[Dict]] = {
        "test": [],
        "val": [],
        "train": [],
    }

    assigned_img_ids: Set[int] = set()

    # Acak urutan gambar
    shuffled_images = list(images)
    rng.shuffle(shuffled_images)

    # Prioritas: test → val → train (isi yang terkecil dulu)
    split_priority = ["test", "val", "train"]

    for img_info in shuffled_images:
        img_id = int(img_info["id"])
        if img_id in assigned_img_ids:
            continue

        img_cats = img_bbox_counts.get(img_id, Counter())
        if not img_cats:
            continue

        # Hitung skor defisit untuk setiap split
        best_split = "train"
        best_score = -float("inf")

        for split_name in split_priority:
            score = 0.0
            relevant = 0
            for cid in all_cat_ids:
                target = targets[split_name].get(cid, 0)
                if target <= 0:
                    continue
                cur = current[split_name].get(cid, 0)
                deficit_ratio = (target - cur) / target  # 1.0 = masih kosong, 0.0 = sudah penuh

                img_has = img_cats.get(cid, 0)
                if img_has > 0:
                    # Beri bobot lebih tinggi jika split ini masih butuh kelas ini
                    if cur < target:
                        score += deficit_ratio * 3.0  # Bonus besar jika belum tercapai
                    else:
                        score -= 1.0  # Penalti jika sudah melebihi target
                    relevant += 1
                else:
                    score += deficit_ratio * 0.5
                    relevant += 1

            if relevant > 0:
                score /= relevant

            if score > best_score:
                best_score = score
                best_split = split_name

        # Cek apakah split terbaik sudah overflow terlalu banyak
        # Jika semua kelas di split target sudah melebihi, coba split lain
        all_overflow = True
        for cid in all_cat_ids:
            if img_cats.get(cid, 0) > 0:
                if current[best_split].get(cid, 0) < targets[best_split].get(cid, 0):
                    all_overflow = False
                    break

        if all_overflow:
            # Cari split lain yang masih butuh
            for alt_split in split_priority:
                if alt_split == best_split:
                    continue
                for cid in all_cat_ids:
                    if img_cats.get(cid, 0) > 0:
                        if current[alt_split].get(cid, 0) < targets[alt_split].get(cid, 0):
                            best_split = alt_split
                            all_overflow = False
                            break
                if not all_overflow:
                    break

        # Assign gambar ke split terbaik
        assignments[best_split].append(img_info)
        assigned_img_ids.add(img_id)
        current[best_split].update(img_cats)

    return assignments["train"], assignments["val"], assignments["test"]


# ======================== AUGMENTASI + BALANCING ========================

def augment_split(
    split_name: str,
    images: List[Dict],
    ann_by_img: Dict[int, List[Dict]],
    cats: Dict[int, str],
    all_cat_ids: List[int],
    target_per_class: int,
    image_dir: Path,
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Augmentasi satu split (train atau val) hingga mencapai target bbox per kelas.

    Parameter:
        split_name      : Nama split (untuk logging)
        images          : List image_info dari split ini
        ann_by_img      : Anotasi per image_id
        cats            : Mapping category_id -> name
        all_cat_ids     : List category_id yang di-balance
        target_per_class: Target bbox per kelas
        image_dir       : Folder tempat gambar (sudah di-enhance) berada
        rng             : Random number generator

    Return:
        (new_images, new_annotations) — daftar lengkap (asli + augmented)
    """
    # Kumpulkan semua images & annotations yang sudah ada
    all_images = copy.deepcopy(images)
    all_annotations = []
    for img_info in images:
        img_id = int(img_info["id"])
        for ann in ann_by_img.get(img_id, []):
            all_annotations.append(copy.deepcopy(ann))

    # Hitung bbox per kelas saat ini
    current_counts = category_counter(all_annotations)

    # Mapping kelas -> gambar yang punya kelas tersebut
    cat_images: Dict[int, List[Dict]] = defaultdict(list)
    for img_info in images:
        img_id = int(img_info["id"])
        seen_cats = set()
        for ann in ann_by_img.get(img_id, []):
            cid = int(ann["category_id"])
            if cid not in seen_cats:
                cat_images[cid].append(img_info)
                seen_cats.add(cid)

    # Cek apakah perlu augmentasi
    needs_aug = any(
        current_counts.get(cid, 0) < target_per_class
        for cid in all_cat_ids
    )
    if not needs_aug:
        print(f"  [{split_name}] Sudah memenuhi target, tidak perlu augmentasi.")
        return all_images, all_annotations

    # Next IDs
    next_img_id = max(int(img["id"]) for img in all_images) + 1
    next_ann_id = max(int(ann["id"]) for ann in all_annotations) + 1 if all_annotations else 0

    total_augmented = 0
    attempts = 0
    max_attempts = 500000

    targets = {cid: target_per_class for cid in all_cat_ids}

    print(f"  [{split_name}] Memulai augmentasi geometri...")
    for cid in all_cat_ids:
        cur = current_counts.get(cid, 0)
        need = targets[cid] - cur
        print(f"    {cats.get(cid, cid):<15}: {cur:>5} → {targets[cid]:>5} (butuh +{max(0, need)})")

    while attempts < max_attempts:
        remaining = {cid: targets[cid] - current_counts.get(cid, 0) for cid in all_cat_ids}
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

        # Baca citra yang SUDAH di-enhance
        src_path = image_dir / img_info["file_name"]
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
        for cid, count in aug_cat_counter.items():
            if current_counts.get(cid, 0) + count > targets.get(cid, 99999) + 50:
                overshoots = True
                break
        if overshoots:
            continue

        # Simpan gambar augmentasi (sudah di-enhance karena sumbernya sudah enhanced)
        original_stem = Path(img_info["file_name"]).stem
        aug_file_name = f"{original_stem}_geo{total_augmented:05d}.jpg"
        aug_path = image_dir / aug_file_name
        cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Update records
        aug_h, aug_w = aug_img.shape[:2]
        new_img_info = {
            "id": next_img_id,
            "file_name": aug_file_name,
            "width": aug_w,
            "height": aug_h,
        }
        all_images.append(new_img_info)

        for bbox, cid in zip(aug_boxes, aug_cats):
            bx, by, bw, bh = bbox
            all_annotations.append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": cid,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            next_ann_id += 1

        current_counts.update(aug_cat_counter)
        next_img_id += 1
        total_augmented += 1

        if total_augmented % 200 == 0:
            rem_str = ", ".join(
                f"{cats.get(c, c)}:{targets[c] - current_counts.get(c, 0)}"
                for c in sorted(all_cat_ids)
            )
            print(f"    [{total_augmented:>5} augmented] remaining: {rem_str}")

    print(f"  [{split_name}] Selesai! Total augmentasi: {total_augmented} gambar baru.")
    for cid in all_cat_ids:
        print(f"    {cats.get(cid, cid):<15}: {current_counts.get(cid, 0):>5} bbox")

    return all_images, all_annotations


# ======================== BUILD COCO JSON ========================

def build_coco_json(
    coco_template: Dict,
    images: List[Dict],
    annotations: List[Dict],
) -> Dict:
    """
    Bangun COCO JSON dengan re-indexing image_id dan annotation_id.
    """
    new_coco = {
        "info": copy.deepcopy(coco_template.get("info", {})),
        "licenses": copy.deepcopy(coco_template.get("licenses", [])),
        "categories": copy.deepcopy(coco_template["categories"]),
        "images": [],
        "annotations": [],
    }

    # Re-index
    old_to_new_img: Dict[int, int] = {}
    for new_idx, img_info in enumerate(images):
        old_id = int(img_info["id"])
        new_img = copy.deepcopy(img_info)
        new_img["id"] = new_idx
        old_to_new_img[old_id] = new_idx
        new_coco["images"].append(new_img)

    new_ann_id = 0
    for ann in annotations:
        old_img_id = int(ann["image_id"])
        if old_img_id not in old_to_new_img:
            continue
        new_ann = copy.deepcopy(ann)
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = old_to_new_img[old_img_id]
        new_coco["annotations"].append(new_ann)
        new_ann_id += 1

    return new_coco


# ======================== MAIN ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split + CLAHE + Median Filter + Augmentasi Geometri Balanced"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/Dataset2026"),
        help="Folder dataset input (berisi gambar + _annotations.coco.json)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/Dataset2026_split_aug_balanced"),
        help="Folder output.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Hapus folder output jika sudah ada")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Target split awal (sebelum augmentasi)
    parser.add_argument("--split-train", type=int, default=SPLIT_TARGET_TRAIN,
                        help="Target bbox per kelas di train sebelum augmentasi")
    parser.add_argument("--split-val", type=int, default=SPLIT_TARGET_VAL,
                        help="Target bbox per kelas di val sebelum augmentasi")
    parser.add_argument("--split-test", type=int, default=SPLIT_TARGET_TEST,
                        help="Target bbox per kelas di test (tanpa augmentasi)")

    # Target augmentasi (setelah augmentasi)
    parser.add_argument("--aug-train", type=int, default=AUG_TARGET_TRAIN,
                        help="Target bbox per kelas di train setelah augmentasi")
    parser.add_argument("--aug-val", type=int, default=AUG_TARGET_VAL,
                        help="Target bbox per kelas di val setelah augmentasi")

    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- Validasi input ----
    annotation_file = args.input / "_annotations.coco.json"
    if not annotation_file.exists():
        print(f"[X] File anotasi tidak ditemukan: {annotation_file}")
        sys.exit(1)

    # ---- Siapkan output ----
    if args.output.exists():
        if not args.overwrite:
            print(f"[X] Folder output {args.output} sudah ada. Gunakan --overwrite.")
            sys.exit(1)
        shutil.rmtree(args.output)

    train_dir = args.output / "train2017"
    val_dir = args.output / "val2017"
    test_dir = args.output / "test2017"
    ann_dir = args.output / "annotations_coco"

    for d in [train_dir, val_dir, test_dir, ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Muat dataset ----
    coco = load_coco(annotation_file)
    cats = {int(c["id"]): c["name"] for c in coco["categories"]}
    # Skip category_id 0 (project label dari Roboflow)
    all_cat_ids = [cid for cid in sorted(cats.keys()) if cid != 0]

    ann_by_img: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[int(ann["image_id"])].append(ann)

    original_counts = category_counter(coco["annotations"])

    print_separator()
    print("  SPLIT + CLAHE + MEDIAN FILTER + AUGMENTASI GEOMETRI BALANCED")
    print_separator()
    print(f"\n  Input           : {args.input}")
    print(f"  Output          : {args.output}")
    print(f"  Total gambar    : {len(coco['images'])}")
    print(f"  Total bbox      : {len(coco['annotations'])}")
    print(f"  Kelas           : {', '.join(cats[cid] for cid in all_cat_ids)}")
    print(f"\n  Distribusi bbox asli:")
    for cid in all_cat_ids:
        print(f"    {cats[cid]:<15}: {original_counts.get(cid, 0):>5} bbox")

    # ======================================================================
    # TAHAP 1: SPLIT DATA BERDASARKAN BBOX
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  TAHAP 1: SPLIT DATA (berdasarkan jumlah bbox per kelas)")
    print(f"{'=' * 70}")
    print(f"  Target split: Train={args.split_train} | Val={args.split_val} | Test={args.split_test} bbox/kelas")

    train_images, val_images, test_images = bbox_balanced_split(
        coco["images"], ann_by_img, all_cat_ids,
        args.split_train, args.split_val, args.split_test,
        rng,
    )

    # Verifikasi: tidak ada data leakage
    train_ids = {int(img["id"]) for img in train_images}
    val_ids = {int(img["id"]) for img in val_images}
    test_ids = {int(img["id"]) for img in test_images}

    assert len(train_ids & val_ids) == 0, "DATA LEAKAGE: ada gambar di train DAN val!"
    assert len(train_ids & test_ids) == 0, "DATA LEAKAGE: ada gambar di train DAN test!"
    assert len(val_ids & test_ids) == 0, "DATA LEAKAGE: ada gambar di val DAN test!"

    # Statistik split
    def count_bbox_per_class(img_list, ann_by_img_dict, cat_ids):
        counts = Counter()
        for img_info in img_list:
            for ann in ann_by_img_dict.get(int(img_info["id"]), []):
                counts[int(ann["category_id"])] += 1
        return counts

    train_bbox = count_bbox_per_class(train_images, ann_by_img, all_cat_ids)
    val_bbox = count_bbox_per_class(val_images, ann_by_img, all_cat_ids)
    test_bbox = count_bbox_per_class(test_images, ann_by_img, all_cat_ids)

    print(f"\n  Hasil split:")
    print(f"  {'Kelas':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-' * 51}")
    for cid in all_cat_ids:
        tr = train_bbox.get(cid, 0)
        va = val_bbox.get(cid, 0)
        te = test_bbox.get(cid, 0)
        print(f"  {cats[cid]:<15} {tr:>8} {va:>8} {te:>8} {tr + va + te:>8}")
    print(f"  {'-' * 51}")
    tr_tot = sum(train_bbox.values())
    va_tot = sum(val_bbox.values())
    te_tot = sum(test_bbox.values())
    print(f"  {'TOTAL':<15} {tr_tot:>8} {va_tot:>8} {te_tot:>8} {tr_tot + va_tot + te_tot:>8}")

    print(f"\n  Gambar: Train={len(train_images)} | Val={len(val_images)} | Test={len(test_images)}")
    print(f"  ✓ Tidak ada data leakage antar split.")

    # ======================================================================
    # TAHAP 2: CLAHE + MEDIAN FILTER KE SEMUA GAMBAR
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  TAHAP 2: CLAHE + MEDIAN FILTER (semua gambar)")
    print(f"{'=' * 70}")

    split_dirs = [
        ("train", train_images, train_dir),
        ("val", val_images, val_dir),
        ("test", test_images, test_dir),
    ]

    for split_name, img_list, target_dir in split_dirs:
        processed = 0
        for img_info in img_list:
            src = args.input / img_info["file_name"]
            dst = target_dir / img_info["file_name"]
            if src.exists():
                img = cv2.imread(str(src))
                if img is not None:
                    img_enhanced = apply_enhancement(img, rng)
                    cv2.imwrite(str(dst), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    processed += 1
                else:
                    print(f"  [!] Gagal baca: {src}")
            else:
                print(f"  [!] Tidak ditemukan: {src}")
        print(f"  [OK] {split_name}: {processed} gambar diproses CLAHE + Median Filter")

    # ======================================================================
    # TAHAP 3: AUGMENTASI GEOMETRI (train & val saja)
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  TAHAP 3: AUGMENTASI GEOMETRI (balancing bbox per kelas)")
    print(f"{'=' * 70}")
    print(f"  Target: Train={args.aug_train} | Val={args.aug_val} bbox/kelas")
    print(f"  Test TIDAK diaugmentasi (tetap {args.split_test} bbox/kelas)")

    # Augmentasi TRAIN
    print(f"\n  --- TRAIN ---")
    train_final_images, train_final_anns = augment_split(
        "TRAIN", train_images, ann_by_img, cats, all_cat_ids,
        args.aug_train, train_dir, rng,
    )

    # Augmentasi VAL
    print(f"\n  --- VAL ---")
    val_final_images, val_final_anns = augment_split(
        "VAL", val_images, ann_by_img, cats, all_cat_ids,
        args.aug_val, val_dir, rng,
    )

    # Test: tidak diaugmentasi, kumpulkan annotations saja
    test_final_images = copy.deepcopy(test_images)
    test_final_anns = []
    for img_info in test_images:
        img_id = int(img_info["id"])
        for ann in ann_by_img.get(img_id, []):
            test_final_anns.append(copy.deepcopy(ann))

    # ======================================================================
    # TAHAP 4: SIMPAN JSON COCO
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  TAHAP 4: MENYIMPAN FILE JSON COCO")
    print(f"{'=' * 70}")

    train_coco = build_coco_json(coco, train_final_images, train_final_anns)
    val_coco = build_coco_json(coco, val_final_images, val_final_anns)
    test_coco = build_coco_json(coco, test_final_images, test_final_anns)

    json_map = [
        ("instances_train2017.json", train_coco),
        ("instances_val2017.json", val_coco),
        ("instances_test2017.json", test_coco),
    ]

    for json_name, split_coco in json_map:
        json_path = ann_dir / json_name
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(split_coco, f, ensure_ascii=False)
        print(f"  [OK] {json_path}")

    # ======================================================================
    # TAHAP 5: RINGKASAN AKHIR
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("  RINGKASAN AKHIR")
    print(f"{'=' * 70}")

    print(f"\n  Struktur output:")
    print(f"    {args.output}/")
    print(f"    ├── annotations_coco/")
    print(f"    │   ├── instances_train2017.json")
    print(f"    │   ├── instances_val2017.json")
    print(f"    │   └── instances_test2017.json")
    print(f"    ├── train2017/  ({len(train_coco['images'])} gambar)")
    print(f"    ├── val2017/    ({len(val_coco['images'])} gambar)")
    print(f"    └── test2017/   ({len(test_coco['images'])} gambar)")

    print(f"\n  DISTRIBUSI BBOX PER KELAS PER SPLIT (SETELAH AUGMENTASI)")
    print(f"  {'Kelas':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-' * 51}")

    for cid in all_cat_ids:
        tr = sum(1 for a in train_coco["annotations"] if int(a["category_id"]) == cid)
        va = sum(1 for a in val_coco["annotations"] if int(a["category_id"]) == cid)
        te = sum(1 for a in test_coco["annotations"] if int(a["category_id"]) == cid)
        total = tr + va + te
        print(f"  {cats[cid]:<15} {tr:>8} {va:>8} {te:>8} {total:>8}")

    tr_total = len(train_coco["annotations"])
    va_total = len(val_coco["annotations"])
    te_total = len(test_coco["annotations"])
    print(f"  {'-' * 51}")
    print(f"  {'TOTAL':<15} {tr_total:>8} {va_total:>8} {te_total:>8} {tr_total + va_total + te_total:>8}")

    # Verifikasi no data leakage
    train_files = {img["file_name"] for img in train_coco["images"]}
    val_files = {img["file_name"] for img in val_coco["images"]}
    test_files = {img["file_name"] for img in test_coco["images"]}

    # Ambil stem asli (tanpa suffix _geoXXXXX) untuk verifikasi leakage
    def get_original_stems(file_set):
        stems = set()
        for f in file_set:
            stem = Path(f).stem
            if "_geo" in stem:
                idx = stem.rfind("_geo")
                stem = stem[:idx]
            stems.add(stem)
        return stems

    train_stems = get_original_stems(train_files)
    val_stems = get_original_stems(val_files)
    test_stems = get_original_stems(test_files)

    leak_tv = train_stems & val_stems  # Diperbolehkan: augmentasi bisa membuat gambar dari sumber berbeda
    leak_tt = train_stems & test_stems
    leak_vt = val_stems & test_stems

    print(f"\n  VERIFIKASI DATA LEAKAGE:")
    if len(leak_tt) == 0 and len(leak_vt) == 0:
        print(f"  ✓ AMAN: Tidak ada gambar test yang bocor ke train/val.")
    else:
        if leak_tt:
            print(f"  ✗ PERINGATAN: {len(leak_tt)} stem gambar ada di TRAIN dan TEST!")
        if leak_vt:
            print(f"  ✗ PERINGATAN: {len(leak_vt)} stem gambar ada di VAL dan TEST!")

    print(f"\n{'=' * 70}")
    print("  SELESAI!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
