"""
augment_geometry_clahe.py

Script augmentasi dataset offline yang FOKUS HANYA PADA GEOMETRI & CLAHE.

Sesuai permintaan:
1. SEMUA gambar di dataset (baik data asli maupun augmentasi) akan dikenakan
   teknik CLAHE terlebih dahulu agar kontrasnya maksimal dan merata.
2. Augmentasi HANYA menggunakan transformasi spasial (Geometri):
   - Horizontal Flip
   - Vertical Flip
   - Rotate (±15 derajat)
   - Zooming / Crop (Scale In)
   (Kombinasi 1-4 teknik ini diterapkan secara acak per gambar augmentasi)
3. Target penyeimbangan: ~5000 bbox per kelas (moler, slabung, ulat_grayak).

Input:
  data/Dataset2026/

Output:
  data/Dataset2026_clahe_geo/

Cara pakai:
  python augment_geometry_clahe.py --overwrite
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

# ======================== TEKNIK AUGMENTASI ========================

def apply_clahe(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Terapkan CLAHE ke gambar. Setting ini sama persis dengan yang Anda sukai.
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
    # Kita tracking valid_indices supaya sinkron dengan category_id nanti
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
        # Jika tidak ada zoom, ambil box yang valid (tidak terpotong secara ekstrem)
        result_boxes = [result_boxes[i] for i in valid_indices]

    return result, result_boxes, valid_indices


# ======================== UTILITAS ========================

def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco

def category_counter(annotations) -> Counter:
    return Counter(int(ann["category_id"]) for ann in annotations)

# ======================== MAIN LOGIC ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/Dataset2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/Dataset2026_clahe_geo"))
    parser.add_argument("--target-moler", type=int, default=5000)
    parser.add_argument("--target-slabung", type=int, default=5000)
    parser.add_argument("--target-ulat-grayak", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    rng = random.Random(SEED)
    np.random.seed(SEED)

    if args.output_dir.exists():
        if not args.overwrite:
            print(f"[X] Folder output {args.output_dir} sudah ada. Gunakan --overwrite.")
            sys.exit(1)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = args.input_dir / "_annotations.coco.json"
    coco = load_coco(annotation_file)
    cats = {int(c["id"]): c["name"] for c in coco["categories"]}
    
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[int(ann["image_id"])].append(ann)

    original_counts = category_counter(coco["annotations"])
    name_to_id = {name.lower().replace("-", "_").replace(" ", "_"): cat_id for cat_id, name in cats.items()}
    
    targets = {
        name_to_id["moler"]: args.target_moler,
        name_to_id["slabung"]: args.target_slabung,
        name_to_id["ulat_grayak"]: args.target_ulat_grayak,
    }

    print("=====================================================")
    print("  CLAHE-GEO AUGMENTATION PIPELINE")
    print("=====================================================")
    
    # ---- 1. Proses Data Asli dengan CLAHE ----
    print("\n[1/3] Menerapkan CLAHE ke seluruh dataset asli...")
    copied = 0
    for img_info in coco["images"]:
        src = args.input_dir / img_info["file_name"]
        dst = args.output_dir / img_info["file_name"]
        if src.exists():
            img = cv2.imread(str(src))
            if img is not None:
                img_clahe = apply_clahe(img, rng)
                cv2.imwrite(str(dst), img_clahe, [cv2.IMWRITE_JPEG_QUALITY, 95])
                copied += 1
    print(f"  [OK] {copied} gambar asli telah diproses CLAHE.")

    # ---- 2. Proses Augmentasi Balancing ----
    print("\n[2/3] Menerapkan Augmentasi Geometri...")
    cat_images = defaultdict(list)
    for img_info in coco["images"]:
        img_id = int(img_info["id"])
        seen_cats = set()
        for ann in ann_by_img.get(img_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in seen_cats:
                cat_images[cat_id].append(img_info)
                seen_cats.add(cat_id)

    new_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": copy.deepcopy(coco["categories"]),
        "images": copy.deepcopy(coco["images"]),
        "annotations": copy.deepcopy(coco["annotations"]),
    }

    next_img_id = max(int(img["id"]) for img in coco["images"]) + 1
    next_ann_id = max(int(ann["id"]) for ann in coco["annotations"]) + 1 if coco["annotations"] else 0
    current_counts = Counter(original_counts)
    
    total_augmented = 0
    attempts = 0
    max_attempts = 500000

    while attempts < max_attempts:
        remaining = {cid: target - current_counts.get(cid, 0) for cid, target in targets.items()}
        needed_cats = [cid for cid, need in remaining.items() if need > 0]
        if not needed_cats:
            break

        attempts += 1
        weights = [remaining[cid] for cid in needed_cats]
        selected_cat = rng.choices(needed_cats, weights=weights, k=1)[0]

        candidates = cat_images.get(selected_cat, [])
        if not candidates: continue
        
        img_info = rng.choice(candidates)
        img_id = int(img_info["id"])
        anns = ann_by_img.get(img_id, [])
        if not anns: continue

        # Baca citra yang SUDAH di-CLAHE dari output_dir
        # Hal ini menjamin citra asli dan augmentasi punya basis CLAHE yang persis sama
        src_path = args.output_dir / img_info["file_name"]
        if not src_path.exists(): continue

        img = cv2.imread(str(src_path))
        if img is None: continue

        boxes = []
        cat_ids_list = []
        for ann in anns:
            boxes.append(ann["bbox"])
            cat_ids_list.append(int(ann["category_id"]))

        # Lakukan transformasi Geometri saja
        aug_img, aug_boxes, valid_indices = aug_spatial(img, boxes, rng)
        if not aug_boxes: continue

        # Filter categories sesuai valid_indices
        aug_cats = [cat_ids_list[i] for i in valid_indices]

        aug_cat_counter = Counter(aug_cats)
        if aug_cat_counter.get(selected_cat, 0) <= 0: continue

        # Pengecekan overshoot target
        overshoots = False
        for cat_id, count in aug_cat_counter.items():
            if current_counts.get(cat_id, 0) + count > targets.get(cat_id, 99999) + 50:
                overshoots = True; break
        if overshoots: continue

        # Simpan gambar augmentasi
        original_stem = Path(img_info["file_name"]).stem
        aug_file_name = f"{original_stem}_geo{total_augmented:05d}.jpg"
        aug_path = args.output_dir / aug_file_name
        cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Update JSON record
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
            rem_str = ", ".join(f"{cats[c]}:{targets[c]-current_counts.get(c,0)}" for c in sorted(targets))
            print(f"  [{total_augmented:>5} augmented] remaining bbox: {rem_str}")

    # ---- 3. Simpan JSON COCO ----
    print("\n[3/3] Menyimpan file _annotations.coco.json...")
    with (args.output_dir / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False)

    print("\n=====================================================")
    print("  HASIL AKHIR AUGMENTASI GEOMETRI + CLAHE")
    print("=====================================================")
    for cat_id in sorted(targets):
        orig = original_counts.get(cat_id, 0)
        final = current_counts.get(cat_id, 0)
        print(f"  {cats[cat_id]:<15} : {orig:>5} -> {final:>5} bbox")
    print(f"  Total gambar akhir : {len(new_coco['images'])}")
    print(f"  Folder output      : {args.output_dir}")
    print("=====================================================")

if __name__ == "__main__":
    main()
