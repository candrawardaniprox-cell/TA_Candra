"""
split_augmented_dataset.py — Split dataset augmented ke format COCO (train/val/test).

Script ini membaca dataset hasil augmentasi dari Dataset2026_augmented
dan membaginya menjadi 3 split (train/val/test) dengan rasio yang dapat dikonfigurasi.

PENTING:
  - Split dilakukan per IMAGE (bukan per bbox).
  - Gambar asli dan gambar augmented-nya akan masuk ke split yang SAMA,
    sehingga tidak ada kebocoran data (data leakage) antara train/val/test.
  - Stratified split berdasarkan kelas dominan per gambar.

Input:
  data/Dataset2026_augmented/
    ├── *.jpg
    └── _annotations.coco.json

Output (struktur seperti 'coco copy'):
  data/Dataset2026_split/
    ├── train2017/
    ├── val2017/
    ├── test2017/
    └── annotations_coco/
        ├── instances_train2017.json
        ├── instances_val2017.json
        └── instances_test2017.json

Cara pakai:
  python split_augmented_dataset.py
  python split_augmented_dataset.py --train 0.70 --val 0.20 --test 0.10
  python split_augmented_dataset.py --input data/Dataset2026_augmented --output data/my_split
  python split_augmented_dataset.py --overwrite
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataset COCO augmented ke train/val/test."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/Dataset2026_augmented"),
        help="Folder dataset input (berisi gambar + _annotations.coco.json)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Folder output. Default: data/Dataset2026_split",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Rasio data train (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.20, help="Rasio data val (default: 0.20)")
    parser.add_argument("--test", type=float, default=0.10, help="Rasio data test (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Hapus folder output jika sudah ada")

    args = parser.parse_args()

    # Validasi rasio
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        parser.error(f"Rasio train+val+test harus = 1.0, tapi dapat {total_ratio:.2f}")

    if args.output is None:
        args.output = Path("data/Dataset2026_split")

    return args


def load_coco(path: Path) -> Dict:
    """Muat file JSON COCO."""
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"JSON COCO tidak punya key wajib: {key}")
    return coco


def get_original_stem(file_name: str) -> str:
    """
    Dapatkan nama asli gambar (tanpa suffix _augXXXXX).
    Contoh:
      '100_jpg.rf.51de4e91b30c30975337a5387bc2b343.jpg' -> '100_jpg.rf.51de4e91b30c30975337a5387bc2b343'
      '100_jpg.rf.51de4e91b30c30975337a5387bc2b343_aug00001.jpg' -> '100_jpg.rf.51de4e91b30c30975337a5387bc2b343'
    """
    stem = Path(file_name).stem
    # Cek apakah ini gambar augmented (mengandung _augXXXXX di akhir)
    if "_aug" in stem:
        # Potong dari _aug terakhir
        idx = stem.rfind("_aug")
        return stem[:idx]
    return stem


def group_images_by_original(images: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Kelompokkan gambar berdasarkan gambar aslinya.
    Gambar asli dan semua augmented-nya akan masuk ke grup yang sama.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for img_info in images:
        original = get_original_stem(img_info["file_name"])
        groups[original].append(img_info)
    return groups


def bbox_counts_per_group(
    groups: Dict[str, List[Dict]],
    ann_by_img: Dict[int, List[Dict]],
) -> Dict[str, Counter]:
    """
    Hitung jumlah bbox per kelas untuk setiap grup gambar.
    """
    result: Dict[str, Counter] = {}
    for group_key, img_list in groups.items():
        cat_counter: Counter = Counter()
        for img_info in img_list:
            img_id = int(img_info["id"])
            for ann in ann_by_img.get(img_id, []):
                cat_counter[int(ann["category_id"])] += 1
        result[group_key] = cat_counter
    return result


def bbox_balanced_split(
    group_keys: List[str],
    group_bbox_counts: Dict[str, Counter],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    all_cat_ids: List[int],
    rng: random.Random,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Split grup gambar agar jumlah bbox per kelas SEIMBANG di setiap split.

    Algoritma greedy:
    1. Hitung total bbox per kelas dari seluruh dataset.
    2. Hitung target bbox per kelas per split (total * rasio).
    3. Acak urutan grup, lalu untuk setiap grup:
       - Hitung "skor kekurangan" setiap split = seberapa jauh split itu
         dari target per kelasnya (rata-rata defisit per kelas).
       - Assign grup ke split yang paling tertinggal (defisit terbesar).
    """
    # Hitung total bbox per kelas
    total_per_cat: Counter = Counter()
    for key in group_keys:
        total_per_cat.update(group_bbox_counts[key])

    # Target bbox per kelas per split
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    targets: Dict[str, Dict[int, float]] = {}
    for split_name, ratio in ratios.items():
        targets[split_name] = {
            cat_id: total_per_cat.get(cat_id, 0) * ratio
            for cat_id in all_cat_ids
        }

    # Current bbox per kelas per split (mulai dari 0)
    current: Dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    # Acak urutan untuk menghindari bias
    shuffled_keys = list(group_keys)
    rng.shuffle(shuffled_keys)

    # Assignment per grup
    assignments: Dict[str, Set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    for group_key in shuffled_keys:
        group_counts = group_bbox_counts[group_key]

        # Hitung skor defisit untuk setiap split
        # Defisit = rata-rata (target - current) / target per kelas yang ada di grup ini
        best_split = "train"
        best_score = -float("inf")

        for split_name in ["train", "val", "test"]:
            deficits = []
            for cat_id in all_cat_ids:
                target = targets[split_name].get(cat_id, 0)
                if target <= 0:
                    continue
                cur = current[split_name].get(cat_id, 0)
                # Defisit ternormalisasi: seberapa jauh dari target (0-1)
                deficit = (target - cur) / target
                # Beri bobot lebih tinggi untuk kelas yang ada di grup ini
                if group_counts.get(cat_id, 0) > 0:
                    deficit *= 2.0
                deficits.append(deficit)

            score = sum(deficits) / len(deficits) if deficits else 0
            if score > best_score:
                best_score = score
                best_split = split_name

        # Assign ke split terbaik
        assignments[best_split].add(group_key)
        current[best_split].update(group_counts)

    return assignments["train"], assignments["val"], assignments["test"]


def build_split_coco(
    coco: Dict,
    image_ids: Set[int],
    ann_by_img: Dict[int, List[Dict]],
) -> Dict:
    """
    Bangun objek COCO baru hanya untuk image_ids tertentu.
    Re-index image_id dan annotation_id mulai dari 0.
    """
    new_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": copy.deepcopy(coco["categories"]),
        "images": [],
        "annotations": [],
    }

    # Re-index
    new_img_id = 0
    new_ann_id = 0
    old_to_new_img: Dict[int, int] = {}

    for img_info in coco["images"]:
        old_id = int(img_info["id"])
        if old_id not in image_ids:
            continue

        new_img = copy.deepcopy(img_info)
        new_img["id"] = new_img_id
        old_to_new_img[old_id] = new_img_id
        new_coco["images"].append(new_img)
        new_img_id += 1

    for old_id, new_id in old_to_new_img.items():
        for ann in ann_by_img.get(old_id, []):
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = new_id
            new_coco["annotations"].append(new_ann)
            new_ann_id += 1

    return new_coco


def print_split_stats(split_name: str, coco_data: Dict, cats: Dict[int, str]):
    """Cetak statistik per split."""
    img_count = len(coco_data["images"])
    counts = Counter(int(ann["category_id"]) for ann in coco_data["annotations"])
    total_bbox = sum(counts.values())
    bbox_str = ", ".join(f"{cats.get(k, k)}={counts.get(k, 0)}" for k in sorted(counts))
    print(f"  {split_name:<10}: {img_count:>6} gambar | {total_bbox:>6} bbox | {bbox_str}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # ---- 1. Baca dataset ----
    annotation_file = args.input / "_annotations.coco.json"
    if not annotation_file.exists():
        print(f"[X] File anotasi tidak ditemukan: {annotation_file}")
        sys.exit(1)

    coco = load_coco(annotation_file)
    cats = {int(c["id"]): c["name"] for c in coco["categories"]}

    # Kelompokkan anotasi per image
    ann_by_img: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[int(ann["image_id"])].append(ann)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  SPLIT DATASET AUGMENTED KE FORMAT COCO")
    print(f"  Rasio: Train={args.train:.0%} | Val={args.val:.0%} | Test={args.test:.0%}")
    print(sep)

    # ---- 2. Kelompokkan gambar asli + augmented-nya ----
    groups = group_images_by_original(coco["images"])
    print(f"\n  Total gambar           : {len(coco['images'])}")
    print(f"  Total grup (asli unik) : {len(groups)}")

    # ---- 3. Bbox-balanced split per grup ----
    group_bbox = bbox_counts_per_group(groups, ann_by_img)
    group_keys = sorted(groups.keys())
    # Kelas yang di-balance (skip cat_id=0 yang merupakan label project Roboflow)
    all_cat_ids = [cat_id for cat_id in sorted(cats.keys()) if cat_id != 0]

    train_group_keys, val_group_keys, test_group_keys = bbox_balanced_split(
        group_keys, group_bbox, args.train, args.val, args.test,
        all_cat_ids, rng,
    )

    print(f"\n  Grup train : {len(train_group_keys)}")
    print(f"  Grup val   : {len(val_group_keys)}")
    print(f"  Grup test  : {len(test_group_keys)}")

    # Kumpulkan image_ids per split
    def collect_image_ids(group_key_set: Set[str]) -> Set[int]:
        ids: Set[int] = set()
        for key in group_key_set:
            for img_info in groups[key]:
                ids.add(int(img_info["id"]))
        return ids

    train_img_ids = collect_image_ids(train_group_keys)
    val_img_ids = collect_image_ids(val_group_keys)
    test_img_ids = collect_image_ids(test_group_keys)

    # ---- 4. Bangun COCO per split ----
    train_coco = build_split_coco(coco, train_img_ids, ann_by_img)
    val_coco = build_split_coco(coco, val_img_ids, ann_by_img)
    test_coco = build_split_coco(coco, test_img_ids, ann_by_img)

    print(f"\n  Statistik per split:")
    print_split_stats("TRAIN", train_coco, cats)
    print_split_stats("VAL", val_coco, cats)
    print_split_stats("TEST", test_coco, cats)

    # ---- 5. Siapkan folder output ----
    if args.output.exists():
        if not args.overwrite:
            print(f"\n[X] Folder output sudah ada: {args.output}")
            print("    Gunakan --overwrite untuk menimpa.")
            sys.exit(1)
        shutil.rmtree(args.output)

    train_dir = args.output / "train2017"
    val_dir = args.output / "val2017"
    test_dir = args.output / "test2017"
    ann_dir = args.output / "annotations_coco"

    for d in [train_dir, val_dir, test_dir, ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- 6. Salin gambar ke folder masing-masing ----
    print(f"\n  Menyalin gambar ke folder output...")

    split_map = [
        ("train", train_coco, train_dir),
        ("val", val_coco, val_dir),
        ("test", test_coco, test_dir),
    ]

    for split_name, split_coco, target_dir in split_map:
        copied = 0
        for img_info in split_coco["images"]:
            src = args.input / img_info["file_name"]
            dst = target_dir / img_info["file_name"]
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"  [!] Gambar tidak ditemukan: {src}")
        print(f"  [OK] {split_name}: {copied} gambar disalin")

    # ---- 7. Tulis JSON COCO per split ----
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

    # ---- 8. Ringkasan akhir ----
    total_bbox_all = sum(
        len(s["annotations"]) for _, s, _ in split_map
    )
    print(f"\n{sep}")
    print("  SPLIT DATASET SELESAI!")
    print(sep)
    print(f"  Output folder   : {args.output}")
    print(f"  Struktur output :")
    print(f"    {args.output}/")
    print(f"    ├── train2017/           ({len(train_coco['images'])} gambar)")
    print(f"    ├── val2017/             ({len(val_coco['images'])} gambar)")
    print(f"    ├── test2017/            ({len(test_coco['images'])} gambar)")
    print(f"    └── annotations_coco/")
    print(f"        ├── instances_train2017.json")
    print(f"        ├── instances_val2017.json")
    print(f"        └── instances_test2017.json")
    print(f"\n  Total bbox semua split: {total_bbox_all}")
    print(sep)

    # Tabel distribusi per kelas per split
    print(f"\n{sep}")
    print("  DISTRIBUSI BBOX PER KELAS PER SPLIT")
    print(sep)
    header = f"  {'Kelas':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
    print(header)
    print(f"  {'-' * 51}")

    for cat_id in sorted(cats):
        if cat_id == 0:
            continue  # Skip project label
        name = cats[cat_id]
        tr = sum(1 for a in train_coco["annotations"] if int(a["category_id"]) == cat_id)
        va = sum(1 for a in val_coco["annotations"] if int(a["category_id"]) == cat_id)
        te = sum(1 for a in test_coco["annotations"] if int(a["category_id"]) == cat_id)
        total = tr + va + te
        print(f"  {name:<15} {tr:>8} {va:>8} {te:>8} {total:>8}")

    tr_total = len(train_coco["annotations"])
    va_total = len(val_coco["annotations"])
    te_total = len(test_coco["annotations"])
    print(f"  {'-' * 51}")
    print(f"  {'TOTAL':<15} {tr_total:>8} {va_total:>8} {te_total:>8} {tr_total+va_total+te_total:>8}")
    print(sep)


if __name__ == "__main__":
    main()
