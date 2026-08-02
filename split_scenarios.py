from __future__ import annotations
"""
split_scenarios.py — Membagi data latih ke 4 skenario penelitian.

Skenario:
  - scenario_25  : 25% data latih (Few-Shot)
  - scenario_50  : 50% data latih (Moderate)
  - scenario_75  : 75% data latih (Extensive)
  - scenario_100 : 100% data latih (Full)

Fitur:
  - Stratified split: distribusi kelas (moler, slabung, ulat_grayak) dipertahankan.
  - Nested/hierarchical: 25% < 50% < 75% < 100%.
  - Mengcopy gambar ke folder masing-masing skenario.
  - Menghasilkan file COCO annotation JSON yang sesuai.
  - Val dan test data tetap sama (tidak dipecah).

Penggunaan:
  python split_scenarios.py

Hasil:
  data/
    scenario_25/
      train2017/          ← gambar 25%
      val2017/            ← symlink/copy dari val asli
      test2017/           ← symlink/copy dari test asli
      annotations_coco/
        instances_train2017.json
        instances_val2017.json
        instances_test2017.json
    scenario_50/  ...
    scenario_75/  ...
    scenario_100/ ...
"""

import json
import shutil
import os
import sys
import random
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

# ======================== KONFIGURASI ========================
SEED = 42
SOURCE_DIR = Path("data") / "coco copy"
OUTPUT_BASE = Path("data")
SCENARIOS = {
    "scenario_25": 0.25,
    "scenario_50": 0.50,
    "scenario_75": 0.75,
    "scenario_100": 1.00,
}

# ======================== FUNGSI UTAMA ========================

def load_coco_json(json_path: Path) -> dict:
    """Muat file COCO JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coco_json(data: dict, json_path: Path):
    """Simpan file COCO JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: {json_path} ({len(data.get('images', []))} images, "
          f"{len(data.get('annotations', []))} annotations)")


def build_image_to_categories(coco_data: dict) -> dict:
    """
    Bangun mapping: image_id -> set(category_ids).
    Digunakan untuk stratified split berdasarkan kelas dominan per gambar.
    """
    img2cats = defaultdict(set)
    for ann in coco_data["annotations"]:
        img2cats[ann["image_id"]].add(ann["category_id"])
    return img2cats


def get_dominant_class(category_ids: set) -> int:
    """
    Untuk gambar dengan multi-class, gunakan category_id terkecil sebagai kelas
    dominan. Ini memastikan konsistensi dalam stratified split.
    """
    return min(category_ids)


def stratified_split_images(coco_data: dict, seed: int = 42):
    """
    Lakukan stratified split pada gambar.
    
    Returns:
        class_groups: dict[int, list[int]] - mapping class_id -> list of image_ids
                      (sudah di-shuffle)
    """
    random.seed(seed)
    
    img2cats = build_image_to_categories(coco_data)
    
    # Kelompokkan gambar berdasarkan kelas dominan
    class_groups = defaultdict(list)
    
    for img in coco_data["images"]:
        img_id = img["id"]
        if img_id in img2cats:
            dominant = get_dominant_class(img2cats[img_id])
            class_groups[dominant].append(img_id)
        else:
            # Gambar tanpa anotasi — masukkan ke grup khusus -1
            class_groups[-1].append(img_id)
    
    # Shuffle setiap grup
    for cls_id in class_groups:
        random.shuffle(class_groups[cls_id])
    
    return class_groups


def select_images_for_ratio(class_groups: dict, ratio: float) -> set:
    """
    Pilih image_ids sesuai rasio, stratified per kelas.
    
    Karena nested (25% ⊂ 50% ⊂ 75% ⊂ 100%), urutan shuffle tetap sama
    (seed fixed), jadi kita cukup ambil N pertama dari setiap grup.
    """
    selected = set()
    for cls_id, img_ids in class_groups.items():
        n_select = max(1, int(len(img_ids) * ratio))  # minimal 1
        if ratio >= 1.0:
            n_select = len(img_ids)
        selected.update(img_ids[:n_select])
    return selected


def filter_coco_data(coco_data: dict, selected_image_ids: set) -> dict:
    """
    Filter COCO data: hanya simpan images dan annotations yang masuk seleksi.
    """
    filtered = deepcopy(coco_data)
    
    # Filter images
    filtered["images"] = [
        img for img in coco_data["images"] 
        if img["id"] in selected_image_ids
    ]
    
    # Filter annotations
    filtered["annotations"] = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in selected_image_ids
    ]
    
    return filtered


def copy_images(image_list: list, src_dir: Path, dst_dir: Path):
    """Copy gambar dari src ke dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for img_info in image_list:
        fname = img_info["file_name"]
        src = src_dir / fname
        dst = dst_dir / fname
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                skipped += 1
        else:
            print(f"  [WARN] Gambar tidak ditemukan: {src}")
    print(f"  [OK] Copied {copied} images, skipped {skipped} (already exist)")


def copy_directory(src: Path, dst: Path, label: str):
    """Copy seluruh folder (untuk val/test)."""
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n_files = len(list(dst.iterdir()))
        print(f"  [OK] {label}: {n_files} files copied")
    else:
        print(f"  [WARN] {label} source not found: {src}")


def print_distribution(coco_data: dict, categories: list):
    """Cetak distribusi kelas dari COCO data."""
    cat_map = {c["id"]: c["name"] for c in categories}
    counts = defaultdict(int)
    for ann in coco_data["annotations"]:
        counts[ann["category_id"]] += 1
    
    parts = []
    for cid in sorted(counts.keys()):
        name = cat_map.get(cid, f"cls_{cid}")
        parts.append(f"{name}={counts[cid]}")
    print(f"    Distribusi: {', '.join(parts)}")


def main():
    print("=" * 65)
    print("  SPLIT SKENARIO DATA LATIH - Hybrid CNN-Transformer")
    print("  Stratified & Nested Split (25% < 50% < 75% < 100%)")
    print("=" * 65)
    
    # ---------- Muat data ----------
    train_json_path = SOURCE_DIR / "annotations_coco" / "instances_train2017.json"
    val_json_path = SOURCE_DIR / "annotations_coco" / "instances_val2017.json"
    test_json_path = SOURCE_DIR / "annotations_coco" / "instances_test2017.json"
    
    print(f"\n[DIR] Source: {SOURCE_DIR}")
    print(f"[FILE] Loading training annotations: {train_json_path}")
    
    train_data = load_coco_json(train_json_path)
    categories = train_data["categories"]
    
    total_images = len(train_data["images"])
    total_anns = len(train_data["annotations"])
    
    print(f"\n[STATS] Dataset Asli:")
    print(f"   Total gambar latih : {total_images}")
    print(f"   Total anotasi      : {total_anns}")
    print_distribution(train_data, categories)
    
    # ---------- Stratified split ----------
    print(f"\n[SPLIT] Melakukan stratified split (seed={SEED})...")
    class_groups = stratified_split_images(train_data, seed=SEED)
    
    for cls_id in sorted(class_groups.keys()):
        cat_name = next((c["name"] for c in categories if c["id"] == cls_id), f"no_ann_{cls_id}")
        print(f"   Kelas '{cat_name}' (id={cls_id}): {len(class_groups[cls_id])} gambar")
    
    # ---------- Proses setiap skenario ----------
    for scenario_name, ratio in SCENARIOS.items():
        print(f"\n{'-' * 65}")
        print(f"[SCENARIO] Skenario: {scenario_name} (rasio={ratio*100:.0f}%)")
        print(f"{'-' * 65}")
        
        scenario_dir = OUTPUT_BASE / scenario_name
        
        # Pilih gambar
        selected_ids = select_images_for_ratio(class_groups, ratio)
        print(f"  [IMG] Gambar terpilih: {len(selected_ids)} / {total_images} "
              f"({len(selected_ids)/total_images*100:.1f}%)")
        
        # Filter COCO data
        filtered_data = filter_coco_data(train_data, selected_ids)
        print(f"  [ANN] Anotasi terpilih: {len(filtered_data['annotations'])} / {total_anns}")
        print_distribution(filtered_data, categories)
        
        # Detail per kelas
        for cls_id in sorted(class_groups.keys()):
            n_total = len(class_groups[cls_id])
            n_selected = len(selected_ids.intersection(class_groups[cls_id]))
            cat_name = next((c["name"] for c in categories if c["id"] == cls_id), f"cls_{cls_id}")
            print(f"    Kelas '{cat_name}': {n_selected}/{n_total} gambar "
                  f"({n_selected/n_total*100:.1f}%)")
        
        # Simpan annotation JSON
        ann_dir = scenario_dir / "annotations_coco"
        save_coco_json(filtered_data, ann_dir / "instances_train2017.json")
        
        # Copy gambar train
        print(f"\n  [COPY] Copying training images...")
        copy_images(
            filtered_data["images"],
            SOURCE_DIR / "train2017",
            scenario_dir / "train2017"
        )
        
        # Copy val dan test (tetap sama untuk semua skenario)
        print(f"  [COPY] Copying val & test data...")
        copy_directory(
            SOURCE_DIR / "val2017",
            scenario_dir / "val2017",
            "val2017"
        )
        copy_directory(
            SOURCE_DIR / "test2017",
            scenario_dir / "test2017",
            "test2017"
        )
        
        # Copy val & test annotations
        if val_json_path.exists():
            shutil.copy2(val_json_path, ann_dir / "instances_val2017.json")
            print(f"  [OK] Val annotations copied")
        if test_json_path.exists():
            shutil.copy2(test_json_path, ann_dir / "instances_test2017.json")
            print(f"  [OK] Test annotations copied")
    
    # ---------- Ringkasan ----------
    print(f"\n{'=' * 65}")
    print(f"  [DONE] SELESAI! Ringkasan Skenario:")
    print(f"{'=' * 65}")
    print(f"{'Skenario':<20} {'Rasio':>8} {'Gambar':>10} {'Anotasi':>10}")
    print(f"{'-' * 50}")
    
    for scenario_name, ratio in SCENARIOS.items():
        selected_ids = select_images_for_ratio(class_groups, ratio)
        filtered = filter_coco_data(train_data, selected_ids)
        print(f"{scenario_name:<20} {ratio*100:>7.0f}% {len(filtered['images']):>10} "
              f"{len(filtered['annotations']):>10}")
    
    print(f"\n{'-' * 50}")
    print(f"{'Sumber (100%)':<20} {'100':>7}% {total_images:>10} {total_anns:>10}")
    
    # ---------- Verifikasi nested property ----------
    print(f"\n[CHECK] Verifikasi properti nested (25% < 50% < 75% < 100%):")
    prev_ids = set()
    prev_name = ""
    for scenario_name, ratio in SCENARIOS.items():
        curr_ids = select_images_for_ratio(class_groups, ratio)
        if prev_ids:
            is_subset = prev_ids.issubset(curr_ids)
            status = "[OK]" if is_subset else "[FAIL]"
            print(f"  {status} {prev_name} subset {scenario_name}: {is_subset}")
        prev_ids = curr_ids
        prev_name = scenario_name
    
    print(f"\n[INFO] Untuk menggunakan skenario, ubah SCENARIO di config.py:")
    print(f"   DATA_ROOT = Path('data') / 'scenario_25'   # Few-Shot")
    print(f"   DATA_ROOT = Path('data') / 'scenario_50'   # Moderate")
    print(f"   DATA_ROOT = Path('data') / 'scenario_75'   # Extensive")
    print(f"   DATA_ROOT = Path('data') / 'scenario_100'  # Full")
    print(f"\n   Atau gunakan argumen command-line (lihat config.py)")


if __name__ == "__main__":
    main()
