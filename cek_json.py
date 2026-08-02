from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_ORDER = ["moler", "slabung", "ulat_grayak"]
CLASS_LABELS = {
    "moler": "Moler",
    "slabung": "Slabung",
    "ulat_grayak": "Ulat Grayak",
}
CLASS_COLORS = {
    "moler": "#d62728",       # red
    "slabung": "#2ca02c",     # green
    "ulat_grayak": "#1f77b4", # blue
}


def normalize_class_name(name):
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ulatgrayak": "ulat_grayak",
        "ulat__grayak": "ulat_grayak",
    }
    return aliases.get(normalized, normalized)


def detect_split_name(json_path):
    stem = Path(json_path).stem.lower()
    if "train" in stem:
        return "Train"
    if "val" in stem:
        return "Val"
    if "test" in stem:
        return "Test"
    return Path(json_path).stem


def collect_coco_stats(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    category_id_to_name = {}
    for category in data.get("categories", []):
        category_id_to_name[category.get("id")] = normalize_class_name(category.get("name"))

    bbox_counter = Counter()
    image_class_sets = defaultdict(set)

    for ann in data.get("annotations", []):
        category_id = ann.get("category_id")
        image_id = ann.get("image_id")
        class_name = category_id_to_name.get(category_id)
        if class_name is None:
            continue

        bbox_counter[class_name] += 1
        image_class_sets[class_name].add(image_id)

    image_counter = Counter({
        class_name: len(image_ids)
        for class_name, image_ids in image_class_sets.items()
    })

    split_name = detect_split_name(json_path)
    return {
        "json_path": json_path,
        "split_name": split_name,
        "category_id_to_name": category_id_to_name,
        "num_images_total": len(data.get("images", [])),
        "num_annotations_total": len(data.get("annotations", [])),
        "image_counter": image_counter,
        "bbox_counter": bbox_counter,
    }


def print_stats(stats):
    print("=" * 60)
    print(f"Menganalisis file: {stats['json_path']}")
    print("=" * 60)
    print(f"Split                : {stats['split_name']}")
    print(f"Total gambar         : {stats['num_images_total']}")
    print(f"Total bounding box   : {stats['num_annotations_total']}")

    print("\n[1] DAFTAR KATEGORI (CLASSES):")
    seen_names = set()
    for category_id, class_name in sorted(stats["category_id_to_name"].items()):
        if class_name in seen_names:
            continue
        seen_names.add(class_name)
        print(f"    - ID: {category_id} | Nama Kelas: '{CLASS_LABELS.get(class_name, class_name)}'")

    print("\n[2] JUMLAH GAMBAR UNIK PER KELAS:")
    for class_key in CLASS_ORDER:
        print(
            f"    - {CLASS_LABELS[class_key]:<12} : "
            f"{stats['image_counter'].get(class_key, 0)} gambar"
        )

    print("\n[3] JUMLAH BOUNDING BOX PER KELAS:")
    for class_key in CLASS_ORDER:
        print(
            f"    - {CLASS_LABELS[class_key]:<12} : "
            f"{stats['bbox_counter'].get(class_key, 0)} bbox"
        )
    print()


def plot_distribution(stats_list, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [stats["split_name"] for stats in stats_list]
    # Warna sesuai permintaan: Train=Orange, Val=Merah, Test=Biru
    split_colors = {"Train": "orange", "Val": "red", "Test": "blue"}

    # Solusi: Lebarkan kanvas (figsize) secara ekstrim agar batang sangat panjang
    # sehingga font berukuran 22 tetap muat tanpa tumpang tindih.
    fig, axes = plt.subplots(2, 1, figsize=(32, 12), sharey=True)
    fig.patch.set_facecolor("white")

    image_ax = axes[0]
    bbox_ax = axes[1]

    class_keys = CLASS_ORDER
    class_labels_list = [CLASS_LABELS[k] for k in class_keys]

    left_image = np.zeros(len(class_keys))
    left_bbox = np.zeros(len(class_keys))

    handles = []
    
    for stats in stats_list:
        split = stats["split_name"]
        color = split_colors.get(split, "gray")
        
        image_values = np.array([stats["image_counter"].get(k, 0) for k in class_keys])
        bbox_values = np.array([stats["bbox_counter"].get(k, 0) for k in class_keys])
        
        bar_img = image_ax.barh(class_labels_list, image_values, left=left_image, color=color, label=split, alpha=0.9, height=0.6)
        bar_box = bbox_ax.barh(class_labels_list, bbox_values, left=left_bbox, color=color, label=split, alpha=0.9, height=0.6)
        
        text_color = 'white' if split == 'Test' else 'black'
        # Tambahkan label angka di tengah batang (dikembalikan ke 22 sesuai permintaan)
        image_ax.bar_label(bar_img, fmt='%d', label_type='center', color=text_color, fontsize=22, fontweight='bold')
        bbox_ax.bar_label(bar_box, fmt='%d', label_type='center', color=text_color, fontsize=22, fontweight='bold')
        
        # Tambahkan tulisan nama kelas ke dalam batang bagian paling kiri (Train)
        if split == "Train":
            for i, class_name in enumerate(class_labels_list):
                image_ax.text(50, i, class_name, va='center', ha='left', color='white', fontsize=26, fontweight='bold',
                              bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=3))
                bbox_ax.text(50, i, class_name, va='center', ha='left', color='white', fontsize=26, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=3))
        
        left_image += image_values
        left_bbox += bbox_values
        
        if len(handles) < len(split_names):
            handles.append(bar_img[0])

    # Tambahkan nilai TOTAL di luar/ujung kanan setiap batang grafik
    for i, (tot_img, tot_box) in enumerate(zip(left_image, left_bbox)):
        # +30 memberi jarak sedikit dari ujung batang
        image_ax.text(tot_img + 30, i, f"{int(tot_img)}", va='center', ha='left', color='black', fontsize=22, fontweight='bold')
        bbox_ax.text(tot_box + 30, i, f"{int(tot_box)}", va='center', ha='left', color='black', fontsize=22, fontweight='bold')

    image_ax.set_title("Distribusi Gambar per Kelas", fontsize=28, fontweight='bold', pad=12)
    image_ax.set_xlabel("Jumlah Gambar", fontsize=22)
    image_ax.invert_yaxis()  # Agar urutan kelas dari atas ke bawah
    image_ax.grid(True, axis="x", alpha=0.35)
    image_ax.set_axisbelow(True)
    image_ax.set_yticks([])  # Hilangkan tulisan Y axis karena sudah di dalam grafik
    image_ax.tick_params(axis='x', labelsize=20)

    bbox_ax.set_title("Distribusi Bounding Box per Kelas", fontsize=28, fontweight='bold', pad=12)
    bbox_ax.set_xlabel("Jumlah Bounding Box", fontsize=22)
    bbox_ax.grid(True, axis="x", alpha=0.35)
    bbox_ax.set_axisbelow(True)
    bbox_ax.set_yticks([])  # Hilangkan tulisan Y axis karena sudah di dalam grafik
    bbox_ax.tick_params(axis='x', labelsize=20)

    fig.legend(
        handles,
        split_names,
        loc="lower center",
        ncol=len(split_names),
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=22,
    )

    fig.suptitle("Grafik Distribusi Data", fontsize=36, fontweight='bold', y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path = output_dir / "grafik_distribusi_data.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik distribusi data disimpan di: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cek distribusi dataset")
    parser.add_argument("--dataset_dir", type=str, default="data/coco copy", help="Folder dataset utama (misal: data/Dataset2025_splitdulu)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    file_json_saya = [
        dataset_path / "annotations_coco" / "instances_train2017.json",
        dataset_path / "annotations_coco" / "instances_val2017.json",
        dataset_path / "annotations_coco" / "instances_test2017.json",
    ]

    semua_stats = []
    for file_path in file_json_saya:
        if file_path.exists():
            stats = collect_coco_stats(file_path)
            print_stats(stats)
            semua_stats.append(stats)
        else:
            print(f"File tidak ditemukan: {file_path}")

    if semua_stats:
        plot_distribution(
            semua_stats,
            output_dir=Path("outputs") / "dataset_analysis" / dataset_path.name,
        )

