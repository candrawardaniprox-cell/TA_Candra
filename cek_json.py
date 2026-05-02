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


def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_distribution(stats_list, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [stats["split_name"] for stats in stats_list]
    x = np.arange(len(split_names))
    width = 0.22

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.patch.set_facecolor("white")

    image_ax = axes[0]
    bbox_ax = axes[1]

    image_handles = []
    for idx, class_key in enumerate(CLASS_ORDER):
        offsets = x + (idx - 1) * width
        image_values = [stats["image_counter"].get(class_key, 0) for stats in stats_list]
        bbox_values = [stats["bbox_counter"].get(class_key, 0) for stats in stats_list]

        image_bars = image_ax.bar(
            offsets,
            image_values,
            width=width,
            color=CLASS_COLORS[class_key],
            label=CLASS_LABELS[class_key],
            alpha=0.95,
        )
        bbox_bars = bbox_ax.bar(
            offsets,
            bbox_values,
            width=width,
            color=CLASS_COLORS[class_key],
            label=CLASS_LABELS[class_key],
            alpha=0.95,
        )

        add_bar_labels(image_ax, image_bars)
        add_bar_labels(bbox_ax, bbox_bars)
        image_handles.append(image_bars[0])

    image_ax.set_title("Distribusi Gambar per Kelas", fontsize=18, pad=12)
    image_ax.set_ylabel("Jumlah Gambar", fontsize=14)
    image_ax.grid(True, axis="y", alpha=0.35)
    image_ax.set_axisbelow(True)

    bbox_ax.set_title("Distribusi Bounding Box per Kelas", fontsize=18, pad=12)
    bbox_ax.set_ylabel("Jumlah Bounding Box", fontsize=14)
    bbox_ax.set_xlabel("Split Dataset", fontsize=14)
    bbox_ax.grid(True, axis="y", alpha=0.35)
    bbox_ax.set_axisbelow(True)
    bbox_ax.set_xticks(x)
    bbox_ax.set_xticklabels(split_names, fontsize=12)

    fig.legend(
        image_handles,
        [CLASS_LABELS[class_key] for class_key in CLASS_ORDER],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=12,
    )

    fig.suptitle("Grafik Distribusi Data", fontsize=22, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    output_path = output_dir / "grafik_distribusi_data.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik distribusi data disimpan di: {output_path}")


if __name__ == "__main__":
    file_json_saya = [
        "data/coco copy/annotations_coco/instances_train2017.json",
        "data/coco copy/annotations_coco/instances_val2017.json",
        "data/coco copy/annotations_coco/instances_test2017.json",
    ]

    semua_stats = []
    for file_path in file_json_saya:
        stats = collect_coco_stats(file_path)
        print_stats(stats)
        semua_stats.append(stats)

    plot_distribution(
        semua_stats,
        output_dir=Path("outputs") / "dataset_analysis",
    )
