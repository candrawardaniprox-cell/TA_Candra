"""
Skrip mandiri untuk membuat salinan dataset COCO tanpa kelas `moler`.

Tujuan:
- Tidak mengubah model atau file training apa pun.
- Tidak dipanggil oleh file lain.
- Membuat dataset baru terpisah agar dataset lama tetap aman.

Perilaku default:
- Membaca dataset dari `data/coco copy/`
- Memproses `train2017`, `val2017`, `test2017`
- Menghapus kategori bernama `moler`
- Menghapus anotasi milik `moler`
- Menghapus gambar yang setelah filtering tidak punya anotasi lagi
- Merapikan `category_id` sisa kelas menjadi berurutan mulai dari 1
- Menyalin gambar yang masih dipakai ke folder output baru

Contoh:
    python filter_out_moler_dataset.py

    python filter_out_moler_dataset.py --output-root "data/coco tanpa moler"

    python filter_out_moler_dataset.py --keep-empty-images

    python filter_out_moler_dataset.py --skip-image-copy
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitSummary:
    split_name: str
    images_before: int
    images_after: int
    annotations_before: int
    annotations_after: int
    removed_annotations: int
    removed_images: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buat dataset COCO baru tanpa kelas `moler`."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data") / "coco copy",
        help="Root dataset input COCO.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "coco_no_moler",
        help="Root dataset output baru.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="moler",
        help="Nama kelas yang ingin dihapus.",
    )
    parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="Pertahankan gambar yang setelah filtering tidak punya anotasi lagi.",
    )
    parser.add_argument(
        "--skip-image-copy",
        action="store_true",
        help="Jangan salin gambar; hanya buat file anotasi baru.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def find_target_category_id(categories: list[dict], target_name: str) -> int:
    for category in categories:
        if str(category.get("name", "")).strip().lower() == target_name.strip().lower():
            return int(category["id"])
    available = ", ".join(str(cat.get("name", "?")) for cat in categories)
    raise ValueError(
        f"Kelas target '{target_name}' tidak ditemukan. Kelas tersedia: {available}"
    )


def build_category_mapping(categories: list[dict], removed_category_id: int) -> tuple[list[dict], dict[int, int]]:
    kept_categories = [cat for cat in categories if int(cat["id"]) != removed_category_id]
    new_categories = []
    category_id_map: dict[int, int] = {}

    for new_id, category in enumerate(kept_categories, start=1):
        old_id = int(category["id"])
        updated = dict(category)
        updated["id"] = new_id
        new_categories.append(updated)
        category_id_map[old_id] = new_id

    return new_categories, category_id_map


def filter_split(
    split_name: str,
    image_dir: Path,
    annotation_path: Path,
    output_root: Path,
    target_class_name: str,
    keep_empty_images: bool,
    skip_image_copy: bool,
) -> SplitSummary:
    coco = load_json(annotation_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    target_category_id = find_target_category_id(categories, target_class_name)
    new_categories, category_id_map = build_category_mapping(categories, target_category_id)

    kept_annotations = []
    kept_image_ids = set()
    for ann in annotations:
        old_category_id = int(ann["category_id"])
        if old_category_id == target_category_id:
            continue

        updated_ann = dict(ann)
        updated_ann["category_id"] = category_id_map[old_category_id]
        kept_annotations.append(updated_ann)
        kept_image_ids.add(int(updated_ann["image_id"]))

    if keep_empty_images:
        kept_images = list(images)
    else:
        kept_images = [img for img in images if int(img["id"]) in kept_image_ids]

    kept_image_ids = {int(img["id"]) for img in kept_images}
    if keep_empty_images:
        kept_annotations = [ann for ann in kept_annotations if int(ann["image_id"]) in kept_image_ids]

    output_annotation_path = output_root / "annotations_coco" / annotation_path.name
    output_image_dir = output_root / image_dir.name

    filtered_coco = {
        **coco,
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": new_categories,
    }
    save_json(output_annotation_path, filtered_coco)

    if not skip_image_copy:
        output_image_dir.mkdir(parents=True, exist_ok=True)
        for image_info in kept_images:
            file_name = image_info["file_name"]
            src = image_dir / file_name
            dst = output_image_dir / file_name
            if not src.exists():
                raise FileNotFoundError(
                    f"Gambar tidak ditemukan saat copy split {split_name}: {src}"
                )
            if not dst.exists():
                shutil.copy2(src, dst)

    return SplitSummary(
        split_name=split_name,
        images_before=len(images),
        images_after=len(kept_images),
        annotations_before=len(annotations),
        annotations_after=len(kept_annotations),
        removed_annotations=len(annotations) - len(kept_annotations),
        removed_images=len(images) - len(kept_images),
    )


def main() -> None:
    args = parse_args()

    input_root = args.input_root
    output_root = args.output_root

    split_specs = [
        ("train", input_root / "train2017", input_root / "annotations_coco" / "instances_train2017.json"),
        ("val", input_root / "val2017", input_root / "annotations_coco" / "instances_val2017.json"),
        ("test", input_root / "test2017", input_root / "annotations_coco" / "instances_test2017.json"),
    ]

    missing = [str(path) for _, image_dir, ann_path in split_specs for path in (image_dir, ann_path) if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Path dataset berikut tidak ditemukan:\n{missing_text}")

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split_name, image_dir, annotation_path in split_specs:
        summary = filter_split(
            split_name=split_name,
            image_dir=image_dir,
            annotation_path=annotation_path,
            output_root=output_root,
            target_class_name=args.target_class,
            keep_empty_images=args.keep_empty_images,
            skip_image_copy=args.skip_image_copy,
        )
        summaries.append(summary)

    print("=" * 72)
    print(f"SELESAI MEMBUAT DATASET TANPA KELAS '{args.target_class}'")
    print(f"Input  : {input_root.resolve()}")
    print(f"Output : {output_root.resolve()}")
    print("=" * 72)
    print(
        f"{'Split':<10}{'Img Before':>12}{'Img After':>12}"
        f"{'Ann Before':>14}{'Ann After':>12}{'Ann Removed':>14}{'Img Removed':>14}"
    )
    for item in summaries:
        print(
            f"{item.split_name:<10}{item.images_before:>12,}{item.images_after:>12,}"
            f"{item.annotations_before:>14,}{item.annotations_after:>12,}"
            f"{item.removed_annotations:>14,}{item.removed_images:>14,}"
        )
    print("=" * 72)
    if args.skip_image_copy:
        print("Catatan: gambar tidak disalin karena --skip-image-copy dipakai.")


if __name__ == "__main__":
    main()
