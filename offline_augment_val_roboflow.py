"""
Offline COCO validation augmentation, Roboflow-like.

Script ini berdiri sendiri dan tidak mengubah pipeline training utama.
Default input mengikuti Config:
  data/coco copy/val2017
  data/coco copy/annotations_coco/instances_val2017.json

Default output menjaga nama folder/file COCO yang sama:
  data/coco copy_val_weighted_default/val2017
  data/coco copy_val_weighted_default/annotations_coco/instances_val2017.json

Contoh:
  python offline_augment_val_roboflow.py
  python offline_augment_val_roboflow.py --multiplier 3
  python offline_augment_val_roboflow.py --multiplier 2
  python offline_augment_val_roboflow.py --balance-to 600
  python offline_augment_val_roboflow.py --class-multipliers "moler=1.17,slabung=0.01,ulat_grayak=0.06" --include-original
  python offline_augment_val_roboflow.py --multiplier 3 --include-original
  python offline_augment_val_roboflow.py --multiplier 3 --seed 123 --overwrite

Teknik augmentasi:
  - Flip horizontal dan/atau vertical secara random.
  - Rotasi 90 derajat: clockwise, counter-clockwise, upside down, atau none.
  - Crop random dengan zoom 0% sampai 20%, lalu resize kembali ke ukuran output.

Catatan:
  - Anotasi bbox COCO [x, y, width, height] ikut ditransform.
  - Bbox yang keluar akibat crop akan di-clip dan dibuang bila terlalu kecil.
  - Segmentation diset kosong karena training project ini memakai bbox.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Preset default sesuai permintaan:
# target total validasi = data asli + augmentasi tambahan (target ~195).
# moler       (asli 92)  -> augmentasi tambahan x1.12
# slabung     (asli 199) -> augmentasi tambahan x0.0
# ulat_grayak (asli 193) -> augmentasi tambahan x0.01 (atau 0.0 jika tidak mau di-augment)
DEFAULT_VAL_EXTRA_CLASS_MULTIPLIERS = {
    "moler": 1.12, 
    "slabung": 0.0, 
    "ulat_grayak": 0.01, 
}
DEFAULT_VAL_CLASS_MULTIPLIERS_ARG = ",".join(
    f"{class_name}={value}"
    for class_name, value in DEFAULT_VAL_EXTRA_CLASS_MULTIPLIERS.items()
)


@dataclass(frozen=True)
class AugmentSpec:
    hflip: bool
    vflip: bool
    rotation: str
    zoom: float
    crop_left: float
    crop_top: float

    def tokens(self) -> List[str]:
        parts: List[str] = []
        if self.hflip:
            parts.append("hflip")
        if self.vflip:
            parts.append("vflip")
        if self.rotation != "none":
            parts.append(f"rot{self.rotation}")
        if self.zoom > 1e-6:
            parts.append(f"crop{int(round(self.zoom * 100)):02d}p")
        return parts or ["identity"]


def default_paths() -> Tuple[Path, Path, Path]:
    try:
        from config import Config

        data_root = Path(Config.DATA_ROOT)
        image_dir = Path(Config.VAL_IMAGES)
        annotation_file = Path(Config.VAL_ANNOTATIONS)
    except Exception:
        data_root = Path("data") / "coco copy"
        image_dir = data_root / "val2017"
        annotation_file = data_root / "annotations_coco" / "instances_val2017.json"
    return data_root, image_dir, annotation_file


def parse_args() -> argparse.Namespace:
    data_root, image_dir, annotation_file = default_paths()

    parser = argparse.ArgumentParser(
        description="Buat dataset validasi COCO teraugmentasi secara offline."
    )
    parser.add_argument(
        "--input-images",
        type=Path,
        default=image_dir,
        help=f"Folder gambar val. Default: {image_dir}",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=annotation_file,
        help=f"File JSON COCO val. Default: {annotation_file}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Folder output dataset baru. Default mengikuti mode: "
            "<data_root>_val_weighted_default, "
            "<data_root>_val_augmented_<multiplier>x, atau <data_root>_val_balanced_<target>"
        ),
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Jumlah citra augmentasi per gambar asli. Gunakan 2 atau 3 sesuai kebutuhan.",
    )
    parser.add_argument(
        "--class-multipliers",
        type=str,
        default=DEFAULT_VAL_CLASS_MULTIPLIERS_ARG,
        help=(
            "Bobot augmentasi per kelas, boleh desimal. Contoh: "
            "\"moler=1.17,slabung=0.01,ulat_grayak=0.06\". "
            "Nilai ini adalah jumlah augmentasi tambahan rata-rata per gambar yang punya kelas itu. "
            f"Default: \"{DEFAULT_VAL_CLASS_MULTIPLIERS_ARG}\" untuk total "
            "moler x5.1739, slabung x3.0067, ulat_grayak x3.0638 termasuk data asli."
        ),
    )
    parser.add_argument(
        "--balance-to",
        type=int,
        default=None,
        help=(
            "Mode balance: target jumlah bbox final per kelas. "
            "Contoh --balance-to 600 akan mencoba membuat setiap kelas berisi 600 bbox."
        ),
    )
    parser.add_argument(
        "--balance-targets",
        type=str,
        default=None,
        help=(
            "Target bbox final per kelas jika ingin beda-beda. Contoh: "
            "\"moler=600,slabung=600,ulat_grayak=600\"."
        ),
    )
    parser.add_argument(
        "--balance-without-original",
        action="store_true",
        help=(
            "Untuk mode balance saja: jangan salin data asli ke output. "
            "Default mode balance adalah menyalin data asli dulu, lalu menambah augmentasi."
        ),
    )
    parser.add_argument(
        "--balance-max-attempts",
        type=int,
        default=200000,
        help="Batas percobaan random saat mencari augmentasi yang tidak melewati target balance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed random agar hasil bisa direproduksi.",
    )
    parser.add_argument(
        "--max-zoom",
        type=float,
        default=0.20,
        help="Maksimum zoom crop. 0.20 berarti crop sampai 20 persen.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Kualitas JPEG output.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=2.0,
        help="Bbox hasil augmentasi dibuang bila width/height lebih kecil dari nilai ini.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.05,
        help="Minimum rasio luas bbox yang masih terlihat setelah crop.",
    )
    parser.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Salin gambar dan anotasi asli juga ke output. Default aktif agar multiplier total "
            "menghitung data asli + augmentasi. Pakai --no-include-original jika hanya ingin augmentasi."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus folder output bila sudah ada.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah gambar sumber untuk uji coba cepat.",
    )

    args = parser.parse_args()
    argv = sys.argv[1:]
    explicit_multiplier = "--multiplier" in argv
    explicit_class_multipliers = "--class-multipliers" in argv
    using_balance_mode = args.balance_to is not None or args.balance_targets is not None

    if explicit_multiplier and not explicit_class_multipliers and not using_balance_mode:
        args.class_multipliers = None

    if args.multiplier < 1:
        parser.error("--multiplier harus minimal 1.")
    if args.balance_to is not None and args.balance_to < 1:
        parser.error("--balance-to harus minimal 1.")
    if args.balance_max_attempts < 1:
        parser.error("--balance-max-attempts harus minimal 1.")
    if not 0.0 <= args.max_zoom <= 0.9:
        parser.error("--max-zoom harus antara 0.0 sampai 0.9.")
    if not 0.0 <= args.min_visibility <= 1.0:
        parser.error("--min-visibility harus antara 0.0 sampai 1.0.")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality harus antara 1 sampai 100.")

    if args.output_root is None:
        if args.balance_to is not None:
            args.output_root = data_root.parent / f"{data_root.name}_val_balanced_{args.balance_to}"
        elif args.balance_targets is not None:
            args.output_root = data_root.parent / f"{data_root.name}_val_balanced_custom"
        elif args.class_multipliers == DEFAULT_VAL_CLASS_MULTIPLIERS_ARG and args.include_original:
            args.output_root = data_root.parent / f"{data_root.name}_val_weighted_default"
        elif args.class_multipliers:
            args.output_root = data_root.parent / f"{data_root.name}_val_weighted_custom"
        else:
            args.output_root = data_root.parent / f"{data_root.name}_val_augmented_{args.multiplier}x"

    return args


def load_coco(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"File JSON tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"JSON COCO tidak punya key wajib: {key}")
    return coco


def ensure_output_dirs(output_root: Path, overwrite: bool) -> Tuple[Path, Path]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Folder output sudah ada: {output_root}\n"
                "Gunakan --overwrite atau pilih --output-root lain."
            )
        shutil.rmtree(output_root)

    image_out_dir = output_root / "val2017"
    annotation_out_dir = output_root / "annotations_coco"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    annotation_out_dir.mkdir(parents=True, exist_ok=True)
    return image_out_dir, annotation_out_dir


def annotations_by_image(coco: Dict) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        grouped[int(ann["image_id"])].append(ann)
    return grouped


def choose_spec(rng: random.Random, max_zoom: float) -> AugmentSpec:
    rotation = rng.choice(["none", "cw", "ccw", "180"])
    hflip = rng.random() < 0.5
    vflip = rng.random() < 0.5
    zoom = rng.uniform(0.0, max_zoom)
    crop_left = rng.random()
    crop_top = rng.random()

    if rotation == "none" and not hflip and not vflip and zoom < 0.01:
        forced = rng.choice(["hflip", "vflip", "cw", "ccw", "180", "crop"])
        if forced == "hflip":
            hflip = True
        elif forced == "vflip":
            vflip = True
        elif forced in {"cw", "ccw", "180"}:
            rotation = forced
        else:
            zoom = max(0.05, min(max_zoom, 0.05))

    return AugmentSpec(
        hflip=hflip,
        vflip=vflip,
        rotation=rotation,
        zoom=zoom,
        crop_left=crop_left,
        crop_top=crop_top,
    )


def clip_box_xyxy(
    box: Sequence[float],
    width: float,
    height: float,
) -> Optional[List[float]]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def crop_resize_image_and_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    spec: AugmentSpec,
    min_visibility: float,
) -> Tuple[Image.Image, List[Optional[List[float]]]]:
    width, height = image.size
    crop_w = max(1.0, width * (1.0 - spec.zoom))
    crop_h = max(1.0, height * (1.0 - spec.zoom))
    max_left = max(0.0, width - crop_w)
    max_top = max(0.0, height - crop_h)
    left = max_left * spec.crop_left
    top = max_top * spec.crop_top
    right = left + crop_w
    bottom = top + crop_h

    if spec.zoom > 1e-6:
        cropped = image.crop(
            (
                int(round(left)),
                int(round(top)),
                int(round(right)),
                int(round(bottom)),
            )
        )
        resized = cropped.resize((width, height), resample=Image.Resampling.LANCZOS)
    else:
        resized = image.copy()

    scale_x = width / crop_w
    scale_y = height / crop_h
    transformed: List[Optional[List[float]]] = []

    for box in boxes:
        original_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
        clipped = [
            max(left, box[0]),
            max(top, box[1]),
            min(right, box[2]),
            min(bottom, box[3]),
        ]
        visible_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if original_area <= 0 or visible_area / original_area < min_visibility:
            transformed.append(None)
            continue

        transformed.append(
            [
                (clipped[0] - left) * scale_x,
                (clipped[1] - top) * scale_y,
                (clipped[2] - left) * scale_x,
                (clipped[3] - top) * scale_y,
            ]
        )

    return resized, transformed


def flip_image_and_boxes(
    image: Image.Image,
    boxes: List[Optional[List[float]]],
    hflip: bool,
    vflip: bool,
) -> Tuple[Image.Image, List[Optional[List[float]]]]:
    width, height = image.size
    out = image
    transformed = copy.deepcopy(boxes)

    if hflip:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for idx, box in enumerate(transformed):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            transformed[idx] = [width - x2, y1, width - x1, y2]

    if vflip:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        for idx, box in enumerate(transformed):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            transformed[idx] = [x1, height - y2, x2, height - y1]

    return out, transformed


def rotate_image_and_boxes(
    image: Image.Image,
    boxes: List[Optional[List[float]]],
    rotation: str,
) -> Tuple[Image.Image, List[Optional[List[float]]]]:
    width, height = image.size

    if rotation == "none":
        return image, boxes

    transformed = copy.deepcopy(boxes)

    if rotation == "cw":
        out = image.rotate(-90, expand=True)
        for idx, box in enumerate(transformed):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            transformed[idx] = [height - y2, x1, height - y1, x2]
    elif rotation == "ccw":
        out = image.rotate(90, expand=True)
        for idx, box in enumerate(transformed):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            transformed[idx] = [y1, width - x2, y2, width - x1]
    elif rotation == "180":
        out = image.rotate(180, expand=True)
        for idx, box in enumerate(transformed):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            transformed[idx] = [width - x2, height - y2, width - x1, height - y1]
    else:
        raise ValueError(f"Rotasi tidak dikenal: {rotation}")

    return out, transformed


def transform_sample(
    image: Image.Image,
    annotations: List[Dict],
    spec: AugmentSpec,
    min_visibility: float,
    min_box_size: float,
) -> Tuple[Image.Image, List[Dict]]:
    boxes_xyxy: List[List[float]] = []
    valid_annotations: List[Dict] = []

    width, height = image.size
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [float(value) for value in bbox]
        clipped = clip_box_xyxy([x, y, x + w, y + h], width, height)
        if clipped is None:
            continue
        boxes_xyxy.append(clipped)
        valid_annotations.append(ann)

    out_image, boxes = crop_resize_image_and_boxes(
        image=image,
        boxes=boxes_xyxy,
        spec=spec,
        min_visibility=min_visibility,
    )
    out_image, boxes = flip_image_and_boxes(out_image, boxes, spec.hflip, spec.vflip)
    out_image, boxes = rotate_image_and_boxes(out_image, boxes, spec.rotation)

    out_width, out_height = out_image.size
    out_annotations: List[Dict] = []
    for ann, box in zip(valid_annotations, boxes):
        if box is None:
            continue
        clipped = clip_box_xyxy(box, out_width, out_height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < min_box_size or box_h < min_box_size:
            continue

        new_ann = copy.deepcopy(ann)
        new_ann["bbox"] = [round(x1, 3), round(y1, 3), round(box_w, 3), round(box_h, 3)]
        new_ann["area"] = round(box_w * box_h, 3)
        new_ann["segmentation"] = []
        out_annotations.append(new_ann)

    return out_image, out_annotations


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=quality, optimize=True)
    else:
        image.save(path)


def output_file_name(original_file_name: str, aug_index: int, spec: AugmentSpec) -> str:
    original = Path(original_file_name)
    suffix = original.suffix if original.suffix.lower() in IMAGE_EXTENSIONS else ".jpg"
    token = "_".join(spec.tokens())
    return f"{original.stem}_aug{aug_index:03d}_{token}{suffix}"


def next_ids(coco: Dict) -> Tuple[int, int]:
    image_ids = [int(image.get("id", -1)) for image in coco.get("images", [])]
    ann_ids = [int(ann.get("id", -1)) for ann in coco.get("annotations", [])]
    return (max(image_ids, default=-1) + 1, max(ann_ids, default=-1) + 1)


def copy_original_sample(
    image_info: Dict,
    annotations: List[Dict],
    source_image_path: Path,
    output_image_dir: Path,
    next_image_id: int,
    next_ann_id: int,
) -> Tuple[Dict, List[Dict], int]:
    shutil.copy2(source_image_path, output_image_dir / image_info["file_name"])
    new_image = copy.deepcopy(image_info)
    new_image["id"] = next_image_id

    new_annotations: List[Dict] = []
    for ann in annotations:
        copied = copy.deepcopy(ann)
        copied["id"] = next_ann_id
        copied["image_id"] = next_image_id
        next_ann_id += 1
        new_annotations.append(copied)

    return new_image, new_annotations, next_ann_id


def category_counter(annotations: Iterable[Dict]) -> Counter:
    return Counter(int(ann["category_id"]) for ann in annotations)


def normalize_class_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def category_lookup(coco: Dict) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for category in coco["categories"]:
        category_id = int(category["id"])
        lookup[str(category_id)] = category_id
        lookup[normalize_class_key(str(category["name"]))] = category_id
    return lookup


def parse_category_values(
    raw_value: Optional[str],
    coco: Dict,
    value_name: str,
    cast_type,
) -> Dict[int, float]:
    if not raw_value:
        return {}

    lookup = category_lookup(coco)
    parsed: Dict[int, float] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Format {value_name} salah pada '{item}'. Pakai format kelas=nilai."
            )
        key, value = item.split("=", 1)
        normalized_key = normalize_class_key(key)
        if normalized_key not in lookup:
            known = ", ".join(category["name"] for category in coco["categories"])
            raise ValueError(f"Kelas tidak dikenal '{key}'. Kelas tersedia: {known}")
        parsed[lookup[normalized_key]] = cast_type(value)
    return parsed


def image_annotation_counts(annotations: Iterable[Dict]) -> Counter:
    return Counter(int(ann["category_id"]) for ann in annotations if ann.get("bbox"))


def planned_repeats_for_image(
    annotations: List[Dict],
    class_multipliers: Dict[int, float],
    default_multiplier: int,
    rng: random.Random,
) -> int:
    if not class_multipliers:
        return default_multiplier

    counts = image_annotation_counts(annotations)
    if not counts:
        return 0

    repeat_weight = max(class_multipliers.get(category_id, 0.0) for category_id in counts)
    if repeat_weight <= 0:
        return 0

    whole = int(repeat_weight)
    fraction = repeat_weight - whole
    return whole + (1 if rng.random() < fraction else 0)


def build_output_coco(coco: Dict, description_suffix: str) -> Dict:
    new_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": copy.deepcopy(coco.get("categories", [])),
        "images": [],
        "annotations": [],
    }
    if "info" in new_coco:
        new_coco["info"]["description"] = (
            str(new_coco["info"].get("description", ""))
            + f" | {description_suffix}"
        ).strip()
    return new_coco


def add_augmented_sample(
    *,
    new_coco: Dict,
    image_info: Dict,
    aug_image: Image.Image,
    aug_annotations: List[Dict],
    aug_file_name: str,
    output_image_dir: Path,
    jpeg_quality: int,
    next_image_id: int,
    next_ann_id: int,
    spec: AugmentSpec,
) -> Tuple[int, int]:
    save_image(aug_image, output_image_dir / aug_file_name, jpeg_quality)

    aug_image_info = copy.deepcopy(image_info)
    aug_image_info["id"] = next_image_id
    aug_image_info["file_name"] = aug_file_name
    aug_image_info["width"], aug_image_info["height"] = aug_image.size
    extra = copy.deepcopy(aug_image_info.get("extra", {}))
    extra["source_file_name"] = image_info["file_name"]
    extra["augmentation"] = {
        "hflip": spec.hflip,
        "vflip": spec.vflip,
        "rotation": spec.rotation,
        "crop_zoom": round(spec.zoom, 6),
    }
    aug_image_info["extra"] = extra

    for aug_ann in aug_annotations:
        aug_ann["id"] = next_ann_id
        aug_ann["image_id"] = next_image_id
        next_ann_id += 1

    new_coco["images"].append(aug_image_info)
    new_coco["annotations"].extend(aug_annotations)
    return next_image_id + 1, next_ann_id


def write_output_json(new_coco: Dict, annotation_out_dir: Path) -> Path:
    output_json = annotation_out_dir / "instances_val2017.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(new_coco, handle, ensure_ascii=False)
    return output_json


def print_summary(
    *,
    args: argparse.Namespace,
    image_out_dir: Path,
    output_json: Path,
    new_coco: Dict,
    missing_images: int,
    skipped_empty_aug: int,
) -> None:
    class_names = {int(cat["id"]): cat["name"] for cat in new_coco["categories"]}
    counts = category_counter(new_coco["annotations"])
    print("\nSelesai.")
    print(f"Output root      : {args.output_root}")
    print(f"Output images    : {image_out_dir}")
    print(f"Output JSON      : {output_json}")
    print(f"Images written   : {len(new_coco['images'])}")
    print(f"Annotations      : {len(new_coco['annotations'])}")
    print(f"Missing images   : {missing_images}")
    print(f"Skipped aug empty: {skipped_empty_aug}")
    print("BBox per class:")
    for category_id in sorted(class_names):
        print(f"  {class_names[category_id]}: {counts.get(category_id, 0)}")


def valid_source_images(
    images: List[Dict],
    ann_by_image: Dict[int, List[Dict]],
    input_images: Path,
) -> Tuple[List[Dict], int]:
    valid_images: List[Dict] = []
    missing_images = 0
    for image_info in images:
        source_image_path = input_images / image_info["file_name"]
        if not source_image_path.exists():
            missing_images += 1
            print(f"[skip] Gambar tidak ditemukan: {source_image_path}")
            continue
        if ann_by_image.get(int(image_info["id"]), []):
            valid_images.append(image_info)
    return valid_images, missing_images


def target_counts_from_args(args: argparse.Namespace, coco: Dict) -> Dict[int, int]:
    targets = {
        int(category["id"]): int(args.balance_to)
        for category in coco["categories"]
        if args.balance_to is not None
    }
    custom_targets = parse_category_values(
        args.balance_targets,
        coco,
        "--balance-targets",
        int,
    )
    targets.update({category_id: int(value) for category_id, value in custom_targets.items()})
    if not targets:
        raise ValueError("Mode balance membutuhkan --balance-to atau --balance-targets.")
    return targets


def print_balance_plan(coco: Dict, original_counts: Counter, targets: Dict[int, int]) -> None:
    class_names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    print("Rencana balance bbox:")
    for category_id in sorted(targets):
        current = original_counts.get(category_id, 0)
        target = targets[category_id]
        if current <= 0:
            total_multiplier = 0.0
            extra_multiplier = 0.0
        else:
            total_multiplier = target / current
            extra_multiplier = max(0, target - current) / current
        print(
            f"  {class_names.get(category_id, category_id)}: "
            f"{current} -> {target} | total x{total_multiplier:.4f} | "
            f"augment tambahan x{extra_multiplier:.4f}"
        )


def run_multiplier_mode(
    *,
    args: argparse.Namespace,
    rng: random.Random,
    coco: Dict,
    image_out_dir: Path,
    annotation_out_dir: Path,
    ann_by_image: Dict[int, List[Dict]],
    images: List[Dict],
) -> None:
    class_multipliers = parse_category_values(
        args.class_multipliers,
        coco,
        "--class-multipliers",
        float,
    )
    new_coco = build_output_coco(coco, f"offline val augmentation {args.multiplier}x")
    next_image_id, next_ann_id = next_ids(coco)
    missing_images = 0
    skipped_empty_aug = 0
    source_aug_counts: Counter = Counter()

    for source_index, image_info in enumerate(images, start=1):
        file_name = image_info["file_name"]
        source_image_path = args.input_images / file_name
        if not source_image_path.exists():
            missing_images += 1
            print(f"[skip] Gambar tidak ditemukan: {source_image_path}")
            continue

        annotations = ann_by_image.get(int(image_info["id"]), [])

        if args.include_original:
            copied_image, copied_annotations, next_ann_id = copy_original_sample(
                image_info=image_info,
                annotations=annotations,
                source_image_path=source_image_path,
                output_image_dir=image_out_dir,
                next_image_id=next_image_id,
                next_ann_id=next_ann_id,
            )
            new_coco["images"].append(copied_image)
            new_coco["annotations"].extend(copied_annotations)
            next_image_id += 1

        repeats = planned_repeats_for_image(
            annotations=annotations,
            class_multipliers=class_multipliers,
            default_multiplier=args.multiplier,
            rng=rng,
        )

        with Image.open(source_image_path) as raw_image:
            image = raw_image.convert("RGB")

            for _ in range(repeats):
                source_aug_counts[file_name] += 1
                aug_index = source_aug_counts[file_name]
                spec = choose_spec(rng, args.max_zoom)
                aug_image, aug_annotations = transform_sample(
                    image=image,
                    annotations=annotations,
                    spec=spec,
                    min_visibility=args.min_visibility,
                    min_box_size=args.min_box_size,
                )

                if annotations and not aug_annotations:
                    skipped_empty_aug += 1
                    continue

                aug_file_name = output_file_name(file_name, aug_index, spec)
                next_image_id, next_ann_id = add_augmented_sample(
                    new_coco=new_coco,
                    image_info=image_info,
                    aug_image=aug_image,
                    aug_annotations=aug_annotations,
                    aug_file_name=aug_file_name,
                    output_image_dir=image_out_dir,
                    jpeg_quality=args.jpeg_quality,
                    next_image_id=next_image_id,
                    next_ann_id=next_ann_id,
                    spec=spec,
                )

        if source_index % 25 == 0 or source_index == len(images):
            print(f"Processed {source_index}/{len(images)} images...")

    output_json = write_output_json(new_coco, annotation_out_dir)
    print_summary(
        args=args,
        image_out_dir=image_out_dir,
        output_json=output_json,
        new_coco=new_coco,
        missing_images=missing_images,
        skipped_empty_aug=skipped_empty_aug,
    )


def run_balance_mode(
    *,
    args: argparse.Namespace,
    rng: random.Random,
    coco: Dict,
    image_out_dir: Path,
    annotation_out_dir: Path,
    ann_by_image: Dict[int, List[Dict]],
    images: List[Dict],
) -> None:
    targets = target_counts_from_args(args, coco)
    original_annotations = [
        ann
        for image_info in images
        for ann in ann_by_image.get(int(image_info["id"]), [])
    ]
    original_counts = category_counter(original_annotations)
    print_balance_plan(coco, original_counts, targets)

    for category_id, target in targets.items():
        current = original_counts.get(category_id, 0)
        if not args.balance_without_original and current > target:
            raise ValueError(
                f"Target kelas id {category_id} ({target}) lebih kecil dari data asli ({current}). "
                "Naikkan target atau gunakan --balance-without-original."
            )

    new_coco = build_output_coco(coco, "offline val class-balanced augmentation")
    next_image_id, next_ann_id = next_ids(coco)
    missing_images = 0
    skipped_empty_aug = 0
    source_aug_counts: Counter = Counter()

    if not args.balance_without_original:
        for image_info in images:
            source_image_path = args.input_images / image_info["file_name"]
            if not source_image_path.exists():
                missing_images += 1
                print(f"[skip] Gambar tidak ditemukan: {source_image_path}")
                continue
            copied_image, copied_annotations, next_ann_id = copy_original_sample(
                image_info=image_info,
                annotations=ann_by_image.get(int(image_info["id"]), []),
                source_image_path=source_image_path,
                output_image_dir=image_out_dir,
                next_image_id=next_image_id,
                next_ann_id=next_ann_id,
            )
            new_coco["images"].append(copied_image)
            new_coco["annotations"].extend(copied_annotations)
            next_image_id += 1

    current_counts = category_counter(new_coco["annotations"])
    candidates, extra_missing = valid_source_images(images, ann_by_image, args.input_images)
    if args.balance_without_original:
        missing_images += extra_missing

    if not candidates:
        raise ValueError("Tidak ada gambar kandidat valid untuk augmentasi balance.")

    attempts = 0
    accepted = 0
    while attempts < args.balance_max_attempts:
        remaining = {
            category_id: target - current_counts.get(category_id, 0)
            for category_id, target in targets.items()
        }
        needed_categories = [category_id for category_id, value in remaining.items() if value > 0]
        if not needed_categories:
            break

        selected_category = rng.choices(
            needed_categories,
            weights=[remaining[category_id] for category_id in needed_categories],
            k=1,
        )[0]
        selected_candidates = [
            image_info
            for image_info in candidates
            if image_annotation_counts(ann_by_image.get(int(image_info["id"]), [])).get(
                selected_category, 0
            )
            > 0
        ]
        if not selected_candidates:
            raise ValueError(f"Tidak ada kandidat gambar untuk category_id={selected_category}.")

        attempts += 1
        image_info = rng.choice(selected_candidates)
        annotations = ann_by_image.get(int(image_info["id"]), [])
        source_image_path = args.input_images / image_info["file_name"]

        with Image.open(source_image_path) as raw_image:
            image = raw_image.convert("RGB")
            spec = choose_spec(rng, args.max_zoom)
            aug_image, aug_annotations = transform_sample(
                image=image,
                annotations=annotations,
                spec=spec,
                min_visibility=args.min_visibility,
                min_box_size=args.min_box_size,
            )

        if not aug_annotations:
            skipped_empty_aug += 1
            continue

        aug_counts = category_counter(aug_annotations)
        if aug_counts.get(selected_category, 0) <= 0:
            skipped_empty_aug += 1
            continue

        overshoots = False
        for category_id, count in aug_counts.items():
            target = targets.get(category_id)
            if target is not None and current_counts.get(category_id, 0) + count > target:
                overshoots = True
                break
        if overshoots:
            continue

        file_name = image_info["file_name"]
        source_aug_counts[file_name] += 1
        aug_file_name = output_file_name(file_name, source_aug_counts[file_name], spec)
        next_image_id, next_ann_id = add_augmented_sample(
            new_coco=new_coco,
            image_info=image_info,
            aug_image=aug_image,
            aug_annotations=aug_annotations,
            aug_file_name=aug_file_name,
            output_image_dir=image_out_dir,
            jpeg_quality=args.jpeg_quality,
            next_image_id=next_image_id,
            next_ann_id=next_ann_id,
            spec=spec,
        )
        current_counts.update(aug_counts)
        accepted += 1

        if accepted % 25 == 0:
            remaining_text = ", ".join(
                f"{category_id}:{targets[category_id] - current_counts.get(category_id, 0)}"
                for category_id in sorted(targets)
            )
            print(f"Accepted {accepted} augmented images | remaining {remaining_text}")

    remaining = {
        category_id: target - current_counts.get(category_id, 0)
        for category_id, target in targets.items()
    }
    if any(value > 0 for value in remaining.values()):
        print(
            "\nPeringatan: target belum terpenuhi sempurna. "
            "Coba naikkan --balance-max-attempts, turunkan --min-visibility, "
            "atau gunakan target yang sedikit lebih tinggi."
        )
        print(f"Sisa kebutuhan: {remaining}")

    output_json = write_output_json(new_coco, annotation_out_dir)
    print_summary(
        args=args,
        image_out_dir=image_out_dir,
        output_json=output_json,
        new_coco=new_coco,
        missing_images=missing_images,
        skipped_empty_aug=skipped_empty_aug,
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    coco = load_coco(args.input_json)
    image_out_dir, annotation_out_dir = ensure_output_dirs(args.output_root, args.overwrite)
    ann_by_image = annotations_by_image(coco)

    images = list(coco["images"])
    if args.limit is not None:
        images = images[: args.limit]

    if args.balance_to is not None or args.balance_targets is not None:
        run_balance_mode(
            args=args,
            rng=rng,
            coco=coco,
            image_out_dir=image_out_dir,
            annotation_out_dir=annotation_out_dir,
            ann_by_image=ann_by_image,
            images=images,
        )
    else:
        run_multiplier_mode(
            args=args,
            rng=rng,
            coco=coco,
            image_out_dir=image_out_dir,
            annotation_out_dir=annotation_out_dir,
            ann_by_image=ann_by_image,
            images=images,
        )


if __name__ == "__main__":
    main()
