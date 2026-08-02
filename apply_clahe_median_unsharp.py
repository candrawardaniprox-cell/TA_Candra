"""
apply_clahe_median_unsharp.py

Script untuk menerapkan 3 teknik Peningkatan Kualitas Citra ke SEMUA gambar
pada dataset yang sudah di-split (train/val/test).

Preprocessing yang diterapkan (berurutan):
  1. CLAHE             - Meningkatkan kontras secara adaptif dan merata.
  2. Median Filter (ksize=3) - Mereduksi noise halus tanpa merusak tepi objek.
  3. Unsharp Masking   - Mempertajam detail gambar setelah noise direduksi.

TANPA augmentasi apapun. Jumlah gambar dan bbox TIDAK berubah.
Anotasi (JSON) akan di-copy apa adanya tanpa perubahan.

Struktur folder input yang diharapkan:
  <input_dir>/
    annotations_coco/
      instances_train2017.json
      instances_val2017.json
      instances_test2017.json
    train2017/
      *.jpg
    val2017/
      *.jpg
    test2017/
      *.jpg

Output akan memiliki struktur yang sama persis, hanya gambar-gambarnya
yang telah diterapkan CLAHE + Median Filter + Unsharp Masking.

Cara pakai:
  python apply_clahe_median_unsharp.py --overwrite
  python apply_clahe_median_unsharp.py --input "data/coco copy" --overwrite
  python apply_clahe_median_unsharp.py --input "data/coco copy" --output "data/coco copy_clahe_median_unsharp" --overwrite
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ======================== KONSTANTA ========================
SEED = 42

# Subfolder yang berisi gambar
IMAGE_SUBFOLDERS = ["train2017", "val2017", "test2017"]

# Subfolder yang berisi anotasi (akan di-copy apa adanya)
ANNOTATION_SUBFOLDER = "annotations_coco"

# Ekstensi gambar yang diproses
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ======================== TEKNIK ENHANCEMENT ========================

def apply_clahe(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Terapkan CLAHE ke gambar.
    Mengubah ke ruang warna LAB, menerapkan CLAHE pada channel L (Lightness),
    lalu mengkonversi kembali ke BGR.

    Parameter CLAHE (clip_limit dan tile_size) sedikit di-random untuk
    memberikan variasi natural antar gambar.
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

    Median Filter efektif menghilangkan noise halus dari kamera
    sambil tetap mempertahankan tepi (edge) objek pada gambar.
    ksize=3 memberikan efek minimal yang aman tanpa merusak detail tekstur.
    """
    return cv2.medianBlur(image, ksize)


def apply_unsharp_masking(
    image: np.ndarray,
    kernel_size: tuple = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0,
) -> np.ndarray:
    """
    Terapkan Unsharp Masking untuk mempertajam gambar.

    Cara kerja:
      1. Buat versi blur dari gambar menggunakan GaussianBlur.
      2. Hitung selisih antara gambar asli dan blur (high-frequency detail).
      3. Tambahkan kembali selisih tersebut ke gambar asli dengan faktor 'amount'.

    Parameter:
      kernel_size : Ukuran kernel Gaussian (default (5,5))
      sigma       : Sigma Gaussian (default 1.0)
      amount      : Kekuatan penajaman (default 1.5, makin besar makin tajam)
      threshold   : Ambang batas minimum perbedaan pixel untuk di-sharpen (default 0)
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def apply_enhancement(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Pipeline enhancement lengkap (3 tahap):
      1. CLAHE            (peningkatan kontras adaptif)
      2. Median Filter    (pengurangan noise halus, ksize=3)
      3. Unsharp Masking  (penajaman detail gambar)
    """
    result = apply_clahe(image, rng)
    result = apply_median_filter(result, ksize=3)
    result = apply_unsharp_masking(result, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0)
    return result


# ======================== MAIN LOGIC ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terapkan CLAHE + Median Filter + Unsharp Masking ke seluruh gambar "
                    "dataset split (train/val/test). Tanpa augmentasi, hanya enhancement."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/coco copy"),
        help="Folder dataset split input (berisi train2017, val2017, test2017, annotations_coco)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Folder output. Default: <input>_clahe_median_unsharp",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Hapus folder output jika sudah ada",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(SEED)

    # Tentukan folder output
    if args.output is None:
        args.output = Path(str(args.input) + "_clahe_median_unsharp")

    print("=====================================================")
    print("  APPLY CLAHE + MEDIAN FILTER + UNSHARP MASKING")
    print("  (TANPA AUGMENTASI - ENHANCEMENT ONLY)")
    print("=====================================================")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print("  Pipeline:")
    print("    1. CLAHE           -> Peningkatan kontras adaptif")
    print("    2. Median Filter   -> Pengurangan noise (ksize=3)")
    print("    3. Unsharp Masking -> Penajaman detail gambar")
    print("=====================================================")

    # Validasi folder input
    if not args.input.exists():
        print(f"\n[X] Folder input tidak ditemukan: {args.input}")
        sys.exit(1)

    # Cek folder output
    if args.output.exists():
        if not args.overwrite:
            print(f"\n[X] Folder output {args.output} sudah ada. Gunakan --overwrite.")
            sys.exit(1)
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    # ---- 1. Copy Anotasi apa adanya ----
    ann_src = args.input / ANNOTATION_SUBFOLDER
    ann_dst = args.output / ANNOTATION_SUBFOLDER
    if ann_src.exists():
        shutil.copytree(str(ann_src), str(ann_dst))
        ann_files = list(ann_dst.glob("*.json"))
        print(f"\n[1/2] Anotasi di-copy apa adanya ({len(ann_files)} file JSON).")
    else:
        print(f"\n[!] Folder anotasi tidak ditemukan: {ann_src}")
        print("    Melanjutkan tanpa anotasi...")

    # ---- 2. Proses Gambar dengan CLAHE + Median Filter + Unsharp Masking ----
    print("\n[2/2] Menerapkan CLAHE + Median Filter + Unsharp Masking ke seluruh gambar...")

    total_processed = 0
    total_skipped = 0

    for subfolder_name in IMAGE_SUBFOLDERS:
        src_folder = args.input / subfolder_name
        dst_folder = args.output / subfolder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        if not src_folder.exists():
            print(f"  [!] Subfolder tidak ditemukan, dilewati: {subfolder_name}")
            continue

        # Ambil semua file gambar
        image_files = sorted([
            f for f in src_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        processed = 0
        skipped = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            # Terapkan CLAHE + Median Filter + Unsharp Masking
            img_enhanced = apply_enhancement(img, rng)

            # Simpan ke folder output dengan nama yang sama
            dst_path = dst_folder / img_path.name
            cv2.imwrite(str(dst_path), img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            processed += 1

        total_processed += processed
        total_skipped += skipped
        print(f"  [OK] {subfolder_name}: {processed} gambar diproses (CLAHE + Median + Unsharp)"
              + (f", {skipped} dilewati" if skipped else ""))

    # ---- Ringkasan ----
    print("\n=====================================================")
    print("  SELESAI - CLAHE + MEDIAN FILTER + UNSHARP MASKING")
    print("=====================================================")
    print(f"  Total gambar diproses : {total_processed}")
    if total_skipped:
        print(f"  Total gambar dilewati : {total_skipped}")
    print(f"  Folder output         : {args.output}")
    print("=====================================================")


if __name__ == "__main__":
    main()
