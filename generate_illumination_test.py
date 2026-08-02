"""
generate_illumination_test.py — Simulasi kondisi pencahayaan pada data test.

Script ini mengubah gambar test asli menggunakan Gamma Correction
untuk mensimulasikan 3 kondisi pencahayaan:
1. Terang (Siang Terik) -> Gamma = 2.0
2. Normal (Baseline)    -> Gamma = 1.0 (Tidak berubah)
3. Gelap (Malam/Mendung)-> Gamma = 0.3

Hasil disimpan di 3 folder terpisah, beserta copy file anotasi.
TIDAK mengubah dataset asli sama sekali.

Output (3 folder):
    data/illumination_test_terang/
    data/illumination_test_normal/
    data/illumination_test_gelap/
"""
from __future__ import annotations

import shutil
from pathlib import Path
import cv2
import numpy as np

# ======================== KONFIGURASI ========================
ILLUMINATION_LEVELS = {
    "terang": {"gamma": 2.0, "label": "Siang Terik (Bright)"},
    "normal": {"gamma": 1.0, "label": "Baseline (Normal)"},
    "gelap":  {"gamma": 0.3, "label": "Mendung/Malam (Dark)"},
}

DATA_ROOT = Path("data") / "scenario_100"
SRC_IMAGE_DIR = DATA_ROOT / "test2017"
SRC_ANNOTATION_DIR = DATA_ROOT / "annotations_coco"
SRC_ANNOTATION_FILE = SRC_ANNOTATION_DIR / "instances_test2017.json"
OUTPUT_BASE = Path("data")

# ======================== FUNGSI GAMMA ========================
def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Terapkan gamma correction pada gambar.
    gamma < 1.0: Lebih terang
    gamma > 1.0: Lebih gelap
    gamma == 1.0: Sama persis
    """
    if gamma == 1.0:
        return image.copy()
    
    # Buat lookup table (LUT) agar pemrosesan lebih cepat untuk setiap piksel
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

# ======================== MAIN ========================
def main():
    if not SRC_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Folder gambar test tidak ditemukan: {SRC_IMAGE_DIR}")
    if not SRC_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"File anotasi test tidak ditemukan: {SRC_ANNOTATION_FILE}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = sorted([
        f for f in SRC_IMAGE_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise RuntimeError(f"Tidak ada gambar ditemukan di: {SRC_IMAGE_DIR}")

    print(f"Ditemukan {len(image_files)} gambar test di: {SRC_IMAGE_DIR}")
    print()

    sep = "=" * 70
    
    for level_key, config in ILLUMINATION_LEVELS.items():
        gamma = config["gamma"]
        label = config["label"]
        folder_name = f"illumination_test_{level_key}"
        
        dst_image_dir = OUTPUT_BASE / folder_name / "test2017"
        dst_annotation_dir = OUTPUT_BASE / folder_name / "annotations_coco"

        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_annotation_dir.mkdir(parents=True, exist_ok=True)

        dst_annotation_file = dst_annotation_dir / "instances_test2017.json"
        shutil.copy2(SRC_ANNOTATION_FILE, dst_annotation_file)

        print(f"[{label.upper()}] Gamma = {gamma}")
        print(f"  Memproses {len(image_files)} gambar -> {dst_image_dir}")

        success_count = 0
        fail_count = 0

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [!] Gagal membaca: {img_path.name}")
                    fail_count += 1
                    continue

                processed_img = apply_gamma_correction(img, gamma)

                dst_path = dst_image_dir / img_path.name
                cv2.imwrite(str(dst_path), processed_img)
                success_count += 1

            except Exception as e:
                print(f"  [X] Error pada {img_path.name}: {e}")
                fail_count += 1

        print(f"  [OK] Selesai: {success_count} berhasil, {fail_count} gagal")
        print(sep)

    print("\nSemua gambar pencahayaan berhasil digenerate!")
    print("Selanjutnya jalankan: python test_illumination.py")

if __name__ == "__main__":
    main()
