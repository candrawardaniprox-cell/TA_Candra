"""
generate_noisy_test.py — Tambahkan noise pada data test untuk eksperimen robustness.

Script ini membaca gambar test asli dari data/scenario_100/test2017/,
menambahkan noise (Gaussian, Salt-and-Pepper, Poisson) pada 3 level
intensitas (rendah, sedang, tinggi), dan menyimpan hasilnya ke folder
terpisah. File anotasi COCO di-copy apa adanya.

Setiap konfigurasi juga menghitung PSNR (Peak Signal-to-Noise Ratio)
agar perbandingan antar jenis noise bisa dilakukan secara adil.

TIDAK mengubah dataset asli sama sekali.

Cara pakai:
    python generate_noisy_test.py

Output (9 folder):
    data/noisy_test_gaussian_rendah/
    data/noisy_test_gaussian_sedang/
    data/noisy_test_gaussian_tinggi/
    data/noisy_test_salt_pepper_rendah/
    data/noisy_test_salt_pepper_sedang/
    data/noisy_test_salt_pepper_tinggi/
    data/noisy_test_poisson_rendah/
    data/noisy_test_poisson_sedang/
    data/noisy_test_poisson_tinggi/
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

# ======================== KONFIGURASI NOISE (3 LEVEL) ========================
# Setiap noise memiliki 3 tingkat intensitas: rendah, sedang, tinggi.
#
# PSNR (Peak Signal-to-Noise Ratio) dihitung otomatis untuk setiap level.
# Semakin rendah PSNR = semakin banyak noise = kualitas gambar semakin buruk.
#   - PSNR > 30 dB  : noise hampir tidak terlihat
#   - PSNR 25-30 dB : noise ringan
#   - PSNR 20-25 dB : noise sedang (terlihat jelas)
#   - PSNR < 20 dB  : noise berat (sangat mengganggu)

NOISE_LEVELS = {
    "gaussian": {
        "rendah":  {"mean": 0, "sigma": 10},   # noise ringan
        "sedang":  {"mean": 0, "sigma": 25},   # noise sedang
        "tinggi":  {"mean": 0, "sigma": 50},   # noise berat
    },
    "salt_pepper": {
        "rendah":  {"amount": 0.01},   # 1% piksel terdampak
        "sedang":  {"amount": 0.05},   # 5% piksel terdampak
        "tinggi":  {"amount": 0.10},   # 10% piksel terdampak
    },
    "poisson": {
        "rendah":  {"scale_factor": 60.0},  # noise ringan (scale besar = noise kecil)
        "sedang":  {"scale_factor": 25.0},  # noise sedang
        "tinggi":  {"scale_factor": 8.0},   # noise berat (scale kecil = noise besar)
    },
}

# Label tampilan untuk output
LEVEL_LABELS = {
    "rendah": "Rendah (Low)",
    "sedang": "Sedang (Medium)",
    "tinggi": "Tinggi (High)",
}

# ======================== PATH ========================
DATA_ROOT = Path("data") / "scenario_100"
SRC_IMAGE_DIR = DATA_ROOT / "test2017"
SRC_ANNOTATION_DIR = DATA_ROOT / "annotations_coco"
SRC_ANNOTATION_FILE = SRC_ANNOTATION_DIR / "instances_test2017.json"
OUTPUT_BASE = Path("data")


# ======================== PSNR ========================
def calculate_psnr(original: np.ndarray, noisy: np.ndarray) -> float:
    """
    Hitung PSNR (Peak Signal-to-Noise Ratio) antara gambar asli dan noisy.

    PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))

    Semakin tinggi PSNR = semakin mirip dengan aslinya (noise sedikit).
    Semakin rendah PSNR = semakin berbeda (noise banyak).
    """
    mse = np.mean((original.astype(np.float64) - noisy.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')  # gambar identik
    max_pixel = 255.0
    psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# ======================== FUNGSI NOISE ========================
def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Tambahkan Gaussian noise (additive) ke gambar.

    Gaussian noise mensimulasikan thermal noise pada sensor kamera,
    terutama saat sensor kepanasan di lingkungan pertanian terbuka.
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float64)
    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.03) -> np.ndarray:
    """
    Tambahkan Salt-and-Pepper noise ke gambar.

    Mensimulasikan debu, kotoran di lensa, atau dead pixel pada sensor
    kamera yang dipasang di lahan pertanian.
    """
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_affected = int(total_pixels * amount)

    # Salt (putih)
    salt_coords = [
        np.random.randint(0, dim, num_affected // 2)
        for dim in image.shape[:2]
    ]
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Pepper (hitam)
    pepper_coords = [
        np.random.randint(0, dim, num_affected // 2)
        for dim in image.shape[:2]
    ]
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy


def add_poisson_noise(image: np.ndarray, scale_factor: float = 25.0) -> np.ndarray:
    """
    Tambahkan Poisson (shot) noise ke gambar.

    Mensimulasikan noise akibat pencahayaan rendah atau kondisi
    di bawah kanopi tanaman dengan cahaya terbatas.
    Noise bergantung pada intensitas piksel asli.

    scale_factor: Semakin besar = noise semakin kecil (halus).
                  Semakin kecil = noise semakin besar (kasar).
    """
    image_scaled = image.astype(np.float64) / 255.0 * scale_factor
    image_scaled = np.clip(image_scaled, 0, scale_factor)
    noisy = np.random.poisson(image_scaled).astype(np.float64)
    noisy = noisy / scale_factor * 255.0
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def apply_noise(image: np.ndarray, noise_type: str, params: dict) -> np.ndarray:
    """Terapkan jenis noise yang dipilih dengan parameter tertentu."""
    if noise_type == "gaussian":
        return add_gaussian_noise(image, params["mean"], params["sigma"])
    elif noise_type == "salt_pepper":
        return add_salt_pepper_noise(image, params["amount"])
    elif noise_type == "poisson":
        return add_poisson_noise(image, params["scale_factor"])
    else:
        raise ValueError(f"Jenis noise tidak dikenal: {noise_type}")


def get_param_display(noise_type: str, params: dict) -> str:
    """Format parameter noise untuk tampilan."""
    if noise_type == "gaussian":
        return f"sigma={params['sigma']}"
    elif noise_type == "salt_pepper":
        return f"amount={params['amount']*100:.0f}%"
    elif noise_type == "poisson":
        return f"scale={params['scale_factor']:.0f}"
    return str(params)


# ======================== MAIN ========================
def main():
    # Validasi path sumber
    if not SRC_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Folder gambar test tidak ditemukan: {SRC_IMAGE_DIR}")
    if not SRC_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"File anotasi test tidak ditemukan: {SRC_ANNOTATION_FILE}")

    # Ambil daftar gambar
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = sorted([
        f for f in SRC_IMAGE_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise RuntimeError(f"Tidak ada gambar ditemukan di: {SRC_IMAGE_DIR}")

    print(f"Ditemukan {len(image_files)} gambar test di: {SRC_IMAGE_DIR}")
    print(f"File anotasi: {SRC_ANNOTATION_FILE}")
    print()

    # Simpan ringkasan PSNR untuk tabel akhir
    psnr_summary = []

    for noise_type, levels in NOISE_LEVELS.items():
        for level_name, params in levels.items():
            folder_name = f"noisy_test_{noise_type}_{level_name}"
            dst_image_dir = OUTPUT_BASE / folder_name / "test2017"
            dst_annotation_dir = OUTPUT_BASE / folder_name / "annotations_coco"

            # Buat folder output
            dst_image_dir.mkdir(parents=True, exist_ok=True)
            dst_annotation_dir.mkdir(parents=True, exist_ok=True)

            # Copy anotasi (tidak diubah)
            dst_annotation_file = dst_annotation_dir / "instances_test2017.json"
            shutil.copy2(SRC_ANNOTATION_FILE, dst_annotation_file)

            param_str = get_param_display(noise_type, params)
            level_label = LEVEL_LABELS[level_name]
            print(f"[{noise_type.upper()} - {level_label}] ({param_str})")
            print(f"  Memproses {len(image_files)} gambar -> {dst_image_dir}")

            success_count = 0
            fail_count = 0
            psnr_values = []

            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  [!] Gagal membaca: {img_path.name}")
                        fail_count += 1
                        continue

                    # Terapkan noise
                    noisy_img = apply_noise(img, noise_type, params)

                    # Hitung PSNR
                    psnr = calculate_psnr(img, noisy_img)
                    psnr_values.append(psnr)

                    # Simpan
                    dst_path = dst_image_dir / img_path.name
                    cv2.imwrite(str(dst_path), noisy_img)
                    success_count += 1

                except Exception as e:
                    print(f"  [X] Error pada {img_path.name}: {e}")
                    fail_count += 1

            avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
            min_psnr = np.min(psnr_values) if psnr_values else 0.0
            max_psnr = np.max(psnr_values) if psnr_values else 0.0

            print(f"  [OK] Selesai: {success_count} berhasil, {fail_count} gagal")
            print(f"  PSNR: avg={avg_psnr:.2f} dB | min={min_psnr:.2f} dB | max={max_psnr:.2f} dB")
            print()

            psnr_summary.append({
                "noise_type": noise_type,
                "level": level_name,
                "level_label": level_label,
                "params": param_str,
                "avg_psnr": avg_psnr,
                "min_psnr": min_psnr,
                "max_psnr": max_psnr,
                "folder": folder_name,
                "count": success_count,
            })

    # ======================== TABEL RINGKASAN ========================
    sep = "=" * 85
    print(sep)
    print("  RINGKASAN PSNR — PERBANDINGAN INTENSITAS NOISE")
    print(sep)
    print(f"  {'Jenis Noise':<16} {'Level':<18} {'Parameter':<14} {'PSNR Rata2':>12} {'PSNR Min':>10} {'PSNR Max':>10}")
    print(f"  {'-'*16} {'-'*18} {'-'*14} {'-'*12} {'-'*10} {'-'*10}")

    for entry in psnr_summary:
        print(
            f"  {entry['noise_type']:<16} "
            f"{entry['level_label']:<18} "
            f"{entry['params']:<14} "
            f"{entry['avg_psnr']:>10.2f} dB "
            f"{entry['min_psnr']:>8.2f} dB "
            f"{entry['max_psnr']:>8.2f} dB"
        )

    print(sep)
    print()
    print("  CATATAN PSNR:")
    print("  - PSNR > 30 dB  : noise hampir tidak terlihat")
    print("  - PSNR 25-30 dB : noise ringan")
    print("  - PSNR 20-25 dB : noise sedang (terlihat jelas)")
    print("  - PSNR < 20 dB  : noise berat (sangat mengganggu)")
    print()
    print("  Bandingkan PSNR rata-rata antar noise type pada level yang sama")
    print("  untuk melihat apakah intensitasnya sebanding.")
    print(sep)
    print()
    print(f"  Total: {len(psnr_summary)} konfigurasi noise berhasil digenerate.")
    print(f"  Selanjutnya jalankan: python test_noisy.py")
    print(sep)


if __name__ == "__main__":
    main()
