"""Cek karakteristik citra dataset untuk analisis kesesuaian CLAHE & median filter."""
from __future__ import annotations
import cv2
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

data_dir = Path("data/Dataset2026")
coco = json.load(open(data_dir / "_annotations.coco.json"))
cats = {c["id"]: c["name"] for c in coco["categories"]}

ann_by_img = defaultdict(list)
for a in coco["annotations"]:
    ann_by_img[a["image_id"]].append(a)

# Ambil 30 gambar acak untuk analisis
import random
random.seed(42)
sample_imgs = random.sample(coco["images"], min(30, len(coco["images"])))

brightness_vals = []
contrast_vals = []
noise_vals = []
resolutions = []

for img_info in sample_imgs:
    path = data_dir / img_info["file_name"]
    img = cv2.imread(str(path))
    if img is None:
        continue
    
    h, w = img.shape[:2]
    resolutions.append((w, h))
    
    # Konversi ke grayscale untuk analisis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Brightness (rata-rata intensitas)
    brightness_vals.append(np.mean(gray))
    
    # Kontras (standar deviasi intensitas)
    contrast_vals.append(np.std(gray))
    
    # Estimasi noise (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_vals.append(laplacian.var())

print("=" * 60)
print("  ANALISIS KARAKTERISTIK CITRA DATASET")
print("=" * 60)

print(f"\n  Jumlah sampel dianalisis: {len(brightness_vals)}")

# Resolusi
widths = [r[0] for r in resolutions]
heights = [r[1] for r in resolutions]
print(f"\n  RESOLUSI:")
print(f"    Lebar  : min={min(widths)}, max={max(widths)}, rata-rata={np.mean(widths):.0f}")
print(f"    Tinggi : min={min(heights)}, max={max(heights)}, rata-rata={np.mean(heights):.0f}")

# Brightness
print(f"\n  BRIGHTNESS (0=gelap, 255=terang):")
print(f"    Min  : {min(brightness_vals):.1f}")
print(f"    Max  : {max(brightness_vals):.1f}")
print(f"    Mean : {np.mean(brightness_vals):.1f}")
print(f"    Std  : {np.std(brightness_vals):.1f}")
low_brightness = sum(1 for b in brightness_vals if b < 80)
high_brightness = sum(1 for b in brightness_vals if b > 200)
print(f"    Gelap (<80)  : {low_brightness}/{len(brightness_vals)} gambar")
print(f"    Terang (>200): {high_brightness}/{len(brightness_vals)} gambar")

# Kontras
print(f"\n  KONTRAS (standar deviasi intensitas):")
print(f"    Min  : {min(contrast_vals):.1f}")
print(f"    Max  : {max(contrast_vals):.1f}")
print(f"    Mean : {np.mean(contrast_vals):.1f}")
print(f"    Std  : {np.std(contrast_vals):.1f}")
low_contrast = sum(1 for c in contrast_vals if c < 40)
print(f"    Kontras rendah (<40): {low_contrast}/{len(contrast_vals)} gambar")

# Noise 
print(f"\n  NOISE LEVEL (Laplacian variance, makin tinggi = lebih tajam/noisy):")
print(f"    Min  : {min(noise_vals):.1f}")
print(f"    Max  : {max(noise_vals):.1f}")
print(f"    Mean : {np.mean(noise_vals):.1f}")
print(f"    Std  : {np.std(noise_vals):.1f}")

# Histogram analysis - cek apakah distribusi intensitas merata
print(f"\n  DISTRIBUSI HISTOGRAM:")
all_hist_std = []
for img_info in sample_imgs[:10]:
    path = data_dir / img_info["file_name"]
    img = cv2.imread(str(path))
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    # Entropy sebagai ukuran pemerataan
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    all_hist_std.append(entropy)

print(f"    Entropy histogram (max=8.0 jika merata sempurna):")
print(f"    Mean entropy: {np.mean(all_hist_std):.2f}")
print(f"    Min entropy : {min(all_hist_std):.2f}")

# Kesimpulan
print(f"\n{'=' * 60}")
print("  KESIMPULAN KESESUAIAN PREPROCESSING")
print(f"{'=' * 60}")

mean_brightness = np.mean(brightness_vals)
mean_contrast = np.mean(contrast_vals)
mean_noise = np.mean(noise_vals)
mean_entropy = np.mean(all_hist_std)

print(f"\n  CLAHE:")
if mean_contrast < 50 or low_contrast > len(brightness_vals) * 0.3:
    print(f"    ✓ SANGAT COCOK - Banyak gambar kontras rendah ({low_contrast}/{len(brightness_vals)})")
elif mean_entropy < 6.5:
    print(f"    ✓ COCOK - Distribusi histogram belum merata (entropy={mean_entropy:.2f})")
else:
    print(f"    △ OPSIONAL - Kontras sudah cukup baik (mean={mean_contrast:.1f}), tapi CLAHE tetap bisa membantu")

print(f"\n  MEDIAN FILTER:")
if mean_noise > 1000:
    print(f"    ✓ COCOK - Level noise tinggi (Laplacian var={mean_noise:.0f})")
elif mean_noise > 300:
    print(f"    △ OPSIONAL - Level noise sedang (Laplacian var={mean_noise:.0f}), ksize=3 aman")
else:
    print(f"    ⚠ HATI-HATI - Level noise rendah (Laplacian var={mean_noise:.0f}), bisa blur detail tekstur")

print()
