import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import torch
from piq import brisque

def calculate_brisque(img_bgr):
    """
    Menghitung skor BRISQUE dari citra BGR.
    Fungsinya sama dengan NIQE (No-Reference Metric).
    Makin kecil nilai BRISQUE = Kualitas citra makin baik secara natural.
    """
    # Ubah format ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Konversi ke format tensor PyTorch: (Batch, Channel, Height, Width), nilai [0, 1]
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Hitung skor BRISQUE
    with torch.no_grad():
        score = brisque(tensor, data_range=1.0)
    return score.item()

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def laplacian_sharpen(image):
    """Sharpen using Laplacian filter"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def high_pass_sharpen(image):
    """Sharpen using a strong high-pass filter"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"File {image_path} tidak ditemukan!")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Gagal membaca gambar. Pastikan format didukung (jpg/png).")
        return
        
    print(f"\nMemproses {os.path.basename(image_path)}...")
    
    # Menghitung BRISQUE Asli
    brisque_asli = calculate_brisque(img)
    print(f"[METRIK] BRISQUE Gambar Asli: {brisque_asli:.4f}")
    
    # 1. CLAHE (seperti yang Anda lakukan sebelumnya)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Median Filter
    median_img = cv2.medianBlur(clahe_img, 3)
    brisque_median = calculate_brisque(median_img)
    print(f"[METRIK] BRISQUE CLAHE + Median Filter: {brisque_median:.4f}")

    # 3. PENAJAMAN (Sharpening)
    usm_img = unsharp_mask(median_img, amount=2.0)
    brisque_usm = calculate_brisque(usm_img)
    print(f"[METRIK] BRISQUE Unsharp Masking: {brisque_usm:.4f}")
    
    laplacian_img = laplacian_sharpen(median_img)
    brisque_laplacian = calculate_brisque(laplacian_img)
    print(f"[METRIK] BRISQUE Laplacian Sharpening: {brisque_laplacian:.4f}")
    
    hp_img = high_pass_sharpen(median_img)
    brisque_hp = calculate_brisque(hp_img)
    print(f"[METRIK] BRISQUE High-Pass Filter: {brisque_hp:.4f}")

    # Simpan hasil di folder 'outputs'
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Menentukan folder outputs di direktori tempat script ini berada
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    
    # Jika folder outputs belum ada (berjaga-jaga), buat foldernya
    os.makedirs(output_dir, exist_ok=True)
        
    cv2.imwrite(os.path.join(output_dir, f"{name}_1_clahe_median{ext}"), median_img)
    cv2.imwrite(os.path.join(output_dir, f"{name}_2_unsharp_mask{ext}"), usm_img)
    cv2.imwrite(os.path.join(output_dir, f"{name}_3_laplacian{ext}"), laplacian_img)
    cv2.imwrite(os.path.join(output_dir, f"{name}_4_highpass{ext}"), hp_img)
    
    print("\n[BERHASIL] File hasil disimpan di folder 'outputs'.")

if __name__ == "__main__":
    print("=== TOOL PENAJAMAN CITRA BLUR KAMERA & METRIK BRISQUE ===")
    
    # Sembunyikan window utama tkinter
    root = tk.Tk()
    root.withdraw()
    
    print("Silakan pilih gambar (bisa lebih dari satu) pada jendela dialog yang muncul...")
    file_paths = filedialog.askopenfilenames(
        title="Pilih Citra yang Ingin Di-enhance",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not file_paths:
        print("Pemilihan dibatalkan.")
    else:
        print("\nMemulai proses dan penghitungan BRISQUE (nilai lebih KECIL = kualitas lebih BAIK)...")
        for idx, img_path in enumerate(file_paths, start=1):
            print(f"\n[{idx}/{len(file_paths)}] ======================")
            process_image(img_path)
            
        print("\n=== SEMUA PEMROSESAN SELESAI ===")
