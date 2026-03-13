import os
from PIL import Image, ImageOps

def resize_gambar():
    # ================= PENGATURAN =================
    # Nama folder tempat kamu menaruh gambar asli
    input_folder = "gambar_asli" 
    
    # Nama folder tempat hasil gambar yang sudah di-resize
    output_folder = "gambar_resize" 
    
    # Ukuran target (lebar, tinggi) -> 512x512
    target_size = (512, 512) 
    # ==============================================

    # Buat folder output otomatis jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Folder '{output_folder}' berhasil dibuat.")

    # Cek apakah folder input ada
    if not os.path.exists(input_folder):
        print(f"❌ Error: Folder '{input_folder}' tidak ditemukan!")
        print(f"👉 Silakan buat folder '{input_folder}' dan masukkan gambar ke dalamnya.")
        return

    # Ambil semua file di dalam folder input
    file_gambar = os.listdir(input_folder)
    gambar_berhasil = 0

    print(f"⏳ Memulai proses resize ke ukuran {target_size}...")
    
    for filename in file_gambar:
        # Hanya proses file yang berekstensi gambar
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path_input = os.path.join(input_folder, filename)
            path_output = os.path.join(output_folder, filename)
            
            try:
                # Buka gambar
                img = Image.open(path_input)
                
                # [PERBAIKAN] Terapkan rotasi sesuai metadata EXIF kamera
                img = ImageOps.exif_transpose(img)
                
                # Konversi ke RGB jika gambar berupa RGBA (PNG transparan)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Lakukan resize
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Simpan gambar
                img_resized.save(path_output)
                
                print(f"✅ Berhasil: {filename}")
                gambar_berhasil += 1
            except Exception as e:
                print(f"❌ Gagal memproses {filename}: {e}")

    print("=======================================")
    print(f"🎉 Selesai! {gambar_berhasil} gambar berhasil di-resize.")
    print(f"Cek hasilnya di folder '{output_folder}'.")

if __name__ == "__main__":
    resize_gambar()