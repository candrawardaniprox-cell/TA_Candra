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
    
    # [PENTING] Angka awal untuk penamaan file baru
    angka_awal = 2001
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
    
    # Gunakan angka_awal sebagai counter penamaan
    counter_nama = angka_awal 

    print(f"⏳ Memulai proses resize dan rename file mulai dari angka {angka_awal}...")
    
    for filename in file_gambar:
        # Hanya proses file yang berekstensi gambar
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path_input = os.path.join(input_folder, filename)
            
            # Ambil ekstensi file asli (misal: .jpg atau .png)
            ekstensi = os.path.splitext(filename)[1].lower()
            
            # Buat nama file baru menggunakan angka counter
            nama_baru = f"{counter_nama}{ekstensi}"
            path_output = os.path.join(output_folder, nama_baru)
            
            try:
                # Buka gambar
                img = Image.open(path_input)
                
                # Terapkan rotasi sesuai metadata EXIF kamera
                img = ImageOps.exif_transpose(img)
                
                # Konversi ke RGB jika gambar berupa RGBA (PNG transparan)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Lakukan resize
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Simpan gambar dengan nama baru
                img_resized.save(path_output)
                
                print(f"✅ Berhasil: {filename} -> menjadi -> {nama_baru}")
                gambar_berhasil += 1
                
                # Tambah angka 1 untuk gambar berikutnya (2001 -> 2002 -> 2003 dst)
                counter_nama += 1 
                
            except Exception as e:
                print(f"❌ Gagal memproses {filename}: {e}")

    print("=======================================")
    print(f"🎉 Selesai! {gambar_berhasil} gambar berhasil diproses.")
    print(f"Cek hasilnya di folder '{output_folder}'.")

if __name__ == "__main__":
    resize_gambar()