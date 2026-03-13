import json
from pathlib import Path

def fix_coco_json(input_path, output_path):
    print(f"Memproses {input_path}...")
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"GAGAL: File tidak ditemukan di {input_path}")
        return
        
    # Mapping berdasarkan isi file JSON asli dari Roboflow Anda:
    # 0 = sehat        -> Target 0 (sehat)
    # 1 = moler        -> Target 2 (moler)
    # 2 = sehat        -> Target 0 (sehat)
    # 3 = slabung      -> Target 1 (slabung)
    # 4 = ulat_grayak  -> Target 3 (ulat_grayak)
    
    id_mapping = {
        0: 0, 
        1: 2, # Moler aslinya 1, kita geser ke 2
        2: 0, # Sehat duplikat (2) kita jadikan 0
        3: 1, # Slabung aslinya 3, kita geser ke 1
        4: 3  # Ulat grayak aslinya 4, kita geser ke 3
    }
    
    # 1. Perbaiki category_id di semua kotak anotasi
    count_fixed = 0
    for ann in data['annotations']:
        old_id = ann['category_id']
        if old_id in id_mapping:
            if old_id == 2:
                count_fixed += 1
            ann['category_id'] = id_mapping[old_id]
            
    # 2. Ganti daftar kategori menjadi 4 kelas yang bersih
    data['categories'] = [
        {"id": 0, "name": "sehat", "supercategory": "none"},
        {"id": 1, "name": "slabung", "supercategory": "none"},
        {"id": 2, "name": "moler", "supercategory": "none"},
        {"id": 3, "name": "ulat_grayak", "supercategory": "none"}
    ]
    
    # Simpan kembali ke file JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Selesai! Berhasil menggabungkan {count_fixed} anotasi 'sehat' duplikat.")
    print("-" * 50)

# Jalankan perbaikan menggunakan nama file .json.json yang benar
train_json = "data/coco copy/annotations_coco/instances_train2017.json.json"
val_json = "data/coco copy/annotations_coco/instances_val2017.json.json"

fix_coco_json(train_json, train_json)
fix_coco_json(val_json, val_json)