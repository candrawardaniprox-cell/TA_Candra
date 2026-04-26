import json
import os
from collections import Counter

def bedah_json_coco(json_path):
    print(f"==================================================")
    print(f"Menganalisis file: {json_path}")
    print(f"==================================================")
    
    if not os.path.exists(json_path):
        print(f"File tidak ditemukan: {json_path}\n")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Cek daftar kategori / kelas
    kategori_dict = {}
    print("[1] DAFTAR KATEGORI (CLASSES):")
    if 'categories' in data:
        for cat in data['categories']:
            cat_id = cat.get('id')
            cat_name = cat.get('name')
            kategori_dict[cat_id] = cat_name
            print(f"    - ID: {cat_id} | Nama Kelas: '{cat_name}'")
    else:
        print("    Tidak ditemukan field 'categories'!")

    # 2. Cek jumlah kotak (bounding box) per kelas
    print("\n[2] JUMLAH OBJEK PER KELAS:")
    if 'annotations' in data:
        kumpulan_id = [ann.get('category_id') for ann in data['annotations']]
        jumlah_per_id = Counter(kumpulan_id)
        
        for cat_id, jumlah in sorted(jumlah_per_id.items()):
            nama = kategori_dict.get(cat_id, "TIDAK DIKETAHUI")
            print(f"    - ID: {cat_id} ('{nama}') memiliki {jumlah} bounding box")
            
        # Cek jika ada anotasi dengan ID yang tidak ada di daftar kategori
        id_tak_terdaftar = set(kumpulan_id) - set(kategori_dict.keys())
        if id_tak_terdaftar:
            print(f"\n    PERINGATAN: Ditemukan anotasi dengan ID yang tidak terdaftar di kategori: {id_tak_terdaftar}")
    else:
        print("    Tidak ditemukan field 'annotations'!")
    
    print("\n")

if __name__ == "__main__":
    # Ganti path ini sesuai dengan lokasi file JSON Anda
    # Contoh format path jika ada di dalam folder data/
    file_json_saya = [
        "data/coco copy/annotations_coco/instances_train2017.json",  # Sesuaikan nama file train Anda
        "data/coco copy/annotations_coco/instances_val2017.json",  # Sesuaikan nama file valid/val Anda
        "data/coco copy/annotations_coco/instances_test2017.json"    # Sesuaikan nama file test Anda
    ]

    for file_path in file_json_saya:
        bedah_json_coco(file_path)