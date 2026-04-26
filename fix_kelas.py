import json
import os

def perbaiki_json(input_path, output_path):
    print(f"Memproses: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Aturan mapping ID baru (menggeser ID 1, 2, 3 menjadi 0, 1, 2)
    id_mapping = {
        1: 0,  # moler
        2: 1,  # slabung
        3: 2   # ulat_grayak
    }

    # 2. Perbarui daftar categories (buang ID 0 'TA_Candra')
    new_categories = []
    for cat in data.get('categories', []):
        old_id = cat['id']
        if old_id in id_mapping:  # Hanya ambil jika ada di mapping (1, 2, 3)
            new_cat = cat.copy()
            new_cat['id'] = id_mapping[old_id]
            new_categories.append(new_cat)
    
    data['categories'] = new_categories

    # 3. Perbarui category_id pada semua bounding box / annotations
    if 'annotations' in data:
        for ann in data['annotations']:
            old_id = ann['category_id']
            if old_id in id_mapping:
                ann['category_id'] = id_mapping[old_id]
            else:
                print(f"  [Peringatan] Ditemukan ID {old_id} yang tidak diketahui!")

    # 4. Simpan ke file output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
    print(f"  -> Berhasil diperbarui!\n")

if __name__ == "__main__":
    # Daftar file JSON yang akan diperbaiki berdasarkan path Anda
    files_to_fix = [
        "data/coco copy/annotations_coco/instances_train2017.json",
        "data/coco copy/annotations_coco/instances_val2017.json",
        "data/coco copy/annotations_coco/instances_test2017.json"
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            # Buat nama file backup (.bak)
            backup_path = file_path + ".bak"
            
            # Ubah nama file asli menjadi backup
            os.rename(file_path, backup_path) 
            
            # Proses file backup dan simpan dengan nama file aslinya
            perbaiki_json(backup_path, file_path)
        else:
            print(f"File tidak ditemukan: {file_path}")