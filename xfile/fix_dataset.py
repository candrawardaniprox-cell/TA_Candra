import json
from pathlib import Path

def fix_coco_json(input_path, output_path, target_classes=["moler", "slabung", "ulat_grayak"]):
    print(f"\n{'='*50}\nMemproses {input_path}...")
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"GAGAL: File tidak ditemukan di {input_path}")
        return
        
    # Buat pemetaan ID baru (moler=0, slabung=1, ulat_grayak=2)
    desired_mapping = {name.lower(): idx for idx, name in enumerate(target_classes)}
    
    # --- 1. MENDATA KELAS DAN MENCARI ID TARGET ---
    old_id_to_new_id = {}
    id_to_name = {} 
    
    for cat in data.get('categories', []):
        cat_name_lower = cat['name'].lower()
        id_to_name[cat['id']] = cat['name']
        
        # Jika kelas ada di daftar target kita, simpan ID aslinya
        if cat_name_lower in desired_mapping:
            old_id_to_new_id[cat['id']] = desired_mapping[cat_name_lower]
            
    if not old_id_to_new_id:
        print("ERROR: Tidak ada satupun kelas target yang ditemukan!")
        print("SOLUSI: Pastikan Anda menggunakan file JSON asli yang belum difilter/dipotong.")
        return
        
    print("  -> Pemetaan ID Kelas yang akan digunakan:")
    for old_id, new_id in old_id_to_new_id.items():
        print(f"     * '{id_to_name[old_id]}' (ID Asli JSON: {old_id}) -> Diubah jadi ID: {new_id}")
    
    # --- [INFO] HITUNG JUMLAH BBOX SEBELUM DIFILTER ---
    bbox_sebelum = {}
    for ann in data.get('annotations', []):
        cat_id = int(ann['category_id'])
        bbox_sebelum[cat_id] = bbox_sebelum.get(cat_id, 0) + 1
        
    print("\n  [INFO] Laporan Jumlah BBox SEBELUM difilter:")
    for cat_id, count in bbox_sebelum.items():
        nama_kelas = id_to_name.get(cat_id, f"Unknown (ID: {cat_id})")
        print(f"    - Kelas '{nama_kelas}' : {count} Bounding Box")
    
    # --- 2. PERBAIKI GAMBAR (Ubah ke Integer) ---
    for img in data.get('images', []):
        if 'width' in img: img['width'] = int(img['width'])
        if 'height' in img: img['height'] = int(img['height'])
    
    # --- 3. FILTER ANOTASI DAN UBAH KE TARGET BARU ---
    new_annotations = []
    bbox_sesudah = {new_id: 0 for new_id in desired_mapping.values()}
    
    for ann in data.get('annotations', []):
        old_id = int(ann['category_id']) 
        
        # Hanya ambil anotasi jika old_id masuk dalam target kita
        if old_id in old_id_to_new_id:
            new_id = old_id_to_new_id[old_id]
            ann['category_id'] = new_id # Timpa dengan ID baru (0, 1, atau 2)
            
            # Ubah koordinat bbox jadi angka desimal
            if 'bbox' in ann:
                ann['bbox'] = [float(x) for x in ann['bbox']]
            if 'area' in ann:
                ann['area'] = float(ann['area'])
                
            new_annotations.append(ann)
            bbox_sesudah[new_id] += 1
            
    data['annotations'] = new_annotations
            
    # --- 4. GANTI KATEGORI (Sisa 3 Kelas Saja) ---
    new_categories = []
    for name, new_id in desired_mapping.items():
        new_categories.append({"id": new_id, "name": name, "supercategory": "none"})
    data['categories'] = new_categories
    
    # --- [INFO] HITUNG JUMLAH BBOX SESUDAH DIFILTER ---
    print("\n  [INFO] Laporan Jumlah BBox SESUDAH difilter (Hanya 3 Kelas):")
    for name, new_id in desired_mapping.items():
        print(f"    - Kelas '{name}' : {bbox_sesudah[new_id]} Bounding Box")
    print("  " + "-"*40)
    
    # Simpan
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ Selesai! Data berhasil disimpan ulang dan bersih dari kelas sampah.")
    print("=" * 50)

if __name__ == '__main__':
    train_json = "data/coco copy/annotations_coco/instances_train2017.json"
    val_json = "data/coco copy/annotations_coco/instances_val2017.json"
    test_json = "data/coco copy/annotations_coco/instances_test2017.json"  # <-- BARIS BARU DITAMBAHKAN

    # SEKARANG MENGGUNAKAN LIST BERISI 3 KELAS
    KELAS_YANG_DIPILIH = ["moler", "slabung", "ulat_grayak"] 

    fix_coco_json(train_json, train_json, target_classes=KELAS_YANG_DIPILIH)
    fix_coco_json(val_json, val_json, target_classes=KELAS_YANG_DIPILIH)
    fix_coco_json(test_json, test_json, target_classes=KELAS_YANG_DIPILIH)  # <-- BARIS BARU DITAMBAHKAN