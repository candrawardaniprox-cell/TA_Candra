import json
import numpy as np
from sklearn.cluster import KMeans

def calculate_anchors(json_path, num_anchors=20, image_size=224):
    print(f"Membaca data dari: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File {json_path} tidak ditemukan!")
        return
        
    # Pengecekan jumlah anotasi
    total_anotasi = len(data.get('annotations', []))
    print(f"Ditemukan {total_anotasi} anotasi di dalam file JSON.")
    
    if total_anotasi == 0:
        print("\nERROR FATAL: File JSON Anda KOSONG!")
        print("SOLUSI: Silakan copy-paste file JSON ASLI ke folder dataset Anda, lalu jalankan fix_dataset.py SATU KALI SAJA.")
        return

    boxes = []
    for ann in data['annotations']:
        if 'bbox' not in ann:
            continue
            
        w = float(ann['bbox'][2])
        h = float(ann['bbox'][3])
        
        # Hindari error pembagian 0 atau box tidak valid
        if w <= 0.0 or h <= 0.0:
            continue
            
        # Normalisasi menggunakan 224 (Sesuai config.py)
        w_norm = w / image_size
        h_norm = h / image_size
        
        boxes.append([w_norm, h_norm])
        
    boxes = np.array(boxes)
    print(f"Total bounding boxes valid yang siap dianalisis: {len(boxes)}")
    
    if len(boxes) == 0:
        print("ERROR: Tidak ada bounding box yang valid!")
        return
        
    # K-Means Clustering
    print(f"Menghitung {num_anchors} Anchor Boxes terbaik dengan K-Means...")
    
    # Jika anotasi lebih sedikit dari anchor yang diminta
    actual_anchors = min(num_anchors, len(boxes))
    
    kmeans = KMeans(n_clusters=actual_anchors, random_state=42, n_init=10)
    kmeans.fit(boxes)
    
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    
    print("\n" + "="*50)
    print("COPY KODE DI BAWAH INI KE config.py ANDA:")
    print("="*50)
    print("    ANCHOR_BOXES = [")
    for anchor in anchors:
        print(f"        ({anchor[0]:.4f}, {anchor[1]:.4f}),")
    print("    ]")
    print(f"    NUM_ANCHORS = {actual_anchors}")
    print("="*50)

if __name__ == '__main__':
    JSON_FILE = 'data/coco copy/annotations_coco/instances_train2017.json'
    # Resolusi diubah ke 224 sesuai dengan Config.IMAGE_SIZE
    calculate_anchors(JSON_FILE, num_anchors=20, image_size=224)