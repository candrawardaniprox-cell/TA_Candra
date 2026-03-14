import json
import numpy as np
from sklearn.cluster import KMeans

def calculate_anchors(json_path, num_anchors=7, image_size=512):
    print(f"Membaca data dari: {json_path}")
    
    # Buka file JSON COCO
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    boxes = []
    # Ambil lebar dan tinggi dari setiap anotasi
    for ann in data['annotations']:
        # Format COCO bbox: [x_min, y_min, width, height]
        w = ann['bbox'][2]
        h = ann['bbox'][3]
        
        # Hindari box error yang ukurannya 0
        if w == 0 or h == 0:
            continue
            
        # Normalisasi ukuran (dibagi ukuran gambar 512)
        w_norm = w / image_size
        h_norm = h / image_size
        
        boxes.append([w_norm, h_norm])
        
    boxes = np.array(boxes)
    print(f"Total bounding boxes yang dianalisis: {len(boxes)}")
    
    # Gunakan K-Means Clustering untuk mencari bentuk kotak paling umum
    print(f"Menghitung {num_anchors} Anchor Boxes terbaik dengan K-Means...")
    kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
    kmeans.fit(boxes)
    
    # Ambil titik tengah cluster
    anchors = kmeans.cluster_centers_
    
    # Urutkan kotak dari ukuran paling kecil ke paling besar (opsional agar rapi)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    
    print("\n" + "="*50)
    print("COPY KODE DI BAWAH INI KE config.py ANDA:")
    print("="*50)
    print("    ANCHOR_BOXES = [")
    for anchor in anchors:
        print(f"        ({anchor[0]:.4f}, {anchor[1]:.4f}),")
    print("    ]")
    print(f"    NUM_ANCHORS = {num_anchors}")
    print("="*50)

if __name__ == '__main__':
    # Ganti path ini jika lokasi file JSON train Anda berbeda
    JSON_FILE = 'data/coco copy/annotations_coco/instances_train2017.json.json'
    
    calculate_anchors(JSON_FILE, num_anchors=7, image_size=512)