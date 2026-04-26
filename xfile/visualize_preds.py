import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from models.hybrid_model import HybridDetector 
from config import Config

# ==========================================
# 1. KONFIGURASI
# ==========================================
MODEL_PATH = "checkpoints/best_model.pth"
IMAGE_DIR = r"D:\TA_Candra\data\coco copy\test2017" 
OUTPUT_DIR = r"D:\TA_Candra\output_test"

# Nilai threshold rendah agar hama kecil (ulat/slabung) lebih mudah muncul
CONF_THRESHOLD = 0.10  
IOU_THRESHOLD = 0.40 

CLASSES = ["moler", "slabung", "ulat_grayak"] 
# Warna BGR untuk OpenCV: Moler(Hijau), Slabung(Merah), Ulat(Biru)
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD MODEL
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Pastikan parameter HybridDetector sesuai dengan arsitektur FPN terbaru Anda
model = HybridDetector(
    num_classes=len(CLASSES),
    image_size=Config.IMAGE_SIZE,
    transformer_layers=Config.TRANSFORMER_LAYERS
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint)) 
model.eval()

# ==========================================
# 3. PREPARASI DATA
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.MEAN, std=Config.STD)
])

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Memulai proses deteksi FPN untuk {len(image_files)} gambar...")

# ==========================================
# 4. LOOPING INFERENSI
# ==========================================
for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: continue
        
    # Resize ke 640x640 sesuai Config FPN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        # MENGGUNAKAN FUNGSI CERDAS MODEL UNTUK FPN
        detections = model.get_detections(
            img_tensor, 
            conf_threshold=CONF_THRESHOLD, 
            nms_iou_threshold=IOU_THRESHOLD
        )
    
    # Ambil hasil deteksi gambar pertama di batch
    det = detections[0]
    final_boxes = det['boxes'].cpu().numpy()
    final_scores = det['scores'].cpu().numpy()
    final_class_ids = det['classes'].cpu().numpy()

    # ==========================================
    # 5. VISUALISASI
    # ==========================================
    img_draw = img_resized.copy()
    
    for i in range(len(final_boxes)):
        # Koordinat sudah dalam skala pixel 640x640
        x1, y1, x2, y2 = map(int, final_boxes[i])
        score = final_scores[i]
        cls_id = int(final_class_ids[i])
        
        if cls_id < len(CLASSES):
            color = COLORS[cls_id]
            label = f"{CLASSES[cls_id]}: {score:.2f}"
            
            # Gambar kotak dan teks
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            # Background teks agar mudah dibaca
            cv2.rectangle(img_draw, (x1, y1-20), (x1+120, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Simpan Hasil
    img_result_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img_result_bgr)
    print(f"-> Saved: {img_name} ({len(final_boxes)} objects)")

print(f"\nSelesai! Cek folder: {OUTPUT_DIR}")