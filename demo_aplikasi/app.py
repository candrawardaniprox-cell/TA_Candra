import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import time

from config import Config
from inference import ObjectDetector

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Deteksi Daun Bawang",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_custom_detector(checkpoint_path, conf_threshold, nms_threshold):
    """Memuat model TA Candra"""
    try:
        detector = ObjectDetector(
            checkpoint_path=checkpoint_path,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_threshold
        )
        return detector
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None

def main():
    st.title("🔍 Deteksi Penyakit Daun Bawang (CNN-Transformer)")
    st.markdown("Aplikasi Tugas Akhir untuk mendeteksi penyakit Moler, Slabung, dan Ulat Grayak.")

    # Menu Samping (Sidebar)
    st.sidebar.header("⚙️ Konfigurasi")
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.05, 0.01)
    nms_threshold = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.sidebar.subheader("Pengaturan Visual")
    show_boxes = st.sidebar.checkbox("Tampilkan Kotak (BBox)", value=True)
    show_labels = st.sidebar.checkbox("Tampilkan Nama Penyakit", value=True)
    show_scores = st.sidebar.checkbox("Tampilkan Skor Kepercayaan", value=True)
    box_thickness = st.sidebar.slider("Ketebalan Garis", 1, 5, 2)

    st.sidebar.subheader("Status Model")
    checkpoint_path = "checkpoints/best_model.pth"
    
    with st.spinner("Memuat Model TA Candra..."):
        detector = load_custom_detector(checkpoint_path, conf_threshold, nms_threshold)

    if detector is not None:
        st.sidebar.success("✓ Model TA Candra berhasil dimuat!")
        with st.sidebar.expander("Informasi Model"):
            st.write("**Arsitektur:** Hybrid CNN-Transformer")
            st.write("**Ukuran Input:** 512x512")
            st.write("**Kelas:** Sehat, Slabung, Moler, Ulat")
    else:
        st.sidebar.error("✗ Gagal memuat model. Pastikan file best_model.pth ada di folder checkpoints.")
        return

    # Area Utama
    col1, col2 = st.columns(2)

    with col1:
        st.header("📤 Upload Gambar")
        uploaded_file = st.file_uploader("Pilih gambar daun bawang...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            st.image(image, caption="Gambar Asli", use_container_width=True)

    with col2:
        st.header("🎯 Hasil Deteksi")
        
        if uploaded_file is not None:
            with st.spinner("Sedang mendeteksi..."):
                start_time = time.time()
                
                # Update setting threshold secara real-time
                detector.conf_threshold = conf_threshold
                detector.nms_iou_threshold = nms_threshold
                
                # Prediksi
                result = detector.predict(image_np, return_image=True)
                inference_time = time.time() - start_time
                
                st.success(f"✓ Proses selesai dalam {inference_time:.2f} detik")
                
                if len(result['boxes']) > 0:
                    if show_boxes:
                        vis_image = image_np.copy()
                        boxes_np = result['boxes'].cpu().numpy() if isinstance(result['boxes'], torch.Tensor) else result['boxes']
                        scores_np = result['scores'].cpu().numpy() if isinstance(result['scores'], torch.Tensor) else result['scores']
                        classes_np = result['classes'].cpu().numpy() if isinstance(result['classes'], torch.Tensor) else result['classes']
                        
                        # Warna berbeda untuk tiap kelas
                        colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255), (255, 0, 0)] 
                        
                        for box, score, cls, class_name in zip(boxes_np, scores_np, classes_np, result['class_names']):
                            x_center, y_center, width, height = box
                            x1, y1 = int(x_center - width/2), int(y_center - height/2)
                            x2, y2 = int(x_center + width/2), int(y_center + height/2)
                            
                            color = colors[int(cls) % len(colors)]
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, box_thickness)
                            
                            if show_labels:
                                label = class_name
                                if show_scores: 
                                    label += f": {score:.2f}"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(vis_image, (x1, y1-th-5), (x1+tw, y1), color, -1)
                                cv2.putText(vis_image, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                                
                        st.image(vis_image, caption=f"Terdeteksi {len(boxes_np)} Objek", use_container_width=True, channels="RGB")
                else:
                    st.warning("Tidak ada objek yang terdeteksi.")
                    st.info("Coba turunkan nilai 'Confidence Threshold' di menu sebelah kiri.")

if __name__ == "__main__":
    main()