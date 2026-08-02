# Hybrid CNN-Transformer Object Detection System

A lightweight and efficient object detection system combining CNN and Transformer architectures for onion leaf disease identification, optimized for NVIDIA GEFORCE RTX 4090 GPU training.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Web UI](#web-ui)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

This project implements a modern object detection system that combines the local feature extraction capabilities of CNNs with the global context modeling of Transformers. The model is designed to be lightweight (~6.01M parameters) while maintaining competitive performance.

**Key Highlights:**
- Hybrid CNN-Transformer architecture (HTEM-style)
- Optimized for NVIDIA GEFORCE RTX 4090 GPU (24GB VRAM)
- Real-time inference capability (>30 FPS)
- Support for 3 onion disease classes (moler, slabung, ulat_grayak)
- Interactive Streamlit web interface
- Comprehensive training and evaluation pipelines

## Features

- **Modular Architecture**: Clean separation of backbone, transformer, and detection head
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for faster training
- **Advanced Augmentation**: Albumentations-based data augmentation pipeline
- **Flexible Loss Function**: GIoU bounding box loss with focal classification loss
- **Multiple Evaluation Metrics**: mAP@0.5, mAP@0.5:0.95, per-class statistics
- **Visualization Tools**: Built-in tools for visualizing detections
- **Web Interface**: Easy-to-use Streamlit UI for inference
- **TensorBoard Integration**: Real-time training monitoring

## Architecture

### Overall Architecture Flow

```text
Input Image [B, 3, 640, 640]
        ↓
┌───────────────────┐
│   CNN Backbone    │  Feature extraction (ResNet /
│                   │  MobileNet)
└───────────────────┘
        ↓
┌───────────────────┐
│    CTE Layer      │  Convolutional Transformer
│                   │  Encoder (Adapts features)
└───────────────────┘
        ↓
┌───────────────────┐
│  Hybrid Stages    │  4 Stages (Feature Embedding + 
│ (Transformer +    │  [CPSA + LFFN] Blocks)
│  Conv)            │
└───────────────────┘
        ↓
┌───────────────────┐
│  FPN Top-Down     │  Fusion of multi-scale features
│  Fusion           │  (P2, P3, P4, P5)
└───────────────────┘
        ↓
┌───────────────────┐
│  Anchor-Free      │  Outputs: [logits, reg_offsets, 
│  Detection Head   │  centerness]
└───────────────────┘
        ↓
Detections: [class_scores, bbox (x1, y1, x2, y2)]
```

### Components

1. **CNN Backbone**
   - ResNet or MobileNet for initial hierarchical feature extraction

2. **CTE (Convolutional Transformer Encoder)**
   - Bridge layer to adapt CNN spatial features into embeddings

3. **Transformer / Hybrid Encoder Stages**
   - Feature Embeddings with CPSA (Convolutional Positional Spatial Attention) and LFFN blocks
   - Configurable channel expansion and reduction ratios

4. **Detection Head**
   - Anchor-free detection head across P2-P5 FPN levels
   - Predicts: [logits (class_scores), reg_offsets, centerness]

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU support)
- NVIDIA GEFORCE RTX 4090 or similar high-end GPU (24GB VRAM recommended)

### Setup

1. **Clone the repository** (or create a new directory)
```bash
cd "D:\Project final year"
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Preparation

This project uses a **private agricultural dataset** (*dataset pribadi*) containing images of onion leaf diseases (`moler`, `slabung`, `ulat_grayak`). The data annotations are structured following the COCO JSON format.

### Current Dataset Structure
```text
data/
└── coco copy/  (or Dataset2026_split, depending on config.SCENARIO)
    ├── annotations_coco/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   └── instances_test2017.json
    ├── train2017/
    ├── val2017/
    └── test2017/
```

### Configuration

You can change the dataset scenario in `config.py` by modifying `SCENARIO` (e.g., `"coco_copy"`, `"augmented_2026"`).

## Usage

### Training

**Basic Training**
```bash
python train.py
```

**Resume from Checkpoint**
```bash
python train.py --resume checkpoints/checkpoint_epoch_10.pth
```

**Training Configuration**

All training hyperparameters are in `config.py`:
- Batch size: 2 to 24 (adjust based on your NVIDIA RTX 4090 capacity)
- Learning rate: 2e-4 (with cosine annealing)
- Epochs: 70
- Image size: 640×640

**Monitor Training**

View training progress in TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

### Evaluation

Evaluate a trained model:

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --batch-size 8 \
    --save-visualizations \
    --output-dir outputs/evaluation
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--batch-size`: Batch size for evaluation (default: 8)
- `--conf-threshold`: Confidence threshold (default: 0.5)
- `--nms-iou-threshold`: NMS IoU threshold (default: 0.45)
- `--save-visualizations`: Save visualization samples
- `--num-vis-samples`: Number of samples to visualize (default: 10)
- `--output-dir`: Output directory (default: outputs/evaluation)

### Inference

**Single Image**
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --visualize
```

**Batch Processing**
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image-dir path/to/images/ \
    --output-dir outputs/inference
```

**Python API**
```python
from inference import ObjectDetector

# Load model
detector = ObjectDetector(
    checkpoint_path='checkpoints/best_model.pth',
    conf_threshold=0.5
)

# Run inference
result = detector.predict('image.jpg')

# Visualize
detector.visualize_prediction(
    'image.jpg',
    save_path='output.jpg',
    show=True
)
```

### Web UI

Launch the interactive Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload images via drag-and-drop
- Adjust confidence and NMS thresholds in real-time
- View detection results with bounding boxes
- See per-class statistics
- Download visualizations

## Project Structure

```
.
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Training script
├── eval.py                   # Evaluation script
├── inference.py              # Inference script
├── app.py                    # Streamlit web UI
├── requirements.txt          # Python dependencies
│
├── models/                   # Model architecture
│   ├── __init__.py
│   ├── backbone.py          # CNN backbone
│   ├── transformer.py       # Transformer encoder
│   ├── detection_head.py    # Detection head
│   └── hybrid_model.py      # Complete model
│
├── data/                     # Data handling
│   ├── __init__.py
│   ├── dataset.py           # Dataset class
│   ├── transforms.py        # Data augmentation
│   └── utils.py             # Data utilities
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── loss.py              # Loss functions
│   ├── metrics.py           # Evaluation metrics
│   ├── nms.py               # Non-Maximum Suppression
│   └── visualization.py     # Visualization tools
│
├── data/                     # Dataset directory
│   └── coco copy/
│       ├── annotations_coco/
│       └── val2017/
│
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
└── outputs/                  # Evaluation outputs
```

## Model Architecture

### Model Size
- **Total Parameters**: ~6.01M

### Design Choices

**Why CNN + Transformer?**
- **CNN**: Efficiently extracts local features with inductive biases
- **Transformer**: Captures global context and long-range dependencies
- **Hybrid**: Best of both worlds - efficiency + global reasoning

**Why Anchor-Free Detection?**
- Simplifies architecture without relying on fixed anchor boxes
- Better adaptation to varying object scales via FPN (P2-P5)
- Highly efficient single-stage design

**Why GIoU Loss?**
- Better gradients than MSE for bounding boxes
- Works well even when boxes don't overlap
- Improves localization accuracy

## Performance

### Expected Metrics (Onion Leaf Disease Dataset)

| Metric | Expected Value |
|--------|---------------|
| mAP@0.5 | Target: > 80% |
| Inference Speed | > 30 FPS on NVIDIA RTX 4090 |
| Training Time | ~1-3 hours |

### Memory Usage
- **Training**: ~12-16GB VRAM on RTX 4090
- **Inference**: ~2-4GB VRAM

### Speed Benchmarks
- **NVIDIA RTX 4090**: High FPS
- **CPU**: ~2-3 FPS

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 8  # or 4
```

**2. Dataset Not Found**
```
FileNotFoundError: Annotation file not found
```
**Solution**: Verify dataset paths in `config.py`
```python
DATA_ROOT = Path("data") / "coco copy"
VAL_ANNOTATIONS = DATA_ROOT / "annotations_coco" / "instances_val2017.json"
```

**3. ImportError for Albumentations**
```
ImportError: cannot import name 'Compose' from 'albumentations'
```
**Solution**: Reinstall albumentations
```bash
pip install --upgrade albumentations
```

**4. Slow Data Loading**
```
# Training is slow due to data loading
```
**Solution**: Adjust num_workers in `config.py`
```python
NUM_WORKERS = 2  # Reduce if too high
```

**5. Model Not Learning**
- Check learning rate (try 5e-5 or 2e-4)
- Verify data augmentation isn't too aggressive
- Check loss weights in config.py
- Ensure anchors match your dataset

### Performance Tips

1. **Mixed Precision Training**: Already enabled in `config.py`
2. **Gradient Accumulation**: For larger effective batch size
3. **Learning Rate Warmup**: First 5 epochs use warmup
4. **Data Augmentation**: Tune in `config.py` if needed

## Configuration

Key settings in `config.py`:

```python
# Model
IMAGE_SIZE = 640
NUM_CLASSES = 3

# Training
BATCH_SIZE = 2 # Up to 24 or higher for RTX 4090
LEARNING_RATE = 2e-4
EPOCHS = 70

# Inference & Detection
CONF_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.40
```

## Testing Components

Test individual components:

```bash
# Test CNN backbone
python models/backbone.py

# Test Transformer
python models/transformer.py

# Test Detection head
python models/detection_head.py

# Test Complete model
python models/hybrid_model.py

# Test Loss function
python utils/loss.py

# Test NMS
python utils/nms.py
```

## References

- **COCO Annotation Format**: Used for structuring the private dataset annotations.
- **Transformer**: "Attention Is All You Need" (Vaswani et al., 2017)
- **DETR**: "End-to-End Object Detection with Transformers" (Carion et al., 2020)
- **YOLOv3**: "YOLOv3: An Incremental Improvement" (Redmon & Farhadi, 2018)
- **GIoU**: "Generalized Intersection over Union" (Rezatofighi et al., 2019)

## License

This project is for educational purposes.

## Acknowledgments

- PyTorch team for the deep learning framework
- Albumentations for data augmentation library
- Streamlit for the web UI framework

---


