"""
Configuration file for Hybrid CNN-Transformer Object Detection System.

This module contains all hyperparameters, paths, and configuration settings
for training, evaluation, and inference.
"""

import torch
from pathlib import Path


class Config:
    """Configuration class containing all hyperparameters and settings."""

    # ==================== Model Architecture ====================
    # Input/Output specifications
    IMAGE_SIZE = 320  # Input image resolution (320x320)
    NUM_CLASSES = 80  # Number of object classes (COCO default)

    # CNN Backbone settings
    BACKBONE_CHANNELS = [3, 32, 64, 128, 256]  # Progressive channel expansion
    BACKBONE_KERNEL_SIZE = 3
    BACKBONE_PADDING = 1

    # Transformer settings
    TRANSFORMER_DIM = 256  # Must match last BACKBONE_CHANNELS value
    TRANSFORMER_HEADS = 8  # Number of attention heads
    TRANSFORMER_LAYERS = 2  # Number of transformer encoder layers
    TRANSFORMER_FF_DIM = 1024  # Feed-forward network dimension
    TRANSFORMER_DROPOUT = 0.1

    # Detection Head settings
    GRID_SIZE = 20  # Output grid size (IMAGE_SIZE / 16 = 320 / 16 = 20)
    NUM_ANCHORS = 3  # Number of anchor boxes per grid cell

    # Anchor boxes (width, height) relative to grid cell - optimized for COCO
    ANCHOR_BOXES = [
        (0.28, 0.22),  # Small objects
        (0.38, 0.48),  # Medium objects
        (0.90, 0.78),  # Large objects
    ]

    # ==================== Training Configuration ====================
    # Optimization
    BATCH_SIZE = 32  # Adjusted for RTX 3060 12GB VRAM
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    EPOCHS = 5
    WARMUP_EPOCHS = 3

    # Learning rate schedule
    LR_SCHEDULER = "cosine"  # Options: "cosine", "step"
    LR_STEP_SIZE = 30  # For step scheduler
    LR_GAMMA = 0.1  # For step scheduler

    # Gradient clipping
    GRAD_CLIP_NORM = 1.0

    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision for RTX 3060

    # ==================== Loss Configuration ====================
    # Loss weights
    LAMBDA_OBJ = 1.0  # Objectness loss weight
    LAMBDA_NOOBJ = 0.5  # No-object loss weight
    LAMBDA_BBOX = 5.0  # Bounding box loss weight
    LAMBDA_CLASS = 1.0  # Classification loss weight

    # Loss types
    BBOX_LOSS_TYPE = "giou"  # Options: "mse", "giou", "ciou"
    USE_FOCAL_LOSS = True  # Use focal loss for classification
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # IoU thresholds for anchor assignment
    IOU_THRESHOLD_POS = 0.5  # IoU > 0.5 = positive anchor
    IOU_THRESHOLD_NEG = 0.4  # IoU < 0.4 = negative anchor

    # ==================== Inference Configuration ====================
    CONF_THRESHOLD = 0.5  # Confidence threshold for detection
    NMS_IOU_THRESHOLD = 0.45  # IoU threshold for NMS
    MAX_DETECTIONS = 100  # Maximum number of detections per image

    # ==================== Data Configuration ====================
    # Dataset paths
    DATA_ROOT = Path("data") / "coco copy"
    TRAIN_IMAGES = DATA_ROOT / "val2017"  # Using val2017 for training (for now)
    VAL_IMAGES = DATA_ROOT / "val2017"
    TRAIN_ANNOTATIONS = DATA_ROOT / "annotations_coco" / "instances_train2017.json"
    VAL_ANNOTATIONS = DATA_ROOT / "annotations_coco" / "instances_val2017.json"

    # Data loading
    NUM_WORKERS = 4  # Number of dataloader workers
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    PERSISTENT_WORKERS = True  # Keep workers alive between epochs

    # Data augmentation
    AUGMENT = True
    HORIZONTAL_FLIP_PROB = 0.5
    BRIGHTNESS_CONTRAST_LIMIT = 0.2
    HUE_SATURATION_VALUE_LIMIT = 20

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==================== Checkpoint & Logging ====================
    # Directories
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    OUTPUT_DIR = Path("outputs")

    # Checkpoint settings
    SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
    KEEP_LAST_N_CHECKPOINTS = 3  # Keep only last N checkpoints

    # Logging
    LOG_FREQUENCY = 10  # Log every N batches
    USE_TENSORBOARD = True

    # ==================== Evaluation Configuration ====================
    EVAL_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    EVAL_FREQUENCY = 1  # Evaluate every N epochs

    # ==================== Device Configuration ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== COCO Class Names ====================
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls):
        """Validate configuration settings."""
        assert cls.IMAGE_SIZE % 16 == 0, "IMAGE_SIZE must be divisible by 16"
        assert cls.GRID_SIZE == cls.IMAGE_SIZE // 16, "GRID_SIZE must be IMAGE_SIZE // 16"
        assert cls.TRANSFORMER_DIM == cls.BACKBONE_CHANNELS[-1], \
            "TRANSFORMER_DIM must match last BACKBONE_CHANNELS value"
        assert len(cls.ANCHOR_BOXES) == cls.NUM_ANCHORS, \
            "Number of ANCHOR_BOXES must match NUM_ANCHORS"
        assert cls.TRANSFORMER_DIM % cls.TRANSFORMER_HEADS == 0, \
            "TRANSFORMER_DIM must be divisible by TRANSFORMER_HEADS"

    @classmethod
    def get_model_size(cls):
        """Calculate approximate model size in parameters."""
        # Approximate calculation
        backbone_params = sum([
            cls.BACKBONE_CHANNELS[i] * cls.BACKBONE_CHANNELS[i+1] *
            cls.BACKBONE_KERNEL_SIZE ** 2
            for i in range(len(cls.BACKBONE_CHANNELS) - 1)
        ])

        transformer_params = cls.TRANSFORMER_LAYERS * (
            4 * cls.TRANSFORMER_DIM ** 2 +  # Attention
            2 * cls.TRANSFORMER_DIM * cls.TRANSFORMER_FF_DIM  # FFN
        )

        head_params = cls.TRANSFORMER_DIM * cls.NUM_ANCHORS * (5 + cls.NUM_CLASSES)

        total_params = backbone_params + transformer_params + head_params
        return total_params / 1e6  # Return in millions

    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("=" * 60)
        print("Hybrid CNN-Transformer Object Detection - Configuration")
        print("=" * 60)
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Grid Size: {cls.GRID_SIZE}x{cls.GRID_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.USE_AMP}")
        print(f"Approximate Model Size: {cls.get_model_size():.2f}M parameters")
        print("=" * 60)


# Create a default config instance
config = Config()


if __name__ == "__main__":
    # Validate and print configuration
    Config.validate_config()
    Config.print_config()
    Config.create_directories()
