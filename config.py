"""
config.py — Konfigurasi Hybrid CNN-Transformer Object Detection.

Perubahan:
  - BASE_OUTPUT_DIR: folder induk semua output.
  - Folder run per-eksperimen (timestamped) dibuat otomatis di train.py.
  - Struktur folder output:
      outputs/
        run_YYYYMMDD_HHMMSS/
          checkpoints/  ← model .pth
          logs/         ← file .log
          graphs/       ← semua gambar grafik .png
          test_results/ ← gambar perbandingan hasil prediksi
"""

import torch
from pathlib import Path


class Config:
    """Semua hyperparameter dan pengaturan sistem."""

    # ==================== Arsitektur Model ====================
    IMAGE_SIZE   = 640
    NUM_CLASSES  = 3

    BACKBONE_CHANNELS  = [3, 32, 64, 128, 256]
    BACKBONE_KERNEL_SIZE = 3
    BACKBONE_PADDING   = 1
    BACKBONE_NAME      = 'resnet18'
    BACKBONE_PRETRAINED = True

    PAPER_CTE_CHANNELS = 32
    PAPER_STAGE_DIMS = [32, 64, 96, 160]
    PAPER_STAGE_LAYOUT = [2, 2, 4, 2]
    PAPER_STAGE_HEADS = [1, 2, 4, 8]
    PAPER_STAGE_REDUCTIONS = [8, 4, 2, 1]
    PAPER_LFFN_EXPANSION_RATIO = 4
    PAPER_LFFN_KERNEL_SIZE = 3
    PAPER_EMBED_KERNEL_SIZE = 2

    # ==================== Stage-2 Classifier ====================
    CLASSIFIER_IMAGE_SIZE = 224
    CLASSIFIER_BATCH_SIZE = 24
    CLASSIFIER_LEARNING_RATE = 1e-4
    CLASSIFIER_WEIGHT_DECAY = 1e-4
    CLASSIFIER_EPOCHS = 30
    CLASSIFIER_DROPOUT = 0.2
    CLASSIFIER_NUM_WORKERS = 0
    CLASSIFIER_BLAS_NUM_THREADS = 1
    CLASSIFIER_CROP_PADDING = 0.15
    CLASSIFIER_MIN_CROP_SIZE = 12
    CLASSIFIER_LABEL_SMOOTHING = 0.05
    CLASSIFIER_CHECKPOINT_NAME = "best_classifier.pth"
    CLASSIFIER_SAVE_VISUALIZATIONS = True
    CLASSIFIER_VIS_MAX_IMAGES = 100
    ENABLE_STAGE2_CLASSIFIER = False

    # Ambang keputusan aman untuk deployment dua tahap.
    STAGE2_CLASS_THRESHOLDS = [0.90, 0.85, 0.85]  # moler, slabung, ulat_grayak
    STAGE2_MIN_MARGIN = 0.15
    STAGE2_UNKNOWN_NAME = "unknown"
    STAGE2_ACTION_MAP = [
        "cabut_tanaman",
        "semprot_pestisida_slabung",
        "semprot_pestisida_ulat_grayak",
        "no_action",
    ]
    STAGE2_PROPOSAL_CONF_THRESHOLD = 0.15
    STAGE2_PROPOSAL_NMS_IOU_THRESHOLD = 0.50
    STAGE2_PROPOSAL_MAX_DETECTIONS = 20

    TRANSFORMER_DIM     = 256
    TRANSFORMER_HEADS   = 4
    TRANSFORMER_LAYERS  = 3
    TRANSFORMER_FF_DIM  = 512
    TRANSFORMER_DROPOUT = 0.2

    GRID_SIZE = IMAGE_SIZE // 16

    # ==================== Training ====================
    BATCH_SIZE    = 2
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY  = 1e-2
    EPOCHS        = 50
    WARMUP_EPOCHS = 5

    LR_SCHEDULER = "cosine"
    LR_STEP_SIZE = 30
    LR_GAMMA     = 0.1

    GRAD_CLIP_NORM = 1.0
    USE_AMP        = True
    CUDA_BENCHMARK = True
    ALLOW_TF32     = True

    # ==================== Loss ====================
    LAMBDA_OBJ   = 0.25
    LAMBDA_NOOBJ = 5.0
    LAMBDA_BBOX  = 1.00
    LAMBDA_CLASS = 1.75
    CLASS_PRIORITY_MODE = True

    BBOX_LOSS_TYPE    = "giou"
    USE_FOCAL_LOSS    = True
    FOCAL_ALPHA       = 0.25
    FOCAL_GAMMA       = 2.0

    IOU_THRESHOLD_POS = 0.4
    IOU_THRESHOLD_NEG = 0.3

    # ==================== Inference ====================
    CONF_THRESHOLD     = 0.35
    NMS_IOU_THRESHOLD  = 0.40
    MAX_DETECTIONS     = 12
    CLASS_CONF_THRESHOLD    = 0.35
    CLASS_NMS_IOU_THRESHOLD = 0.40
    CLASS_MAX_DETECTIONS    = 10
    CLASS_METRIC_CONF_THRESHOLD = 0.45
    CLASS_METRIC_NMS_IOU_THRESHOLD = 0.30
    CLASS_METRIC_MAX_DETECTIONS = 10
    CLASS_METRIC_USE_CENTERNESS = False
    CLASS_METRIC_USE_SECOND_NMS = True
    CLASS_METRIC_SECOND_NMS_IOU_THRESHOLD = 0.20
    DET_CONF_THRESHOLD      = 0.20
    DET_NMS_IOU_THRESHOLD   = 0.50
    DET_MAX_DETECTIONS      = 50
    DET_PRE_NMS_TOPK        = 300
    USE_CENTERNESS_IN_SCORE = True
    CENTERNESS_SCORE_WEIGHT = 0.35

    # ==================== Data ====================
    DATA_ROOT          = Path("data") / "coco copy"
    TRAIN_IMAGES       = DATA_ROOT / "train2017"
    VAL_IMAGES         = DATA_ROOT / "val2017"
    TEST_IMAGES        = DATA_ROOT / "test2017"
    TRAIN_ANNOTATIONS  = DATA_ROOT / "annotations_coco" / "instances_train2017.json"
    VAL_ANNOTATIONS    = DATA_ROOT / "annotations_coco" / "instances_val2017.json"
    TEST_ANNOTATIONS   = DATA_ROOT / "annotations_coco" / "instances_test2017.json"

    NUM_WORKERS        = 4
    PIN_MEMORY         = True
    PERSISTENT_WORKERS = True

    AUGMENT                   = False
    AUGMENT_REPEAT_FACTOR     = 2
    MEDIAN_BLUR_PROB          = 0.2
    MEDIAN_BLUR_LIMIT         = 3
    HORIZONTAL_FLIP_PROB      = 0.5
    VERTICAL_FLIP_PROB        = 0.2
    ROTATE_PROB               = 0.25
    ROTATE_LIMIT              = 12
    RANDOM_RESIZED_CROP_PROB  = 0.2
    RANDOM_RESIZED_CROP_SCALE = (0.85, 1.0)
    SHIFT_SCALE_ROTATE_PROB   = 0.35
    SHIFT_LIMIT               = 0.06
    SCALE_LIMIT               = 0.12
    COLOR_JITTER_PROB         = 0.5
    COLOR_JITTER_BRIGHTNESS   = 0.2
    COLOR_JITTER_CONTRAST     = 0.2
    COLOR_JITTER_SATURATION   = 0.2
    COLOR_JITTER_HUE          = 0.05
    RANDOM_BRIGHTNESS_CONTRAST_PROB = 0.35
    CLAHE_PROB                = 0.15

    CLASSIFIER_HORIZONTAL_FLIP_PROB = 0.5
    CLASSIFIER_VERTICAL_FLIP_PROB = 0.1
    CLASSIFIER_ROTATE_LIMIT = 10
    CLASSIFIER_ROTATE_PROB = 0.25
    CLASSIFIER_COLOR_JITTER_PROB = 0.4
    CLASSIFIER_RANDOM_BRIGHTNESS_CONTRAST_PROB = 0.3
    CLASSIFIER_CLAHE_PROB = 0.1

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    # ==================== Output — SATU FOLDER INDUK ====================
    # Semua output (checkpoint, log, grafik, test result) disimpan di sini.
    # train.py akan membuat subfolder run_YYYYMMDD_HHMMSS/ di dalamnya.
    BASE_OUTPUT_DIR = Path("outputs")

    # Path-path di bawah ini diisi oleh train.py setelah RUN_DIR dibuat.
    # Jangan diubah manual.
    CHECKPOINT_DIR  = BASE_OUTPUT_DIR / "checkpoints"   # fallback jika train.py belum set
    LOG_DIR         = BASE_OUTPUT_DIR / "logs"
    GRAPHS_DIR      = BASE_OUTPUT_DIR / "graphs"
    TEST_RESULT_DIR = BASE_OUTPUT_DIR / "test_results"
    CLASSIFIER_CHECKPOINT_PATH = BASE_OUTPUT_DIR / CLASSIFIER_CHECKPOINT_NAME

    # ==================== Checkpoint & Logging ====================
    SAVE_FREQUENCY         = 5
    KEEP_LAST_N_CHECKPOINTS = 3
    LOG_FREQUENCY          = 10
    USE_TENSORBOARD        = False   # Dimatikan; pakai file PNG

    # ==================== Evaluasi ====================
    EVAL_IOU_THRESHOLDS = [0.5]
    EVAL_FREQUENCY      = 1   # Validasi setiap N epoch
    TRAIN_EVAL_FREQUENCY = 10  # Evaluasi train set setiap N epoch (untuk per-class metrics)
    TRAIN_GRAPH_EVAL_FREQUENCY = 1  # Hitung metrik train untuk titik grafik setiap N epoch
    TEST_VIS_SAMPLES    = 300  # Jumlah gambar visualisasi saat testing
    CHECKPOINT_METRIC   = "mAP@0.50"
    STRICT_TARGET_VALIDATION = True
    SKIP_INVALID_BATCHES = True
    CUDA_DEBUG_SYNC = False
    EMPTY_CACHE_PER_EVAL_BATCH = False

    # ==================== Device ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== Nama Kelas ====================
    COCO_CLASSES = ['moler', 'slabung', 'ulat_grayak']
    CLASS_COLORS = [
        (0, 0, 255),    # moler       -> merah (BGR)
        (0, 255, 0),    # slabung     -> hijau
        (255, 0, 0),    # ulat_grayak -> biru
    ]

    # ==================== Bobot Kelas Loss ====================
    # Diperkuat untuk slabung dan ulat_grayak (lihat loss_fixed.py)
    LOSS_CLASS_ALPHA      = [0.40, 0.30, 0.40]   # focal alpha per kelas
    LOSS_CLASS_MULTIPLIER = [2.4,  1.8,  2.4]    # pengali bobot per kelas

    @classmethod
    def setup_run_dirs(cls, run_dir: Path):
        """Dipanggil dari train.py untuk mengatur semua subfolder output."""
        cls.RUN_DIR        = run_dir
        cls.CHECKPOINT_DIR = run_dir / "checkpoints"
        cls.LOG_DIR        = run_dir / "logs"
        cls.GRAPHS_DIR     = run_dir / "graphs"
        cls.TEST_RESULT_DIR = run_dir / "test_results"
        cls.CLASSIFIER_CHECKPOINT_PATH = cls.CHECKPOINT_DIR / cls.CLASSIFIER_CHECKPOINT_NAME

        for d in [cls.CHECKPOINT_DIR, cls.LOG_DIR,
                  cls.GRAPHS_DIR, cls.TEST_RESULT_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_directories(cls):
        """Buat folder dasar jika belum ada."""
        cls.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("Hybrid CNN-Transformer Object Detection — Config")
        print("=" * 60)
        print(f"Image Size    : {cls.IMAGE_SIZE}×{cls.IMAGE_SIZE}")
        print(f"Num Classes   : {cls.NUM_CLASSES}  {cls.COCO_CLASSES}")
        print(f"Batch Size    : {cls.BATCH_SIZE}")
        print(f"Epochs        : {cls.EPOCHS}")
        print(f"Learning Rate : {cls.LEARNING_RATE}")
        print(f"Device        : {cls.DEVICE}")
        print(f"Augmentasi    : {cls.AUGMENT}")
        print(f"Class Alpha   : {cls.LOSS_CLASS_ALPHA}")
        print(f"Class Mult    : {cls.LOSS_CLASS_MULTIPLIER}")
        print("=" * 60)


config = Config()

if __name__ == "__main__":
    Config.print_config()
