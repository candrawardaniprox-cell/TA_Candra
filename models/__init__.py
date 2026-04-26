"""
Models package untuk Hybrid CNN-Transformer Object Detector.

Komponen utama (sesuai paper Vikhe et al., 2025):
  - CTE   : Convolution Token Embedding
  - CPSA  : Convolutional Parameter-Sharing Multi-Head Attention
  - LFFN  : Local Feed-Forward Network (sandglass DSC)
  - HybridTransformerBlock : satu unit encoder (CPSA + LFFN)
  - HybridStage            : CTE + N × HybridTransformerBlock
  - HybridDetector         : model lengkap untuk deteksi objek
"""

from .transformer import CTE, CPSA, LFFN, HybridTransformerBlock
from .hybrid_model import HybridDetector, HybridStage
from .detection_head import AnchorFreeDetectionHead
from .backbone import DynamicCNNBackbone
from .paper_classifier import PaperDiseaseClassifier, hierarchical_classifier_loss

__all__ = [
    # Komponen transformer (paper)
    'CTE',
    'CPSA',
    'LFFN',
    'HybridTransformerBlock',
    # Stage hierarkis
    'HybridStage',
    # Model utama
    'HybridDetector',
    # Kepala deteksi
    'AnchorFreeDetectionHead',
    # Backbone
    'DynamicCNNBackbone',
    'PaperDiseaseClassifier',
    'hierarchical_classifier_loss',
]
