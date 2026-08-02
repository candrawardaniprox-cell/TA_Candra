"""
Models package untuk Object Detection.

Berisi 4 arsitektur detector:
  1. HybridDetector        : CNN-Transformer Hybrid (Vikhe et al., 2025)
  2. PlainCNNDetector      : Plain CNN baseline (LeCun et al., 1998)
  3. ResNetDetector        : ResNet-18 baseline (He et al., CVPR 2016)
  4. VGGDetector           : VGG-16-BN baseline (Simonyan & Zisserman, ICLR 2015)

Komponen utama CNN-Transformer (sesuai paper Vikhe et al., 2025):
  - CTE   : Convolution Token Embedding
  - CPSA  : Convolutional Parameter-Sharing Multi-Head Attention
  - LFFN  : Local Feed-Forward Network (sandglass DSC)
  - HybridTransformerBlock : satu unit encoder (CPSA + LFFN)
  - HybridStage            : CTE + N × HybridTransformerBlock
  - HybridDetector         : model lengkap untuk deteksi objek
"""
from __future__ import annotations

from .transformer import CTE, CPSA, LFFN, HybridTransformerBlock
from .hybrid_model import HybridDetector, HybridStage
from .detection_head import AnchorFreeDetectionHead
from .backbone import DynamicCNNBackbone
from .paper_classifier import PaperDiseaseClassifier, hierarchical_classifier_loss
from .detector_base import BaseDetector
from .plain_cnn_detector import PlainCNNDetector
from .resnet_detector import ResNetDetector
from .vgg_detector import VGGDetector
from .mobilenet_detector import MobileNetDetector

__all__ = [
    # Komponen transformer (paper)
    'CTE',
    'CPSA',
    'LFFN',
    'HybridTransformerBlock',
    # Stage hierarkis
    'HybridStage',
    # Model utama (5 arsitektur)
    'HybridDetector',
    'PlainCNNDetector',
    'ResNetDetector',
    'VGGDetector',
    'MobileNetDetector',
    # Base class
    'BaseDetector',
    # Kepala deteksi
    'AnchorFreeDetectionHead',
    # Backbone
    'DynamicCNNBackbone',
    'PaperDiseaseClassifier',
    'hierarchical_classifier_loss',
]
