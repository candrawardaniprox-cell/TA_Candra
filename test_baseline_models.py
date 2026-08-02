"""Sanity-check untuk semua 3 baseline model."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
cfg.Config.BACKBONE_PRETRAINED = False

import torch

def test_model(name, ModelClass):
    print(f"\nTesting {name}...")
    m = ModelClass(num_classes=3, image_size=640, transformer_dim=256)
    m.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        o = m(x)
    print(f"  logits:     {o['logits'].shape}")
    print(f"  reg_offsets: {o['reg_offsets'].shape}")
    print(f"  centerness: {o['centerness'].shape}")
    print(f"  grids:      {o['grids']}")
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  params:     {params:.2f}M")

    # Test attributes for train.py compatibility
    assert hasattr(m, 'backbone'), "Missing: backbone"
    assert hasattr(m, 'stage_p3'), "Missing: stage_p3"
    assert hasattr(m, 'stage_p4'), "Missing: stage_p4"
    assert hasattr(m, 'stage_p5'), "Missing: stage_p5"
    assert hasattr(m, 'lat_p3'), "Missing: lat_p3"
    assert hasattr(m, 'lat_p4'), "Missing: lat_p4"
    assert hasattr(m, 'smooth_p3'), "Missing: smooth_p3"
    assert hasattr(m, 'smooth_p4'), "Missing: smooth_p4"
    assert hasattr(m, 'detection_head'), "Missing: detection_head"
    print(f"  attributes: OK")

    d = m.get_detections(x, conf_threshold=0.01)
    print(f"  detections: {len(d[0]['boxes'])} boxes")

    cd = m.get_class_oriented_detections(x, conf_threshold=0.01)
    print(f"  class_det:  {len(cd[0]['boxes'])} boxes")

    print(f"  {name} PASSED!")
    del m, x, o, d, cd

print("=" * 60)
print("SANITY CHECK: 3 Baseline Models")
print("=" * 60)

from models.plain_cnn_detector import PlainCNNDetector
from models.resnet_detector import ResNetDetector
from models.vgg_detector import VGGDetector

test_model("PlainCNNDetector", PlainCNNDetector)
test_model("ResNetDetector", ResNetDetector)
test_model("VGGDetector", VGGDetector)

# Test factory function
print("\n\nTesting build_detector factory...")
for model_type in ['hybrid', 'plain_cnn', 'resnet', 'vgg16']:
    cfg.Config.MODEL_TYPE = model_type
    cfg.Config.DETECTOR_USE_BACKBONE = True
    cfg.Config.DETECTOR_USE_CTE = True
    cfg.Config.BACKBONE_NAME = 'resnet18'
    
    from models import HybridDetector
    from train import build_detector
    m = build_detector()
    expected_cls = {
        'hybrid': HybridDetector,
        'plain_cnn': PlainCNNDetector,
        'resnet': ResNetDetector,
        'vgg16': VGGDetector,
    }[model_type]
    assert isinstance(m, expected_cls), f"Factory returned {type(m)} for {model_type}"
    print(f"  MODEL_TYPE='{model_type}' -> {type(m).__name__} OK")
    del m

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
