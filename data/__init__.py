"""
Data module for object detection dataset handling.

This module contains:
- dataset: Custom dataset class for COCO format data
- transforms: Augmentation pipeline using Albumentations
- utils: Data loading utilities and collate functions
"""

from .dataset import ObjectDetectionDataset
from .transforms import get_train_transforms, get_val_transforms
from .utils import collate_fn, create_dataloaders

__all__ = [
    'ObjectDetectionDataset',
    'get_train_transforms',
    'get_val_transforms',
    'collate_fn',
    'create_dataloaders',
]
