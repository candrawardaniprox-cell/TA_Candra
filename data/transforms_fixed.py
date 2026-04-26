"""
Transform pipelines for train, validation, and inference.

Augmentation on the training pipeline is controlled explicitly by arguments
from config/train.py so hyperparameter experiments can flip it on or off
without editing this file again.
"""

from typing import Callable

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: int = 640,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    augment: bool = False,
    median_blur_prob: float = 0.3,
    median_blur_limit: int = 3,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.2,
    rotate_limit: int = 15,
    rotate_prob: float = 0.4,
    random_resized_crop_prob: float = 0.5,
    random_resized_crop_scale: tuple = (0.75, 1.0),
    shift_scale_rotate_prob: float = 0.35,
    shift_limit: float = 0.06,
    scale_limit: float = 0.12,
    color_jitter_prob: float = 0.4,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.05,
    random_brightness_contrast_prob: float = 0.35,
    clahe_prob: float = 0.15,
):
    """Training transforms with optional augmentation."""
    transforms = []

    if augment:
        transforms.extend([
            A.MedianBlur(blur_limit=median_blur_limit, p=median_blur_prob),
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.VerticalFlip(p=vertical_flip_prob),
            A.Affine(
                translate_percent={
                    "x": (-shift_limit, shift_limit),
                    "y": (-shift_limit, shift_limit),
                },
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                rotate=(-rotate_limit, rotate_limit),
                interpolation=cv2.INTER_CUBIC,
                mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                p=shift_scale_rotate_prob,
            ),
            A.Rotate(
                limit=rotate_limit,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                p=rotate_prob,
            ),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=random_resized_crop_scale,
                interpolation=cv2.INTER_CUBIC,
                p=random_resized_crop_prob,
            ),
            A.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
                p=color_jitter_prob,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=color_jitter_brightness,
                contrast_limit=color_jitter_contrast,
                p=random_brightness_contrast_prob,
            ),
            A.CLAHE(p=clahe_prob),
        ])

    transforms.extend([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.1,
        ),
    )


def get_val_transforms(
    image_size: int = 640,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
):
    """Validation / test transforms."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc', label_fields=['labels'],
    ))


def get_inference_transforms(
    image_size: int = 640,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> Callable:
    """Transform for inference / Streamlit."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=mean, std=std, p=1.0),
        ToTensorV2(),
    ])


def get_classifier_train_transforms(
    image_size: int = 224,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.1,
    rotate_limit: int = 10,
    rotate_prob: float = 0.25,
    color_jitter_prob: float = 0.4,
    random_brightness_contrast_prob: float = 0.3,
    clahe_prob: float = 0.1,
):
    return A.Compose([
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.VerticalFlip(p=vertical_flip_prob),
        A.Rotate(
            limit=rotate_limit,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_CONSTANT,
            p=rotate_prob,
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=color_jitter_prob,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=random_brightness_contrast_prob,
        ),
        A.CLAHE(p=clahe_prob),
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_classifier_val_transforms(
    image_size: int = 224,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
):
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def denormalize_image(
    image,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Convert image back to [0, 1] range for visualization."""
    import numpy as np
    import torch

    if isinstance(image, torch.Tensor):
        image = image.clone()
        for i in range(3):
            image[i] = image[i] * std[i] + mean[i]
        return torch.clamp(image, 0, 1)

    image = image.copy().astype(np.float32)
    for i in range(3):
        image[..., i] = image[..., i] * std[i] + mean[i]
    return np.clip(image, 0, 1)
