"""
Data augmentation transforms for object detection.

This module provides transform pipelines using Albumentations library,
which properly handles bounding box transformations during augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Callable


def get_train_transforms(
    image_size: int = 320,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    h_flip_prob: float = 0.5,
    brightness_contrast_limit: float = 0.2,
    hue_saturation_limit: int = 20
) -> Callable:
    """
    Get training data augmentation pipeline.

    The pipeline includes:
    - Random horizontal flip
    - Random brightness/contrast adjustment
    - Random hue/saturation/value adjustment
    - Resize to target size
    - Normalization
    - Conversion to tensor

    All transforms properly handle bounding box coordinates.

    Args:
        image_size: Target image size (square)
        mean: Mean values for normalization (ImageNet default)
        std: Standard deviation for normalization (ImageNet default)
        h_flip_prob: Probability of horizontal flip
        brightness_contrast_limit: Limit for brightness/contrast adjustment
        hue_saturation_limit: Limit for HSV adjustment

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=h_flip_prob),

        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=brightness_contrast_limit,
            contrast_limit=brightness_contrast_limit,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=hue_saturation_limit,
            sat_shift_limit=hue_saturation_limit,
            val_shift_limit=hue_saturation_limit,
            p=0.5
        ),

        # Additional augmentations for robustness
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.GaussianBlur(p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.2),

        # Resize to target size
        A.Resize(height=image_size, width=image_size, p=1.0),

        # Normalization
        A.Normalize(mean=mean, std=std, p=1.0),
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',  # [x_min, y_min, x_max, y_max]
        label_fields=['labels'],
        min_visibility=0.3,  # Remove boxes with < 30% visibility after augmentation
        min_area=0.0
    ))


def get_val_transforms(
    image_size: int = 320,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
) -> Callable:
    """
    Get validation/test data transform pipeline.

    No augmentation, only resize and normalization.

    Args:
        image_size: Target image size (square)
        mean: Mean values for normalization
        std: Standard deviation for normalization

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        # Resize to target size
        A.Resize(height=image_size, width=image_size, p=1.0),

        # Normalization
        A.Normalize(mean=mean, std=std, p=1.0),
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',  # [x_min, y_min, x_max, y_max]
        label_fields=['labels'],
        min_visibility=0.0,
        min_area=0.0
    ))


def get_inference_transforms(
    image_size: int = 320,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
) -> Callable:
    """
    Get inference transform pipeline.

    Similar to validation but without bounding box parameters.

    Args:
        image_size: Target image size
        mean: Mean values for normalization
        std: Standard deviation for normalization

    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=mean, std=std, p=1.0),
    ])


def denormalize_image(
    image,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
):
    """
    Denormalize an image for visualization.

    Args:
        image: Normalized image tensor [C, H, W] or [H, W, C]
        mean: Mean values used for normalization
        std: Standard deviation used for normalization

    Returns:
        Denormalized image
    """
    import torch
    import numpy as np

    # Handle tensor input
    if isinstance(image, torch.Tensor):
        image = image.clone()
        # Check if channel-first [C, H, W]
        if image.shape[0] == 3:
            for i in range(3):
                image[i] = image[i] * std[i] + mean[i]
        else:  # Channel-last [H, W, C]
            for i in range(3):
                image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        image = torch.clamp(image, 0, 1)
        return image

    # Handle numpy input
    elif isinstance(image, np.ndarray):
        image = image.copy()
        # Check if channel-first [C, H, W]
        if image.shape[0] == 3:
            for i in range(3):
                image[i] = image[i] * std[i] + mean[i]
        else:  # Channel-last [H, W, C]
            for i in range(3):
                image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        image = np.clip(image, 0, 1)
        return image

    return image


def test_transforms():
    """Test function to verify transforms."""
    print("Testing transforms...")

    import numpy as np

    # Create dummy image and boxes
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([
        [100, 100, 200, 200],  # [x_min, y_min, x_max, y_max]
        [300, 300, 400, 400]
    ], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    print(f"Original image shape: {image.shape}")
    print(f"Original boxes:\n{boxes}")

    # Test training transforms
    train_transform = get_train_transforms(image_size=320)
    transformed = train_transform(image=image, bboxes=boxes, labels=labels)

    print(f"\nAfter training transforms:")
    print(f"  Image shape: {transformed['image'].shape}")
    print(f"  Number of boxes: {len(transformed['bboxes'])}")
    if len(transformed['bboxes']) > 0:
        print(f"  Boxes:\n{np.array(transformed['bboxes'])}")

    # Test validation transforms
    val_transform = get_val_transforms(image_size=320)
    transformed = val_transform(image=image, bboxes=boxes, labels=labels)

    print(f"\nAfter validation transforms:")
    print(f"  Image shape: {transformed['image'].shape}")
    print(f"  Number of boxes: {len(transformed['bboxes'])}")

    # Test inference transforms
    inf_transform = get_inference_transforms(image_size=320)
    transformed = inf_transform(image=image)

    print(f"\nAfter inference transforms:")
    print(f"  Image shape: {transformed['image'].shape}")

    print("\n✓ Transforms test passed!")


if __name__ == "__main__":
    test_transforms()
