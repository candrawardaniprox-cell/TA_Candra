"""
Data loading utilities for object detection.

This module provides utility functions for creating dataloaders and
handling batching with variable-size bounding boxes.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, Dict[str, List]]:
    """
    Custom collate function for batching detection data.

    Since each image can have a different number of objects, we need a custom
    collate function to handle variable-length bounding box lists.

    Args:
        batch: List of dictionaries from dataset, each containing:
            - image: Tensor [3, H, W]
            - boxes: Tensor [N, 4] (different N for each image)
            - labels: Tensor [N]
            - image_id: int

    Returns:
        Tuple of:
        - images: Batched image tensor [B, 3, H, W]
        - targets: Dictionary with lists:
            - boxes: List of [N_i, 4] tensors
            - labels: List of [N_i] tensors
            - image_ids: List of image IDs
    """
    images = []
    boxes_list = []
    labels_list = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        boxes_list.append(item['boxes'])
        labels_list.append(item['labels'])
        image_ids.append(item['image_id'])

    # Stack images into a batch
    images = torch.stack(images, dim=0)

    # Keep boxes and labels as lists (variable length)
    targets = {
        'boxes': boxes_list,
        'labels': labels_list,
        'image_ids': image_ids
    }

    return images, targets


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for both dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )

    return train_loader, val_loader


def download_coco_subset(
    output_dir: str = "data",
    split: str = "val",
    num_images: int = 100
):
    """
    Download a small subset of COCO dataset for testing.

    This is a helper function to download a subset of COCO for quick testing.
    For full training, please download the complete COCO dataset.

    Args:
        output_dir: Directory to save the data
        split: Dataset split ('train' or 'val')
        num_images: Number of images to download

    Note:
        This requires fiftyone library: pip install fiftyone
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        from pathlib import Path

        print(f"Downloading {num_images} COCO {split} images...")

        # Download dataset
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            max_samples=num_images,
            dataset_dir=output_dir
        )

        print(f"✓ Downloaded {len(dataset)} images to {output_dir}")
        print(f"Dataset info: {dataset}")

    except ImportError:
        print("Error: fiftyone library not installed.")
        print("Install with: pip install fiftyone")
        print("\nAlternatively, download COCO dataset manually from:")
        print("  http://images.cocodataset.org/zips/train2017.zip")
        print("  http://images.cocodataset.org/zips/val2017.zip")
        print("  http://images.cocodataset.org/annotations/annotations_trainval2017.zip")


def calculate_dataset_statistics(dataset, num_samples: int = 100) -> Dict:
    """
    Calculate statistics about the dataset.

    Useful for understanding data distribution and class balance.

    Args:
        dataset: Dataset instance
        num_samples: Number of samples to analyze

    Returns:
        Dictionary with statistics
    """
    from collections import Counter

    num_samples = min(num_samples, len(dataset))
    class_counts = Counter()
    num_objects_per_image = []
    box_sizes = []

    print(f"Analyzing {num_samples} samples...")

    for i in range(num_samples):
        sample = dataset[i]
        boxes = sample['boxes']
        labels = sample['labels']

        num_objects_per_image.append(len(labels))

        for label in labels.tolist():
            class_counts[label] += 1

        for box in boxes:
            # Calculate box area
            width, height = box[2].item(), box[3].item()
            box_sizes.append(width * height)

    stats = {
        'num_samples_analyzed': num_samples,
        'avg_objects_per_image': sum(num_objects_per_image) / len(num_objects_per_image),
        'max_objects_per_image': max(num_objects_per_image),
        'min_objects_per_image': min(num_objects_per_image),
        'total_objects': sum(num_objects_per_image),
        'class_distribution': dict(class_counts.most_common(10)),
        'avg_box_area': sum(box_sizes) / len(box_sizes) if box_sizes else 0,
    }

    return stats


def print_dataset_info(dataset):
    """
    Print detailed information about the dataset.

    Args:
        dataset: Dataset instance
    """
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"Number of images: {len(dataset)}")

    # Get category names
    category_names = dataset.get_all_category_names()
    print(f"Number of categories: {len(category_names)}")
    print(f"Categories: {', '.join(category_names[:10])}...")

    # Calculate statistics
    stats = calculate_dataset_statistics(dataset, num_samples=min(100, len(dataset)))

    print("\nDataset Statistics:")
    print(f"  Average objects per image: {stats['avg_objects_per_image']:.2f}")
    print(f"  Max objects in an image: {stats['max_objects_per_image']}")
    print(f"  Min objects in an image: {stats['min_objects_per_image']}")
    print(f"  Total objects analyzed: {stats['total_objects']}")
    print(f"  Average box area: {stats['avg_box_area']:.2f}")

    print("\nTop 10 most common classes:")
    for class_id, count in stats['class_distribution'].items():
        class_name = dataset.get_category_name(class_id)
        print(f"  {class_name}: {count} objects")

    print("=" * 60)


def test_data_utils():
    """Test function to verify data utilities."""
    print("Testing data utilities...")

    # Create dummy batch data
    batch = [
        {
            'image': torch.randn(3, 320, 320),
            'boxes': torch.tensor([[100, 100, 50, 50], [200, 200, 60, 60]]),
            'labels': torch.tensor([0, 1]),
            'image_id': 1
        },
        {
            'image': torch.randn(3, 320, 320),
            'boxes': torch.tensor([[150, 150, 70, 70]]),
            'labels': torch.tensor([2]),
            'image_id': 2
        }
    ]

    # Test collate function
    images, targets = collate_fn(batch)

    print(f"Batched images shape: {images.shape}")
    print(f"Number of targets: {len(targets['boxes'])}")
    print(f"Boxes per image: {[len(b) for b in targets['boxes']]}")
    print(f"Labels per image: {[len(l) for l in targets['labels']]}")

    assert images.shape == (2, 3, 320, 320), "Unexpected batch shape"
    assert len(targets['boxes']) == 2, "Unexpected number of box tensors"
    assert len(targets['boxes'][0]) == 2, "Unexpected number of boxes in first image"
    assert len(targets['boxes'][1]) == 1, "Unexpected number of boxes in second image"

    print("\n✓ Data utils test passed!")


if __name__ == "__main__":
    test_data_utils()
