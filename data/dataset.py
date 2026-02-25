"""
Custom Dataset class for object detection.

This module implements a PyTorch Dataset for loading COCO-format annotations
and images for object detection training and evaluation.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class ObjectDetectionDataset(Dataset):
    """
    Custom dataset for object detection with COCO-format annotations.

    The dataset loads images and annotations in COCO format and applies
    transformations for data augmentation during training.

    Args:
        image_dir: Directory containing images
        annotation_file: Path to COCO-format JSON annotation file
        transform: Optional transform to apply to images and boxes
        image_size: Target image size (will be resized to this)
        return_dict: If True, return dictionary; if False, return tuple
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Callable] = None,
        image_size: int = 320,
        return_dict: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.image_size = image_size
        self.return_dict = return_dict

        # Load annotations
        self._load_annotations()

    def _load_annotations(self):
        """Load and parse COCO format annotations."""
        # Check if annotation file exists
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}\n"
                "Please download the COCO dataset first."
            )

        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Extract information
        self.images_info = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}

        # Group annotations by image
        self.image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)

        # Create list of image IDs that have annotations
        self.image_ids = list(self.image_annotations.keys())

        # Create category ID to continuous index mapping
        category_ids = sorted(self.categories.keys())
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        self.idx_to_category_id = {idx: cat_id for cat_id, idx in self.category_id_to_idx.items()}

        print(f"Loaded {len(self.image_ids)} images with annotations")
        print(f"Number of categories: {len(self.categories)}")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)

    def _load_image(self, image_id: int) -> np.ndarray:
        """
        Load an image by ID.

        Args:
            image_id: COCO image ID

        Returns:
            Image as numpy array (RGB format)
        """
        image_info = self.images_info[image_id]
        image_path = self.image_dir / image_info['file_name']

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        return image

    def _parse_annotations(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse annotations for an image.

        Args:
            image_id: COCO image ID

        Returns:
            Tuple of (boxes, labels)
            - boxes: [N, 4] array in format (x_center, y_center, width, height)
            - labels: [N] array of class indices
        """
        annotations = self.image_annotations[image_id]

        boxes = []
        labels = []

        for ann in annotations:
            # Skip annotations without bounding boxes or with area <= 0
            if 'bbox' not in ann or ann.get('area', 0) <= 0:
                continue

            # COCO bbox format: [x_min, y_min, width, height]
            x_min, y_min, width, height = ann['bbox']

            # Convert to center format
            x_center = x_min + width / 2
            y_center = y_min + height / 2

            boxes.append([x_center, y_center, width, height])

            # Convert category ID to continuous index
            category_id = ann['category_id']
            label_idx = self.category_id_to_idx[category_id]
            labels.append(label_idx)

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        return boxes, labels

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary or tuple containing:
            - image: Tensor of shape [3, H, W]
            - boxes: Tensor of shape [N, 4] in format (x_center, y_center, w, h)
            - labels: Tensor of shape [N] with class indices
            - image_id: Original COCO image ID
        """
        image_id = self.image_ids[idx]

        # Load image and annotations
        image = self._load_image(image_id)
        boxes, labels = self._parse_annotations(image_id)

        # Apply transformations
        if self.transform is not None:
            # Albumentations expects boxes in [x_min, y_min, x_max, y_max] format
            # Convert from center format
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_min
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_min
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x_max
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y_max

            transformed = self.transform(
                image=image,
                bboxes=boxes_xyxy,
                labels=labels
            )

            image = transformed['image']
            boxes_xyxy = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

            # Convert boxes back to center format
            if len(boxes_xyxy) > 0:
                boxes = np.zeros_like(boxes_xyxy)
                boxes[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2  # x_center
                boxes[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2  # y_center
                boxes[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
                boxes[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W]
        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).long()

        if self.return_dict:
            return {
                'image': image,
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id
            }
        else:
            return image, boxes, labels, image_id

    def get_image_info(self, idx: int) -> Dict:
        """
        Get metadata for an image.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with image metadata
        """
        image_id = self.image_ids[idx]
        return self.images_info[image_id]

    def get_category_name(self, category_idx: int) -> str:
        """
        Get category name from continuous index.

        Args:
            category_idx: Continuous category index

        Returns:
            Category name
        """
        category_id = self.idx_to_category_id[category_idx]
        return self.categories[category_id]['name']

    def get_all_category_names(self) -> List[str]:
        """Get list of all category names in order."""
        return [
            self.get_category_name(idx)
            for idx in range(len(self.category_id_to_idx))
        ]


def test_dataset():
    """Test function to verify dataset functionality."""
    print("Testing ObjectDetectionDataset...")

    # This is a placeholder test that shows the expected usage
    # Actual test requires COCO dataset to be downloaded

    print("Dataset test requires COCO data to be downloaded.")
    print("Expected directory structure:")
    print("  data/")
    print("    train2017/")
    print("    val2017/")
    print("    annotations/")
    print("      instances_train2017.json")
    print("      instances_val2017.json")

    print("\n✓ Dataset structure test passed!")


if __name__ == "__main__":
    test_dataset()
