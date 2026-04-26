"""
Dataset crop penyakit dari anotasi COCO untuk stage-2 classifier.

Setiap anotasi bbox menjadi satu sampel crop. Crop diberi sedikit konteks di
sekitar bbox agar classifier melihat pola penyakit dan bagian daun di sekitarnya.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DiseaseCropDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Callable] = None,
        crop_padding: float = 0.15,
        min_crop_size: int = 12,
        repeat_factor: int = 1,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.crop_padding = max(0.0, float(crop_padding))
        self.min_crop_size = max(2, int(min_crop_size))
        self.repeat_factor = max(1, int(repeat_factor))

        self._load_annotations()

    def _load_annotations(self) -> None:
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file tidak ditemukan: {self.annotation_file}")

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.images_info = {img["id"]: img for img in coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in coco_data["categories"]}
        category_ids = sorted(self.categories.keys())
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

        samples = []
        for ann in coco_data["annotations"]:
            if "bbox" not in ann or ann.get("area", 0) <= 0:
                continue
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            samples.append(
                {
                    "image_id": ann["image_id"],
                    "bbox_xywh": [float(x), float(y), float(w), float(h)],
                    "label": self.category_id_to_idx[ann["category_id"]],
                }
            )

        if not samples:
            raise RuntimeError("Tidak ada anotasi bbox yang valid untuk classifier crops.")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_factor

    def _load_image(self, image_id: int) -> np.ndarray:
        image_info = self.images_info[image_id]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    def _crop_with_context(self, image: np.ndarray, bbox_xywh: list[float]) -> np.ndarray:
        x, y, w, h = bbox_xywh
        img_h, img_w = image.shape[:2]

        pad_x = w * self.crop_padding
        pad_y = h * self.crop_padding
        x1 = max(0, int(np.floor(x - pad_x)))
        y1 = max(0, int(np.floor(y - pad_y)))
        x2 = min(img_w, int(np.ceil(x + w + pad_x)))
        y2 = min(img_h, int(np.ceil(y + h + pad_y)))

        if (x2 - x1) < self.min_crop_size:
            extra = self.min_crop_size - (x2 - x1)
            x1 = max(0, x1 - extra // 2)
            x2 = min(img_w, x2 + extra - extra // 2)
        if (y2 - y1) < self.min_crop_size:
            extra = self.min_crop_size - (y2 - y1)
            y1 = max(0, y1 - extra // 2)
            y2 = min(img_h, y2 + extra - extra // 2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError("Crop kosong terdeteksi.")
        return crop

    def __getitem__(self, idx: int) -> dict:
        real_idx = idx % len(self.samples)
        sample = self.samples[real_idx]
        image = self._load_image(sample["image_id"])
        crop = self._crop_with_context(image, sample["bbox_xywh"])
        label = int(sample["label"])

        if self.transform is not None:
            transformed = self.transform(image=crop)
            crop = transformed["image"]

        if isinstance(crop, np.ndarray):
            crop = torch.from_numpy(crop).permute(2, 0, 1).float()
        else:
            crop = crop.float()

        return {
            "image": crop,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": sample["image_id"],
            "bbox_xywh": torch.tensor(sample["bbox_xywh"], dtype=torch.float32),
        }
