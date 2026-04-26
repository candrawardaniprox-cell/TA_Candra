"""
dataset.py — Perbaikan Bug #3:
- HAPUS paksa-expand bbox ke 16px (merusak ground truth).
- Ganti dengan FILTER: bbox yang terlalu kecil (< 4px) dibuang, bukan digelembungkan.
- Perbaikan minor: konversi ke tensor lebih aman.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        image_dir:       str,
        annotation_file: str,
        transform:       Optional[Callable] = None,
        image_size:      int  = 640,
        return_dict:     bool = True,
        min_box_size:    int  = 4,   # Hapus bbox lebih kecil dari N piksel (setelah resize)
        repeat_factor:   int  = 1,
    ):
        self.image_dir       = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.transform       = transform
        self.image_size      = image_size
        self.return_dict     = return_dict
        self.min_box_size    = min_box_size
        self.repeat_factor   = max(1, int(repeat_factor))

        self._load_annotations()

    def _load_annotations(self):
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file tidak ditemukan: {self.annotation_file}")

        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images_info     = {img['id']: img for img in coco_data['images']}
        self.categories      = {cat['id']: cat for cat in coco_data['categories']}

        self.image_annotations = {}
        for ann in coco_data['annotations']:
            iid = ann['image_id']
            self.image_annotations.setdefault(iid, []).append(ann)

        self.image_ids = list(self.image_annotations.keys())

        category_ids            = sorted(self.categories.keys())
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        self.idx_to_category_id = {idx: cat_id for cat_id, idx in self.category_id_to_idx.items()}

        print(f"Loaded {len(self.image_ids)} images | {len(self.categories)} categories")

    def __len__(self) -> int:
        return len(self.image_ids) * self.repeat_factor

    def __getitem__(self, idx: int) -> Dict:
        import random

        max_retries = 5
        candidate_idx = idx
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                return self._load_sample(candidate_idx)
            except Exception as e:
                last_error = e
                real_idx = candidate_idx % len(self.image_ids)
                image_id = self.image_ids[real_idx] if self.image_ids else None
                print(
                    f"\nWarning: Gagal memuat index {candidate_idx} "
                    f"(image_id={image_id}, percobaan {attempt}/{max_retries}): {e}. "
                    "Mengganti dengan sample lain."
                )
                candidate_idx = random.randint(0, len(self) - 1)

        raise RuntimeError(
            f"Gagal memuat sample setelah {max_retries} percobaan. "
            f"Index awal={idx}. Error terakhir: {last_error}"
        )

    def _load_image(self, image_id: int) -> np.ndarray:
        info       = self.images_info[image_id]
        image_path = self.image_dir / info['file_name']
        with Image.open(image_path) as image:
            return np.array(image.convert('RGB'))

    def _parse_annotations(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        annotations = self.image_annotations[image_id]
        boxes, labels = [], []
        for ann in annotations:
            if 'bbox' not in ann or ann.get('area', 0) <= 0:
                continue
            x_min, y_min, width, height = ann['bbox']
            # Abaikan bbox yang sangat kecil (noise annotasi)
            if width < 1 or height < 1:
                continue
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            boxes.append([x_center, y_center, width, height])
            labels.append(self.category_id_to_idx[ann['category_id']])

        if boxes:
            return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    def _load_sample(self, idx: int) -> Dict:
        try:
            real_idx  = idx % len(self.image_ids)
            image_id  = self.image_ids[real_idx]
            image     = self._load_image(image_id)
            boxes, labels = self._parse_annotations(image_id)

            if self.transform is not None and len(boxes) > 0:
                # Konversi ke xyxy untuk Albumentations
                boxes_xyxy       = np.zeros_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

                # Clip agar tidak keluar batas gambar
                img_h, img_w = image.shape[:2]
                if boxes_xyxy.max() <= 1.5:  # format normalized
                    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, 1.0)
                    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, 1.0)
                else:                         # format piksel
                    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, img_w)
                    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, img_h)

                transformed = self.transform(
                    image=image, bboxes=boxes_xyxy, labels=labels
                )
                image      = transformed['image']
                boxes_xyxy = np.array(transformed['bboxes'], dtype=np.float32)
                labels     = np.array(transformed['labels'], dtype=np.int64)

                if len(boxes_xyxy) > 0:
                    # Kembalikan ke format cxywh (format GT yang dipakai oleh loss.py)
                    boxes       = np.zeros_like(boxes_xyxy)
                    boxes[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2  # cx
                    boxes[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2  # cy
                    boxes[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]         # w
                    boxes[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]         # h

                    # BUG #3 FIX: FILTER bbox kecil, jangan diperbesar!
                    # Paksa expand ke 16px merusak ground truth (bbox palsu)
                    valid_mask = (boxes[:, 2] >= self.min_box_size) & \
                                 (boxes[:, 3] >= self.min_box_size)
                    boxes  = boxes[valid_mask]
                    labels = labels[valid_mask]
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)

            elif self.transform is not None and len(boxes) == 0:
                # Tidak ada bbox, tetap transform gambar saja
                transformed = self.transform(image=image, bboxes=[], labels=[])
                image       = transformed['image']

            # Final sanitization agar target selalu aman sebelum masuk dataloader/loss
            if len(boxes) > 0:
                finite_mask = np.isfinite(boxes).all(axis=1)
                size_mask = (boxes[:, 2] > 1e-6) & (boxes[:, 3] > 1e-6)
                label_mask = (labels >= 0) & (labels < len(self.category_id_to_idx))
                valid_mask = finite_mask & size_mask & label_mask
                boxes = boxes[valid_mask]
                labels = labels[valid_mask]

                if len(boxes) > 0:
                    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, self.image_size)
                    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, self.image_size)
                    boxes[:, 2] = np.clip(boxes[:, 2], 1e-6, self.image_size)
                    boxes[:, 3] = np.clip(boxes[:, 3], 1e-6, self.image_size)

            # Konversi ke tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = image.float()

            boxes  = torch.from_numpy(boxes).float()
            labels = torch.from_numpy(labels).long()

            if self.return_dict:
                return {'image': image, 'boxes': boxes, 'labels': labels, 'image_id': image_id}
            return image, boxes, labels, image_id

        except Exception as e:
            raise RuntimeError(f"Gagal memuat index {idx}: {e}") from e

    def get_image_info(self, idx):
        return self.images_info[self.image_ids[idx]]

    def get_category_name(self, category_idx):
        return self.categories[self.idx_to_category_id[category_idx]]['name']

    def get_all_category_names(self):
        return [self.get_category_name(i) for i in range(len(self.category_id_to_idx))]
