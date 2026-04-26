"""
Visualization utilities for object detection.

This module provides functions to visualize detection results by drawing
bounding boxes, labels, and confidence scores on images.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from pathlib import Path

from config import Config


# Color palette for different classes (BGR format for OpenCV)
COLORS = list(getattr(Config, 'CLASS_COLORS', [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
]))
LABEL_FONT_SCALE = float(getattr(Config, 'VIS_LABEL_FONT_SCALE', 0.34))
LABEL_THICKNESS = int(getattr(Config, 'VIS_LABEL_THICKNESS', 1))
LABEL_MARGIN = int(getattr(Config, 'VIS_LABEL_MARGIN', 2))
LABEL_OUTLINE_THICKNESS = int(getattr(Config, 'VIS_LABEL_OUTLINE_THICKNESS', 1))
LABEL_BOX_ALPHA = float(getattr(Config, 'VIS_LABEL_BOX_ALPHA', 0.55))
LABEL_TEXT_COLOR = tuple(getattr(Config, 'VIS_LABEL_TEXT_COLOR', (245, 245, 245)))
LABEL_OUTLINE_COLOR = tuple(getattr(Config, 'VIS_LABEL_OUTLINE_COLOR', (20, 20, 20)))


def get_color(class_id: int) -> Tuple[int, int, int]:
    """
    Get a consistent color for a given class ID.

    Args:
        class_id: Class identifier

    Returns:
        BGR color tuple
    """
    return COLORS[class_id % len(COLORS)]


def _resolve_label_and_class_id(
    label: Union[str, int, float, np.generic],
    class_names: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """
    Ubah label prediksi/GT menjadi nama kelas dan class_id yang konsisten.

    Prediksi sering datang sebagai float seperti 1.0 atau 2.0; selama nilainya
    sangat dekat ke integer valid, kita pakai sebagai index kelas resmi.
    """
    numeric_types = (int, float, np.integer, np.floating)

    if class_names is not None and isinstance(label, numeric_types):
        label_float = float(label)
        label_int = int(round(label_float))
        if abs(label_float - label_int) < 1e-6 and 0 <= label_int < len(class_names):
            return class_names[label_int], label_int

    label_text = str(label)
    if class_names is not None:
        for idx, name in enumerate(class_names):
            if label_text == name:
                return name, idx

    return label_text, hash(label_text) % len(COLORS)


def draw_bounding_box(
    image: np.ndarray,
    box: Union[List[float], np.ndarray, torch.Tensor],
    label: str,
    color: Tuple[int, int, int],
    score: Optional[float] = None,
    thickness: int = 2,
    box_format: str = 'cxywh',
) -> np.ndarray:
    """
    Draw a single bounding box on an image.

    Args:
        image: Input image (BGR format)
        box: Bounding box dalam format `cxywh` atau `xyxy`
        label: Class label text
        color: Box color in BGR format
        score: Optional confidence score to display
        thickness: Box line thickness

    Returns:
        Image with bounding box drawn
    """
    # Convert box to numpy if needed
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()

    if box_format == 'xyxy':
        x1, y1, x2, y2 = box
    else:
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    img_h, img_w = image.shape[:2]

    # Selalu jaga bbox tetap berada di dalam gambar.
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    if score is not None:
        text = f"{label} | Acc : {float(score):.2f}"
    else:
        text = label

    # Ukuran teks dibuat tetap dan lebih kecil agar rapi di layar laptop.
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
    )

    # Label diposisikan di tengah bbox. Jika terlalu mepet atas, pindah ke bawah bbox.
    text_x = x1 + max(0, ((x2 - x1) - text_width) // 2)
    text_x = max(0, min(text_x, img_w - text_width - 1))

    above_y = y1 - LABEL_MARGIN
    below_y = y2 + text_height + baseline + LABEL_MARGIN
    if above_y - text_height - baseline >= 0:
        text_y = above_y
    elif below_y < img_h:
        text_y = below_y
    else:
        text_y = min(img_h - baseline - 1, max(text_height + baseline + 1, y1 + text_height + baseline))

    # Tambahkan background gelap semi-transparan agar teks lebih terbaca tanpa outline tebal.
    box_pad_x = 4
    box_pad_y = 3
    rect_x1 = max(0, text_x - box_pad_x)
    rect_y1 = max(0, text_y - text_height - baseline - box_pad_y)
    rect_x2 = min(img_w - 1, text_x + text_width + box_pad_x)
    rect_y2 = min(img_h - 1, text_y + box_pad_y)

    overlay = image.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, LABEL_BOX_ALPHA, image, 1.0 - LABEL_BOX_ALPHA, 0, image)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        LABEL_OUTLINE_COLOR,
        LABEL_OUTLINE_THICKNESS,
        cv2.LINE_AA
    )
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        LABEL_TEXT_COLOR,
        LABEL_THICKNESS,
        cv2.LINE_AA
    )

    return image


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Union[List, np.ndarray, torch.Tensor],
    labels: Union[List[str], np.ndarray, torch.Tensor],
    scores: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    thickness: int = 2,
    box_format: str = 'cxywh',
) -> np.ndarray:
    """
    Draw multiple bounding boxes on an image.

    Args:
        image: Input image (BGR or RGB format, will be converted to BGR)
        boxes: Bounding boxes in format [N, 4] (`cxywh` atau `xyxy`)
        labels: Class labels [N] (either class IDs or names)
        scores: Optional confidence scores [N]
        class_names: Optional list of class names for ID to name mapping
        thickness: Box line thickness

    Returns:
        Image with all bounding boxes drawn
    """
    # Make a copy to avoid modifying original
    image = image.copy()

    # Convert float image to uint8 when needed.
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

    # Convert tensors to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Draw each box
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]

        label_text, class_id = _resolve_label_and_class_id(label, class_names)

        # Get score if available
        score = scores[i] if scores is not None else None

        # Get color for this class
        color = get_color(class_id)

        # Draw the box
        image = draw_bounding_box(
            image, box, label_text, color, score, thickness, box_format=box_format
        )

    return image


def visualize_detections(
    image: Union[np.ndarray, str, Path],
    detections: dict,
    class_names: Optional[List[str]] = None,
    conf_threshold: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> np.ndarray:
    """
    Visualize object detections on an image.

    Args:
        image: Input image (numpy array, file path, or Path object)
        detections: Dictionary containing:
            - 'boxes': [N, 4] bounding boxes
            - 'scores': [N] confidence scores
            - 'classes': [N] class predictions
        class_names: List of class names for display
        conf_threshold: Minimum confidence to display
        save_path: Optional path to save the visualization
        show: Whether to display the image using cv2.imshow

    Returns:
        Image with visualizations
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Failed to load image from {image}")

    # Filter by confidence threshold
    if conf_threshold > 0:
        mask = detections['scores'] >= conf_threshold
        boxes = detections['boxes'][mask]
        scores = detections['scores'][mask]
        classes = detections['classes'][mask]
    else:
        boxes = detections['boxes']
        scores = detections['scores']
        classes = detections['classes']

    # Draw boxes
    result = draw_bounding_boxes(
        image, boxes, classes, scores, class_names, box_format='xyxy'
    )

    # Save if requested
    if save_path is not None:
        cv2.imwrite(str(save_path), result)

    # Show if requested
    if show:
        cv2.imshow("Detections", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def create_detection_grid(
    images: List[np.ndarray],
    detections_list: List[dict],
    class_names: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    conf_threshold: float = 0.0
) -> np.ndarray:
    """
    Create a grid of images with detections.

    Args:
        images: List of images
        detections_list: List of detection dictionaries (one per image)
        class_names: List of class names
        grid_size: Optional (rows, cols) for grid layout
        conf_threshold: Minimum confidence to display

    Returns:
        Grid image
    """
    n_images = len(images)

    # Determine grid size if not provided
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    # Visualize each image
    vis_images = []
    for image, detections in zip(images, detections_list):
        vis = visualize_detections(
            image, detections, class_names, conf_threshold
        )
        vis_images.append(vis)

    # Pad with blank images if needed
    while len(vis_images) < rows * cols:
        blank = np.zeros_like(vis_images[0])
        vis_images.append(blank)

    # Get image dimensions
    h, w = vis_images[0].shape[:2]

    # Create grid
    grid_rows = []
    for i in range(rows):
        row_images = vis_images[i * cols:(i + 1) * cols]
        grid_row = np.hstack(row_images)
        grid_rows.append(grid_row)

    grid = np.vstack(grid_rows)

    return grid


def draw_anchor_boxes(
    image: np.ndarray,
    grid_size: int,
    anchors: List[Tuple[float, float]],
    image_size: int = 320
) -> np.ndarray:
    """
    Visualize anchor boxes on a grid.

    Useful for understanding the anchor box configuration.

    Args:
        image: Background image
        grid_size: Size of detection grid
        anchors: List of (width, height) anchor box dimensions (relative)
        image_size: Image size

    Returns:
        Image with anchor boxes drawn
    """
    image = image.copy()

    # Calculate grid cell size
    cell_size = image_size / grid_size

    # Draw grid
    for i in range(grid_size + 1):
        pos = int(i * cell_size)
        cv2.line(image, (pos, 0), (pos, image_size), (128, 128, 128), 1)
        cv2.line(image, (0, pos), (image_size, pos), (128, 128, 128), 1)

    # Draw anchor boxes at center of some cells (for visualization)
    sample_cells = [(i, j) for i in range(0, grid_size, 2) for j in range(0, grid_size, 2)]

    for cell_y, cell_x in sample_cells[:20]:  # Limit to 20 cells for clarity
        cell_center_x = int((cell_x + 0.5) * cell_size)
        cell_center_y = int((cell_y + 0.5) * cell_size)

        # Draw each anchor at this cell
        for anchor_idx, (anchor_w, anchor_h) in enumerate(anchors):
            # Convert relative dimensions to pixels
            box_w = int(anchor_w * image_size)
            box_h = int(anchor_h * image_size)

            x1 = cell_center_x - box_w // 2
            y1 = cell_center_y - box_h // 2
            x2 = cell_center_x + box_w // 2
            y2 = cell_center_y + box_h // 2

            color = get_color(anchor_idx)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image


def test_visualization():
    """Test function to verify visualization utilities."""
    print("Testing visualization utilities...")

    # Create a dummy image
    image = np.ones((320, 320, 3), dtype=np.uint8) * 255

    # Create dummy detections
    detections = {
        'boxes': torch.tensor([
            [160, 160, 100, 100],
            [80, 80, 60, 60],
            [240, 240, 80, 80]
        ]),
        'scores': torch.tensor([0.95, 0.87, 0.92]),
        'classes': torch.tensor([0, 1, 2])
    }

    class_names = ['person', 'car', 'dog']

    # Test visualization
    vis_image = visualize_detections(
        image, detections, class_names, conf_threshold=0.5
    )

    print(f"Visualized image shape: {vis_image.shape}")

    # Test anchor box visualization
    anchors = [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)]
    anchor_vis = draw_anchor_boxes(image.copy(), grid_size=20, anchors=anchors)
    print(f"Anchor visualization shape: {anchor_vis.shape}")

    print("✓ Visualization test passed!")


if __name__ == "__main__":
    test_visualization()
