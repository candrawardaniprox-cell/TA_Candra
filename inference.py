"""
Inference script for Hybrid CNN-Transformer Object Detection.

This script provides functions to run inference on single images or batches
of images using a trained model.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict
import argparse

from config import Config
from models import HybridDetector
from data import get_inference_transforms
from utils import visualize_detections


class ObjectDetector:
    """
    Wrapper class for object detection inference.

    This class handles model loading, preprocessing, inference, and
    post-processing for easy use.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on ('cuda' or 'cpu')
        conf_threshold: Confidence threshold for detections
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections per image
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 100
    ):
        self.device = device if device else Config.DEVICE
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Setup transform
        self.transform = get_inference_transforms(
            image_size=Config.IMAGE_SIZE,
            mean=Config.MEAN,
            std=Config.STD
        )

        print("Model loaded successfully!")

    def _load_model(self, checkpoint_path: str) -> HybridDetector:
        """Load model from checkpoint."""
        model = HybridDetector(
            num_classes=Config.NUM_CLASSES,
            image_size=Config.IMAGE_SIZE,
            backbone_channels=Config.BACKBONE_CHANNELS,
            transformer_dim=Config.TRANSFORMER_DIM,
            transformer_heads=Config.TRANSFORMER_HEADS,
            transformer_layers=Config.TRANSFORMER_LAYERS,
            transformer_ff_dim=Config.TRANSFORMER_FF_DIM,
            num_anchors=Config.NUM_ANCHORS,
            anchors=Config.ANCHOR_BOXES,
            dropout=Config.TRANSFORMER_DROPOUT
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model

    def preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: Input image (path, numpy array, or PIL Image)

        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        # Apply transform
        transformed = self.transform(image=image)
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).float()

        return image_tensor

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_image: bool = False
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: Input image
            return_image: Whether to return the original image

        Returns:
            Dictionary containing:
            - boxes: [N, 4] bounding boxes (x_center, y_center, width, height)
            - scores: [N] confidence scores
            - classes: [N] predicted class indices
            - class_names: [N] predicted class names
            - image: Original image (if return_image=True)
        """
        # Store original image if needed
        if return_image:
            if isinstance(image, (str, Path)):
                orig_image = np.array(Image.open(image).convert('RGB'))
            elif isinstance(image, Image.Image):
                orig_image = np.array(image.convert('RGB'))
            else:
                orig_image = image.copy()

        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)

        # Inference
        detections = self.model.get_detections(
            image_tensor,
            conf_threshold=self.conf_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            max_detections=self.max_detections
        )[0]  # Get first (and only) image

        # Add class names
        class_names = [
            Config.COCO_CLASSES[int(cls)] if int(cls) < len(Config.COCO_CLASSES)
            else f"Class_{int(cls)}"
            for cls in detections['classes']
        ]

        result = {
            'boxes': detections['boxes'],
            'scores': detections['scores'],
            'classes': detections['classes'],
            'class_names': class_names
        }

        if return_image:
            result['image'] = orig_image

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Dict]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images

        Returns:
            List of detection dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results

    def visualize_prediction(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        save_path: Union[str, Path] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Run inference and visualize results.

        Args:
            image: Input image
            save_path: Optional path to save visualization
            show: Whether to display the image

        Returns:
            Visualization image
        """
        # Get predictions
        result = self.predict(image, return_image=True)

        # Prepare detections for visualization
        detections = {
            'boxes': result['boxes'],
            'scores': result['scores'],
            'classes': result['classes']
        }

        # Visualize
        vis_image = visualize_detections(
            result['image'],
            detections,
            class_names=Config.COCO_CLASSES,
            conf_threshold=0.0,  # Already filtered
            save_path=save_path,
            show=show
        )

        return vis_image


def main(args):
    """Main inference function for command-line usage."""

    print("=" * 70)
    print("Hybrid CNN-Transformer Object Detection - Inference")
    print("=" * 70)

    # Create detector
    detector = ObjectDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections
    )

    # Process single image
    if args.image:
        print(f"\nProcessing image: {args.image}")
        result = detector.predict(args.image)

        print(f"\nDetected {len(result['boxes'])} objects:")
        for i, (box, score, class_name) in enumerate(
            zip(result['boxes'], result['scores'], result['class_names'])
        ):
            print(f"  {i+1}. {class_name}: {score:.3f} at {box.numpy()}")

        # Visualize if requested
        if args.visualize:
            output_path = args.output_dir / f"detection_{Path(args.image).stem}.jpg"
            detector.visualize_prediction(
                args.image,
                save_path=output_path,
                show=args.show
            )
            print(f"\nVisualization saved to: {output_path}")

    # Process directory of images
    elif args.image_dir:
        print(f"\nProcessing images from: {args.image_dir}")
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        print(f"Found {len(image_paths)} images")

        for image_path in image_paths:
            print(f"\nProcessing: {image_path.name}")
            result = detector.predict(image_path)
            print(f"  Detected {len(result['boxes'])} objects")

            if args.visualize:
                output_path = args.output_dir / f"detection_{image_path.stem}.jpg"
                detector.visualize_prediction(
                    image_path,
                    save_path=output_path
                )

        print(f"\nAll visualizations saved to: {args.output_dir}")

    else:
        print("Error: Please provide --image or --image-dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference with Hybrid CNN-Transformer Object Detector'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory of images for batch inference')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/inference'),
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--nms-iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--max-detections', type=int, default=100,
                        help='Maximum detections per image')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--show', action='store_true',
                        help='Display visualization images')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
