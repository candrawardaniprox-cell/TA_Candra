"""
Evaluation script for Hybrid CNN-Transformer Object Detection.

This script evaluates a trained model on the validation set and computes
comprehensive metrics including mAP at various IoU thresholds.
"""

import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from config import Config
from models import HybridDetector
from data import ObjectDetectionDataset, get_val_transforms
from torch.utils.data import DataLoader
from utils import calculate_map, batched_nms, visualize_detections


def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    boxes_list = []
    labels_list = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        boxes_list.append(item['boxes'])
        labels_list.append(item['labels'])
        image_ids.append(item['image_id'])

    images = torch.stack(images, dim=0)
    targets = {
        'boxes': boxes_list,
        'labels': labels_list,
        'image_ids': image_ids
    }

    return images, targets


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    conf_threshold=0.5,
    nms_iou_threshold=0.45,
    max_detections=100,
    save_visualizations=False,
    output_dir=None,
    num_vis_samples=10
):
    """
    Evaluate model on validation set.

    Args:
        model: The detection model
        dataloader: Validation dataloader
        device: Device to evaluate on
        conf_threshold: Confidence threshold for detections
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        save_visualizations: Whether to save visualization images
        output_dir: Directory to save visualizations
        num_vis_samples: Number of samples to visualize

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_images = []

    print("Running inference on validation set...")

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
        # Move to device
        images_gpu = images.to(device)

        # Get predictions
        detections = model.get_detections(
            images_gpu,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections
        )

        # Store predictions and targets
        for det, target_boxes, target_labels in zip(
            detections, targets['boxes'], targets['labels']
        ):
            all_predictions.append(det)
            all_targets.append({
                'boxes': target_boxes,
                'labels': target_labels
            })

        # Store some images for visualization
        if save_visualizations and len(all_images) < num_vis_samples:
            all_images.append(images.cpu().numpy())

    print(f"\nProcessed {len(all_predictions)} images")

    # Calculate mAP at different IoU thresholds
    print("\nCalculating mAP metrics...")
    map_results = calculate_map(
        all_predictions,
        all_targets,
        num_classes=Config.NUM_CLASSES,
        iou_thresholds=Config.EVAL_IOU_THRESHOLDS
    )

    # Calculate per-class statistics
    print("\nCalculating per-class statistics...")
    class_stats = calculate_class_statistics(all_predictions, all_targets)

    # Save visualizations if requested
    if save_visualizations and output_dir is not None:
        print(f"\nSaving visualizations to {output_dir}...")
        save_visualization_samples(
            all_images,
            all_predictions[:num_vis_samples],
            output_dir,
            class_names=Config.COCO_CLASSES
        )

    results = {
        'map_metrics': map_results,
        'class_statistics': class_stats,
        'num_images': len(all_predictions)
    }

    return results


def calculate_class_statistics(predictions, targets):
    """
    Calculate per-class detection statistics.

    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries

    Returns:
        Dictionary with per-class statistics
    """
    from collections import defaultdict

    class_counts = defaultdict(int)
    class_detections = defaultdict(int)

    for target in targets:
        for label in target['labels']:
            class_counts[label.item()] += 1

    for pred in predictions:
        for label in pred['classes']:
            class_detections[label.item()] += 1

    stats = {
        'ground_truth_counts': dict(class_counts),
        'detection_counts': dict(class_detections)
    }

    return stats


def save_visualization_samples(images, predictions, output_dir, class_names):
    """
    Save visualization samples.

    Args:
        images: List of image arrays
        predictions: List of prediction dictionaries
        output_dir: Directory to save visualizations
        class_names: List of class names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (image_batch, pred) in enumerate(zip(images, predictions)):
        # Get first image from batch
        image = image_batch[0].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]

        # Denormalize
        mean = np.array(Config.MEAN)
        std = np.array(Config.STD)
        image = image * std + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Visualize detections
        vis_image = visualize_detections(
            image,
            pred,
            class_names=class_names,
            conf_threshold=0.0,  # Already filtered
            save_path=output_dir / f'sample_{idx:03d}.jpg'
        )


def print_results(results):
    """
    Print evaluation results in a formatted way.

    Args:
        results: Dictionary with evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nNumber of images evaluated: {results['num_images']}")

    print("\n" + "-" * 70)
    print("mAP Metrics:")
    print("-" * 70)
    for key, value in results['map_metrics'].items():
        print(f"  {key:20s}: {value:.4f}")

    print("\n" + "-" * 70)
    print("Class Statistics (Top 10 most common):")
    print("-" * 70)
    class_stats = results['class_statistics']
    gt_counts = class_stats['ground_truth_counts']

    # Sort by ground truth count
    sorted_classes = sorted(gt_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"{'Class ID':<10} {'Class Name':<20} {'Ground Truth':<15} {'Detections':<15}")
    print("-" * 70)
    for class_id, gt_count in sorted_classes:
        det_count = class_stats['detection_counts'].get(class_id, 0)
        class_name = Config.COCO_CLASSES[class_id] if class_id < len(Config.COCO_CLASSES) else f"Class_{class_id}"
        print(f"{class_id:<10} {class_name:<20} {gt_count:<15} {det_count:<15}")

    print("\n" + "=" * 70)


def save_results(results, output_path):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary with evaluation results
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    serializable_results = {
        'num_images': results['num_images'],
        'map_metrics': results['map_metrics'],
        'class_statistics': {
            'ground_truth_counts': {str(k): v for k, v in results['class_statistics']['ground_truth_counts'].items()},
            'detection_counts': {str(k): v for k, v in results['class_statistics']['detection_counts'].items()}
        }
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main(args):
    """Main evaluation function."""

    print("=" * 70)
    print("Hybrid CNN-Transformer Object Detection - Evaluation")
    print("=" * 70)

    # Setup device
    device = Config.DEVICE
    print(f"\nUsing device: {device}")

    # Create dataset
    print("\nLoading validation dataset...")
    val_transform = get_val_transforms(
        image_size=Config.IMAGE_SIZE,
        mean=Config.MEAN,
        std=Config.STD
    )

    val_dataset = ObjectDetectionDataset(
        image_dir=Config.VAL_IMAGES,
        annotation_file=Config.VAL_ANNOTATIONS,
        transform=val_transform,
        image_size=Config.IMAGE_SIZE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY
    )

    print(f"Validation dataset: {len(val_dataset)} images")

    # Create model
    print("\nCreating model...")
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
    model = model.to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Evaluate
    results = evaluate_model(
        model,
        val_loader,
        device,
        conf_threshold=args.conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir,
        num_vis_samples=args.num_vis_samples
    )

    # Print results
    print_results(results)

    # Save results
    if args.save_results:
        output_path = Path(args.output_dir) / 'evaluation_results.json'
        save_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Hybrid CNN-Transformer Object Detector')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--nms-iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--max-detections', type=int, default=100,
                        help='Maximum detections per image')
    parser.add_argument('--save-visualizations', action='store_true',
                        help='Save visualization samples')
    parser.add_argument('--num-vis-samples', type=int, default=10,
                        help='Number of visualization samples to save')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                        help='Directory to save outputs')
    parser.add_argument('--save-results', action='store_true', default=True,
                        help='Save results to JSON file')

    args = parser.parse_args()
    main(args)
