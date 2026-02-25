"""
Training script for Hybrid CNN-Transformer Object Detection.

This script handles the complete training pipeline including:
- Data loading and augmentation
- Model initialization
- Training loop with mixed precision
- Validation and checkpointing
- Logging and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime

from config import Config
from models import HybridDetector
from data import ObjectDetectionDataset, get_train_transforms, get_val_transforms, create_dataloaders
from utils import DetectionLoss, calculate_map, batched_nms


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files

    Returns:
        Logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('object_detection')
    logger.setLevel(logging.INFO)

    # File handler
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int,
    logger,
    writer=None,
    log_frequency: int = 10
) -> dict:
    """
    Train for one epoch.

    Args:
        model: The detection model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
        writer: TensorBoard writer
        log_frequency: How often to log (in batches)

    Returns:
        Dictionary with average losses
    """
    model.train()

    total_loss = 0.0
    total_obj_loss = 0.0
    total_bbox_loss = 0.0
    total_class_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = images.to(device)

        # Forward pass with mixed precision
        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses['total_loss']

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        total_loss += loss.item()
        total_obj_loss += losses['obj_loss'].item()
        total_bbox_loss += losses['bbox_loss'].item()
        total_class_loss += losses['class_loss'].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'obj': f"{losses['obj_loss'].item():.4f}",
            'bbox': f"{losses['bbox_loss'].item():.4f}",
            'cls': f"{losses['class_loss'].item():.4f}"
        })

        # Log to tensorboard
        if writer is not None and batch_idx % log_frequency == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/ObjLoss', losses['obj_loss'].item(), global_step)
            writer.add_scalar('Train/BBoxLoss', losses['bbox_loss'].item(), global_step)
            writer.add_scalar('Train/ClassLoss', losses['class_loss'].item(), global_step)

    # Calculate averages
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'bbox_loss': total_bbox_loss / num_batches,
        'class_loss': total_class_loss / num_batches
    }

    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device,
    epoch: int,
    logger,
    writer=None
) -> dict:
    """
    Validate the model.

    Args:
        model: The detection model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger instance
        writer: TensorBoard writer

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # Collect predictions and targets for mAP calculation
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Validation')

    for images, targets in pbar:
        # Move to device
        images = images.to(device)

        # Forward pass
        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            losses = criterion(outputs, targets)

        total_loss += losses['total_loss'].item()
        num_batches += 1

        # Get detections for mAP calculation
        detections = model.get_detections(
            images,
            conf_threshold=Config.CONF_THRESHOLD,
            nms_iou_threshold=Config.NMS_IOU_THRESHOLD,
            max_detections=Config.MAX_DETECTIONS
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

        pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})

    # Calculate metrics
    avg_loss = total_loss / num_batches

    # Calculate mAP (on subset to save time)
    if len(all_predictions) > 0:
        map_metrics = calculate_map(
            all_predictions[:min(len(all_predictions), 500)],
            all_targets[:min(len(all_targets), 500)],
            num_classes=Config.NUM_CLASSES,
            iou_thresholds=[0.5]
        )
    else:
        map_metrics = {'mAP@0.50': 0.0}

    metrics = {
        'val_loss': avg_loss,
        **map_metrics
    }

    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        for key, value in map_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
    filename: str = None
):
    """
    Save model checkpoint.

    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Validation metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Optional custom filename
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'

    checkpoint_path = checkpoint_dir / filename

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, scheduler=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Epoch number and metrics from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    return epoch, metrics


def train(args):
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    # Setup
    Config.create_directories()
    logger = setup_logging(Config.LOG_DIR)

    logger.info("=" * 60)
    logger.info("Starting Hybrid CNN-Transformer Object Detection Training")
    logger.info("=" * 60)

    # Print configuration
    Config.validate_config()
    Config.print_config()

    # Setup device
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading datasets...")

    train_transform = get_train_transforms(
        image_size=Config.IMAGE_SIZE,
        mean=Config.MEAN,
        std=Config.STD,
        h_flip_prob=Config.HORIZONTAL_FLIP_PROB,
        brightness_contrast_limit=Config.BRIGHTNESS_CONTRAST_LIMIT,
        hue_saturation_limit=Config.HUE_SATURATION_VALUE_LIMIT
    )

    val_transform = get_val_transforms(
        image_size=Config.IMAGE_SIZE,
        mean=Config.MEAN,
        std=Config.STD
    )

    try:
        train_dataset = ObjectDetectionDataset(
            image_dir=Config.TRAIN_IMAGES,
            annotation_file=Config.TRAIN_ANNOTATIONS,
            transform=train_transform,
            image_size=Config.IMAGE_SIZE
        )

        val_dataset = ObjectDetectionDataset(
            image_dir=Config.VAL_IMAGES,
            annotation_file=Config.VAL_ANNOTATIONS,
            transform=val_transform,
            image_size=Config.IMAGE_SIZE
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please download the COCO dataset first. See README.md for instructions.")
        return

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Val batches per epoch: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
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
    model.print_model_summary()

    # Create loss function
    criterion = DetectionLoss(
        num_classes=Config.NUM_CLASSES,
        lambda_obj=Config.LAMBDA_OBJ,
        lambda_noobj=Config.LAMBDA_NOOBJ,
        lambda_bbox=Config.LAMBDA_BBOX,
        lambda_class=Config.LAMBDA_CLASS,
        iou_threshold_pos=Config.IOU_THRESHOLD_POS,
        iou_threshold_neg=Config.IOU_THRESHOLD_NEG,
        bbox_loss_type=Config.BBOX_LOSS_TYPE,
        use_focal_loss=Config.USE_FOCAL_LOSS,
        focal_alpha=Config.FOCAL_ALPHA,
        focal_gamma=Config.FOCAL_GAMMA
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Create scheduler
    if Config.LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.EPOCHS,
            eta_min=Config.LEARNING_RATE * 0.01
        )
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=Config.LR_STEP_SIZE,
            gamma=Config.LR_GAMMA
        )

    # Create scaler for mixed precision
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Setup tensorboard
    writer = None
    if Config.USE_TENSORBOARD:
        writer = SummaryWriter(Config.LOG_DIR / 'tensorboard')

    # Load checkpoint if resuming
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        best_map = metrics.get('mAP@0.50', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, Config.EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, writer, Config.LOG_FREQUENCY
        )

        logger.info(
            f"Train - Loss: {train_losses['total_loss']:.4f}, "
            f"Obj: {train_losses['obj_loss']:.4f}, "
            f"BBox: {train_losses['bbox_loss']:.4f}, "
            f"Class: {train_losses['class_loss']:.4f}"
        )

        # Validate
        if (epoch + 1) % Config.EVAL_FREQUENCY == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, logger, writer
            )

            logger.info(
                f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                f"mAP@0.5: {val_metrics.get('mAP@0.50', 0.0):.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    val_metrics, Config.CHECKPOINT_DIR
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            current_map = val_metrics.get('mAP@0.50', 0.0)
            if current_map > best_map:
                best_map = current_map
                best_checkpoint = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    val_metrics, Config.CHECKPOINT_DIR, 'best_model.pth'
                )
                logger.info(f"New best model! mAP: {best_map:.4f}, saved to {best_checkpoint}")

        # Step scheduler
        scheduler.step()

    logger.info("\nTraining completed!")
    logger.info(f"Best mAP@0.5: {best_map:.4f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-Transformer Object Detector')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train(args)
