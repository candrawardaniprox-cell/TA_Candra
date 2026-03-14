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
from torch.amp import GradScaler, autocast
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode tanpa GUI
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config
from models import HybridDetector
from data import ObjectDetectionDataset, get_train_transforms, get_val_transforms, create_dataloaders
from utils import DetectionLoss, calculate_map, batched_nms
# Import fungsi perhitungan Precision dan Recall
from utils.metrics import calculate_precision_recall


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('object_detection')
    logger.setLevel(logging.INFO)

    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def generate_confusion_matrix(predictions, targets, num_classes, save_path):
    """
    Menghasilkan dan menyimpan Confusion Matrix untuk Object Detection.
    """
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    iou_thresh = 0.5
    
    for pred, target in zip(predictions, targets):
        p_boxes = pred['boxes'].cpu().numpy()
        p_classes = pred['classes'].cpu().numpy()
        p_scores = pred['scores'].cpu().numpy()
        
        t_boxes = target['boxes'].cpu().numpy()
        t_labels = target['labels'].cpu().numpy()
        
        if len(t_boxes) == 0 and len(p_boxes) == 0:
            continue
            
        if len(t_boxes) == 0:
            for pc in p_classes:
                cm[num_classes][int(pc)] += 1
            continue
            
        if len(p_boxes) == 0:
            for tc in t_labels:
                cm[int(tc)][num_classes] += 1
            continue
            
        def to_corners(boxes):
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            return np.stack([x1, y1, x2, y2], axis=1)
        
        pb_c = to_corners(p_boxes)
        tb_c = to_corners(t_boxes)
        
        ious = np.zeros((len(pb_c), len(tb_c)))
        for i, pb in enumerate(pb_c):
            for j, tb in enumerate(tb_c):
                ix1, iy1 = max(pb[0], tb[0]), max(pb[1], tb[1])
                ix2, iy2 = min(pb[2], tb[2]), min(pb[3], tb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                uni = (pb[2]-pb[0])*(pb[3]-pb[1]) + (tb[2]-tb[0])*(tb[3]-tb[1]) - inter
                ious[i, j] = inter / (uni + 1e-6)
                
        matched_targets = set()
        sorted_idx = np.argsort(-p_scores)
        
        for p_idx in sorted_idx:
            best_iou = 0
            best_t_idx = -1
            for t_idx in range(len(t_boxes)):
                if t_idx in matched_targets:
                    continue
                if ious[p_idx, t_idx] > best_iou:
                    best_iou = ious[p_idx, t_idx]
                    best_t_idx = t_idx
                    
            p_cls = int(p_classes[p_idx])
            if best_iou >= iou_thresh:
                t_cls = int(t_labels[best_t_idx])
                cm[t_cls][p_cls] += 1
                matched_targets.add(best_t_idx)
            else:
                cm[num_classes][p_cls] += 1
                
        for t_idx in range(len(t_boxes)):
            if t_idx not in matched_targets:
                t_cls = int(t_labels[t_idx])
                cm[t_cls][num_classes] += 1
                
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix @ IoU 0.50', fontsize=14)
    plt.colorbar()
    
    classes = [f'Class {i}' for i in range(num_classes)] + ['Background']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=11, fontweight='bold')
                     
    plt.ylabel('True Label (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_one_epoch(
    model: nn.Module, dataloader, criterion, optimizer, scaler,
    device, epoch: int, logger, writer=None, log_frequency: int = 10
) -> dict:
    model.train()

    total_loss = 0.0
    total_obj_loss = 0.0
    total_bbox_loss = 0.0
    total_class_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses['total_loss']

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        total_obj_loss += losses['obj_loss'].item() if isinstance(losses['obj_loss'], torch.Tensor) else losses['obj_loss']
        total_bbox_loss += losses['bbox_loss'].item() if isinstance(losses['bbox_loss'], torch.Tensor) else losses['bbox_loss']
        total_class_loss += losses['class_loss'].item() if isinstance(losses['class_loss'], torch.Tensor) else losses['class_loss']
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'obj': f"{losses['obj_loss'].item():.4f}",
            'bbox': f"{losses['bbox_loss'].item():.4f}",
            'cls': f"{losses['class_loss'].item():.4f}"
        })

        if writer is not None and batch_idx % log_frequency == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/ObjLoss', losses['obj_loss'].item(), global_step)
            writer.add_scalar('Train/BBoxLoss', losses['bbox_loss'].item(), global_step)
            writer.add_scalar('Train/ClassLoss', losses['class_loss'].item(), global_step)

    avg_losses = {
        'total_loss': total_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'bbox_loss': total_bbox_loss / num_batches,
        'class_loss': total_class_loss / num_batches
    }

    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module, dataloader, criterion, device, epoch: int, logger, writer=None
) -> tuple:
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Validation')

    for images, targets in pbar:
        images = images.to(device)

        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            losses = criterion(outputs, targets)

        total_loss += losses['total_loss'].item()
        num_batches += 1

        detections = model.get_detections(
            images,
            conf_threshold=Config.CONF_THRESHOLD,
            nms_iou_threshold=Config.NMS_IOU_THRESHOLD,
            max_detections=Config.MAX_DETECTIONS
        )

        for det, target_boxes, target_labels in zip(detections, targets['boxes'], targets['labels']):
            all_predictions.append(det)
            all_targets.append({
                'boxes': target_boxes,
                'labels': target_labels
            })

        pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})

    avg_loss = total_loss / num_batches

    if len(all_predictions) > 0:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
        map_metrics = calculate_map(
            all_predictions[:min(len(all_predictions), 500)],
            all_targets[:min(len(all_targets), 500)],
            num_classes=Config.NUM_CLASSES,
            iou_thresholds=iou_thresholds
        )
    else:
        map_metrics = {'mAP@0.50': 0.0, 'mAP@[0.5:0.95]': 0.0}

    metrics = {
        'val_loss': avg_loss,
        **map_metrics
    }

    if writer is not None:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        for key, value in map_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)

    return metrics, all_predictions, all_targets


def save_checkpoint(
    model: nn.Module, optimizer, scheduler, epoch: int, metrics: dict, checkpoint_dir: Path, filename: str = None
):
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
    checkpoint = torch.load(checkpoint_path, weights_only=False)
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
    Config.create_directories()
    logger = setup_logging(Config.LOG_DIR)

    logger.info("=" * 60)
    logger.info("Starting Hybrid CNN-Transformer Object Detection Training")
    logger.info("=" * 60)

    Config.validate_config()
    Config.print_config()

    device = Config.DEVICE
    logger.info(f"Using device: {device}")
    logger.info("Loading datasets...")

    train_transform = get_train_transforms(
        image_size=Config.IMAGE_SIZE, mean=Config.MEAN, std=Config.STD,
        h_flip_prob=Config.HORIZONTAL_FLIP_PROB, brightness_contrast_limit=Config.BRIGHTNESS_CONTRAST_LIMIT,
        hue_saturation_limit=Config.HUE_SATURATION_VALUE_LIMIT
    )

    val_transform = get_val_transforms(image_size=Config.IMAGE_SIZE, mean=Config.MEAN, std=Config.STD)

    try:
        train_dataset = ObjectDetectionDataset(
            image_dir=Config.TRAIN_IMAGES, annotation_file=Config.TRAIN_ANNOTATIONS,
            transform=train_transform, image_size=Config.IMAGE_SIZE
        )
        val_dataset = ObjectDetectionDataset(
            image_dir=Config.VAL_IMAGES, annotation_file=Config.VAL_ANNOTATIONS,
            transform=val_transform, image_size=Config.IMAGE_SIZE
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        return

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, persistent_workers=Config.PERSISTENT_WORKERS
    )

    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")

    logger.info("Creating model...")
    model = HybridDetector(
        num_classes=Config.NUM_CLASSES, image_size=Config.IMAGE_SIZE,
        backbone_channels=Config.BACKBONE_CHANNELS, transformer_dim=Config.TRANSFORMER_DIM,
        transformer_heads=Config.TRANSFORMER_HEADS, transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_ff_dim=Config.TRANSFORMER_FF_DIM, num_anchors=Config.NUM_ANCHORS,
        anchors=Config.ANCHOR_BOXES, dropout=Config.TRANSFORMER_DROPOUT
    )
    model = model.to(device)

    criterion = DetectionLoss(
        num_classes=Config.NUM_CLASSES, lambda_obj=Config.LAMBDA_OBJ, lambda_noobj=Config.LAMBDA_NOOBJ,
        lambda_bbox=Config.LAMBDA_BBOX, lambda_class=Config.LAMBDA_CLASS,
        iou_threshold_pos=Config.IOU_THRESHOLD_POS, iou_threshold_neg=Config.IOU_THRESHOLD_NEG,
        bbox_loss_type=Config.BBOX_LOSS_TYPE, use_focal_loss=Config.USE_FOCAL_LOSS,
        focal_alpha=Config.FOCAL_ALPHA, focal_gamma=Config.FOCAL_GAMMA
    )

    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.LEARNING_RATE * 0.1},
        {'params': other_params, 'lr': Config.LEARNING_RATE}
    ], weight_decay=Config.WEIGHT_DECAY)

    if Config.LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.LEARNING_RATE * 0.01)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)

    scaler = GradScaler('cuda', enabled=Config.USE_AMP)

    writer = SummaryWriter(Config.LOG_DIR / 'tensorboard') if Config.USE_TENSORBOARD else None

    start_epoch = 0
    best_map = 0.0
    
    # === Inisialisasi Riwayat Grafik ===
    history_train_loss = []
    history_val_loss = []
    history_map = []
    history_map_strict = []
    history_train_bbox = []
    history_train_cls = []
    history_train_obj = []
    history_precision = [] # BARU
    history_recall = []    # BARU

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        best_map = metrics.get('mAP@0.50', 0.0)

        history_file = Path('training_history.json')
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    past_history = json.load(f)
                
                history_train_loss = past_history.get('train_loss', [])[:start_epoch]
                history_val_loss = past_history.get('val_loss', [])[:start_epoch]
                history_map = past_history.get('map', [])[:start_epoch]
                history_map_strict = past_history.get('map_strict', [])[:start_epoch]
                history_train_bbox = past_history.get('train_bbox', [])[:start_epoch]
                history_train_cls = past_history.get('train_cls', [])[:start_epoch]
                history_train_obj = past_history.get('train_obj', [])[:start_epoch]
                history_precision = past_history.get('precision', [])[:start_epoch] # BARU
                history_recall = past_history.get('recall', [])[:start_epoch]       # BARU
                
                logger.info(f"Berhasil memuat riwayat grafik untuk {len(history_train_loss)} epoch sebelumnya.")
            except Exception as e:
                logger.warning(f"Gagal memuat riwayat grafik: {e}")

    logger.info("Starting training...")

    for epoch in range(start_epoch, Config.EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train 1 Epoch
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, writer, Config.LOG_FREQUENCY
        )
        
        history_train_loss.append(train_losses['total_loss'])
        history_train_bbox.append(train_losses['bbox_loss'])
        history_train_cls.append(train_losses['class_loss'])
        history_train_obj.append(train_losses['obj_loss'])

        logger.info(f"Train - Loss: {train_losses['total_loss']:.4f}, Obj: {train_losses['obj_loss']:.4f}, BBox: {train_losses['bbox_loss']:.4f}, Class: {train_losses['class_loss']:.4f}")

        # Evaluasi
        if (epoch + 1) % Config.EVAL_FREQUENCY == 0:
            val_metrics, val_preds, val_targets = validate(model, val_loader, criterion, device, epoch, logger, writer)

            # === HITUNG PRECISION DAN RECALL ===
            precisions, recalls = calculate_precision_recall(val_preds, val_targets, iou_threshold=0.5, num_classes=Config.NUM_CLASSES)
            mean_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
            mean_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
            
            history_precision.append(mean_precision)
            history_recall.append(mean_recall)
            # ===================================

            history_val_loss.append(val_metrics['val_loss'])
            history_map.append(val_metrics.get('mAP@0.50', 0.0))
            history_map_strict.append(val_metrics.get('mAP@[0.5:0.95]', 0.0))

            logger.info(f"Val - Loss: {val_metrics['val_loss']:.4f}, mAP@0.50: {val_metrics.get('mAP@0.50', 0.0):.4f}")
            logger.info(f"Val - Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}")

            # Simpan Data JSON
            with open('training_history.json', 'w') as f:
                json.dump({
                    'train_loss': history_train_loss, 
                    'val_loss': history_val_loss, 
                    'map': history_map,
                    'map_strict': history_map_strict,
                    'train_bbox': history_train_bbox,
                    'train_cls': history_train_cls,
                    'train_obj': history_train_obj,
                    'precision': history_precision, # BARU
                    'recall': history_recall        # BARU
                }, f)

            # Setup Sumbu X
            x_train = list(range(start_epoch + 1, start_epoch + 1 + len(history_train_loss)))
            x_val = [start_epoch + i for i in range(1, len(history_train_loss) + 1) if i % Config.EVAL_FREQUENCY == 0]

            # 1. Plot Loss
            plt.figure(figsize=(10, 5))
            plt.plot(x_train, history_train_loss, label='Train Loss', color='blue', linewidth=2)
            if len(x_val) == len(history_val_loss):
                plt.plot(x_val, history_val_loss, label='Validation Loss', color='red', marker='o', linewidth=2)
            plt.title('Training and Validation Loss', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('loss_graph.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Plot mAP
            if len(x_val) == len(history_map):
                plt.figure(figsize=(10, 5))
                plt.plot(x_val, history_map, label='mAP@0.50', color='green', marker='s', linewidth=2)
                plt.plot(x_val, history_map_strict, label='mAP@[0.5:0.95]', color='teal', marker='^', linewidth=2)
                plt.title('Validation Mean Average Precision (mAP)', fontsize=14)
                plt.xlabel('Epochs', fontsize=12)
                plt.ylabel('mAP', fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig('map_graph.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 3. Plot PRECISION DAN RECALL (BARU)
            if len(x_val) == len(history_precision):
                plt.figure(figsize=(10, 5))
                plt.plot(x_val, history_precision, label='Precision', color='blue', marker='o', linewidth=2)
                plt.plot(x_val, history_recall, label='Recall', color='darkorange', marker='D', linewidth=2)
                plt.title('Validation Precision & Recall @ IoU 0.50', fontsize=14)
                plt.xlabel('Epochs', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.ylim(0, 1.05) # Memastikan sumbu Y dari 0 sampai 1
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig('precision_recall_graph.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            # 4. Plot BBox Loss
            plt.figure(figsize=(10, 5))
            plt.plot(x_train, history_train_bbox, label='Train BBox Loss', color='orange', linewidth=2)
            plt.title('Training Bounding Box Loss', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('loss_bbox_graph.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 5. Plot Class Loss
            plt.figure(figsize=(10, 5))
            plt.plot(x_train, history_train_cls, label='Train Class Loss', color='purple', linewidth=2)
            plt.title('Training Classification Loss', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('loss_class_graph.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 6. Plot Obj Loss
            plt.figure(figsize=(10, 5))
            plt.plot(x_train, history_train_obj, label='Train Objectness Loss', color='brown', linewidth=2)
            plt.title('Training Objectness Loss', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig('loss_obj_graph.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 7. Confusion Matrix
            try:
                generate_confusion_matrix(val_preds, val_targets, Config.NUM_CLASSES, 'confusion_matrix.png')
                logger.info("--> 7 Grafik Evaluasi Utama berhasil diperbarui!")
            except Exception as e:
                logger.error(f"Gagal membuat Confusion Matrix: {e}")

            # Menyimpan Model Terbaik
            current_map = val_metrics.get('mAP@0.50', 0.0)
            if current_map > best_map:
                best_map = current_map
                best_checkpoint = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    val_metrics, Config.CHECKPOINT_DIR, 'best_model.pth'
                )
                logger.info(f"New best model! mAP: {best_map:.4f}, saved to {best_checkpoint}")

        if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
            current_metrics = val_metrics if 'val_metrics' in locals() else {}
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, current_metrics, Config.CHECKPOINT_DIR)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        scheduler.step()

    logger.info("\nTraining completed!")
    logger.info(f"Best mAP@0.5: {best_map:.4f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-Transformer Object Detector')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train(args)