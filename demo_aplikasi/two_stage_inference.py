"""
Inference dua tahap yang lebih aman:
1. Detector mencari kandidat area penyakit.
2. Classifier HTEM menentukan kelas final atau `unknown`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

from config import Config
from data import get_classifier_val_transforms, get_inference_transforms
from models import HybridDetector, PaperDiseaseClassifier
from utils.visualization import draw_bounding_boxes


def _load_rgb_image(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image.copy()


def _crop_xyxy(image: np.ndarray, box_xyxy: np.ndarray, padding: float) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(np.float32).tolist()
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * padding
    py = bh * padding
    x1 = max(0, int(np.floor(x1 - px)))
    y1 = max(0, int(np.floor(y1 - py)))
    x2 = min(w, int(np.ceil(x2 + px)))
    y2 = min(h, int(np.ceil(y2 + py)))
    return image[y1:y2, x1:x2]


def _visualize_two_stage_predictions(
    image_rgb: np.ndarray,
    results: List[Dict],
    save_path: str | Path | None = None,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not results:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image_bgr)
        return image_bgr

    boxes = [res["box_xyxy"] for res in results]
    labels = [res["class_name"] for res in results]
    scores = [res["classifier_confidence"] for res in results]
    vis = draw_bounding_boxes(
        image_bgr,
        boxes,
        labels,
        scores=scores,
        class_names=list(Config.COCO_CLASSES[:Config.NUM_CLASSES]),
        thickness=2,
        box_format="xyxy",
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)
    return vis


class SafeTwoStagePredictor:
    def __init__(
        self,
        detector_checkpoint: str,
        classifier_checkpoint: str,
        device: str | None = None,
    ):
        self.device = torch.device(device) if device else Config.DEVICE
        self.detector = self._load_detector(detector_checkpoint)
        self.classifier = self._load_classifier(classifier_checkpoint)
        self.detector_tf = get_inference_transforms(
            image_size=Config.IMAGE_SIZE,
            mean=Config.MEAN,
            std=Config.STD,
        )
        self.classifier_tf = get_classifier_val_transforms(
            image_size=Config.CLASSIFIER_IMAGE_SIZE,
            mean=Config.MEAN,
            std=Config.STD,
        )

    def _load_detector(self, checkpoint_path: str) -> HybridDetector:
        model = HybridDetector(
            num_classes=Config.NUM_CLASSES,
            image_size=Config.IMAGE_SIZE,
            transformer_dim=Config.TRANSFORMER_DIM,
            transformer_heads=Config.TRANSFORMER_HEADS,
            transformer_layers=Config.TRANSFORMER_LAYERS,
            transformer_ff_dim=Config.TRANSFORMER_FF_DIM,
            dropout=Config.TRANSFORMER_DROPOUT,
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model

    def _load_classifier(self, checkpoint_path: str) -> PaperDiseaseClassifier:
        model = PaperDiseaseClassifier(
            num_classes=Config.NUM_CLASSES,
            dropout=Config.CLASSIFIER_DROPOUT,
            stage_dims=list(getattr(Config, "PAPER_STAGE_DIMS", [24, 32, 48, 64])),
            stage_layout=list(getattr(Config, "PAPER_STAGE_LAYOUT", [2, 2, 4, 2])),
            stage_heads=list(getattr(Config, "PAPER_STAGE_HEADS", [1, 2, 4, 8])),
            stage_reductions=list(getattr(Config, "PAPER_STAGE_REDUCTIONS", [8, 4, 2, 1])),
            cte_channels=int(getattr(Config, "PAPER_CTE_CHANNELS", 16)),
            expansion_ratio=int(getattr(Config, "PAPER_LFFN_EXPANSION_RATIO", 4)),
            kernel_size=int(getattr(Config, "PAPER_LFFN_KERNEL_SIZE", 3)),
            embed_kernel_size=int(getattr(Config, "PAPER_EMBED_KERNEL_SIZE", 2)),
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_detector(self, image_rgb: np.ndarray) -> torch.Tensor:
        transformed = self.detector_tf(image=image_rgb)
        image_tensor = transformed["image"]
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)
        return image_tensor.unsqueeze(0).float().to(self.device)

    def _preprocess_classifier(self, crop_rgb: np.ndarray) -> torch.Tensor:
        transformed = self.classifier_tf(image=crop_rgb)
        image_tensor = transformed["image"]
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)
        return image_tensor.unsqueeze(0).float().to(self.device)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> List[Dict]:
        image_rgb = _load_rgb_image(image)
        detector_input = self._preprocess_detector(image_rgb)
        detections = self.detector.get_detections(
            detector_input,
            conf_threshold=Config.STAGE2_PROPOSAL_CONF_THRESHOLD,
            nms_iou_threshold=Config.STAGE2_PROPOSAL_NMS_IOU_THRESHOLD,
            max_detections=Config.STAGE2_PROPOSAL_MAX_DETECTIONS,
        )[0]

        if len(detections["boxes"]) == 0:
            return []

        # Jadikan proposal class-agnostic agar salah kelas detector tidak dibawa ke tahap 2.
        keep = batched_nms(
            detections["boxes"],
            detections["scores"],
            torch.zeros_like(detections["classes"], dtype=torch.long),
            Config.STAGE2_PROPOSAL_NMS_IOU_THRESHOLD,
        )

        results = []
        for idx in keep.tolist():
            box = detections["boxes"][idx].detach().cpu().numpy()
            det_score = float(detections["scores"][idx].item())
            crop = _crop_xyxy(image_rgb, box, Config.CLASSIFIER_CROP_PADDING)
            if crop.size == 0:
                continue

            classifier_input = self._preprocess_classifier(crop)
            safety = self.classifier.predict_with_safety(
                classifier_input,
                class_thresholds=list(Config.STAGE2_CLASS_THRESHOLDS),
                min_margin=float(Config.STAGE2_MIN_MARGIN),
                unknown_index=Config.NUM_CLASSES,
            )

            final_class = int(safety["final_classes"][0].item())
            probs = safety["probabilities"][0].detach().cpu().numpy()
            top_prob = float(safety["top_probs"][0].item())
            margin = float(safety["margins"][0].item())

            if final_class < Config.NUM_CLASSES:
                class_name = Config.COCO_CLASSES[final_class]
                action = Config.STAGE2_ACTION_MAP[final_class]
            else:
                class_name = Config.STAGE2_UNKNOWN_NAME
                action = Config.STAGE2_ACTION_MAP[-1]

            results.append(
                {
                    "box_xyxy": box,
                    "detector_score": det_score,
                    "classifier_probabilities": probs,
                    "classifier_confidence": top_prob,
                    "classifier_margin": margin,
                    "class_id": final_class,
                    "class_name": class_name,
                    "action": action,
                }
            )

        return results


def main(args: argparse.Namespace) -> None:
    predictor = SafeTwoStagePredictor(
        detector_checkpoint=args.detector_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        device=args.device,
    )
    image_rgb = _load_rgb_image(args.image)
    results = predictor.predict(image_rgb)
    print(f"Jumlah keputusan aman: {len(results)}")
    for idx, res in enumerate(results, start=1):
        print(
            f"{idx}. class={res['class_name']} | action={res['action']} | "
            f"det_score={res['detector_score']:.3f} | cls_conf={res['classifier_confidence']:.3f} | "
            f"margin={res['classifier_margin']:.3f}"
        )

    output_path = Path(args.output) if args.output else (
        Path("outputs") / "two_stage_inference" / f"{Path(args.image).stem}_two_stage.jpg"
    )
    _visualize_two_stage_predictions(image_rgb, results, output_path)
    print(f"Visualisasi dua tahap disimpan di: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference dua tahap detector + classifier safety.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default=str(Config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--classifier-checkpoint", type=str, default=str(Config.CLASSIFIER_CHECKPOINT_PATH))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    main(parser.parse_args())
