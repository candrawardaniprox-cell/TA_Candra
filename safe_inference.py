"""
Launcher sederhana untuk inference dua tahap dengan satu perintah.

Contoh:
    python safe_inference.py --image path\ke\gambar.jpg
"""

from __future__ import annotations

import argparse

from config import Config
from demo_aplikasi.two_stage_inference import main as two_stage_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference aman satu perintah untuk detector + classifier.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    forwarded = argparse.Namespace(
        image=args.image,
        detector_checkpoint=args.detector_checkpoint or str(Config.CHECKPOINT_DIR / "best_model.pth"),
        classifier_checkpoint=args.classifier_checkpoint or str(Config.CLASSIFIER_CHECKPOINT_PATH),
        device=args.device,
        output=args.output,
    )
    two_stage_main(forwarded)
