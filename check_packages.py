import sys
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError as e:
    print("torch: NOT FOUND -", e)

try:
    import cv2
    print("cv2:", cv2.__version__)
except ImportError as e:
    print("cv2: NOT FOUND -", e)

try:
    import numpy
    print("numpy:", numpy.__version__)
except ImportError as e:
    print("numpy: NOT FOUND -", e)

try:
    import albumentations
    print("albumentations:", albumentations.__version__)
except ImportError as e:
    print("albumentations: NOT FOUND -", e)

try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except ImportError as e:
    print("ultralytics: NOT FOUND -", e)

print("\nSemua cek selesai!")
