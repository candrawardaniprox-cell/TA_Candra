import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    def __init__(self, channels=None, kernel_size=None, padding=None):
        super().__init__()
        
        # 1. Download ResNet18 yang sudah "Pintar" (Pre-trained ImageNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Potong ResNet18 hanya sampai 'layer3'
        # Ini secara ajaib akan menghasilkan output 256 channels dan ukuran 32x32
        # (Sangat sempurna untuk masuk ke Transformer Encoder Anda)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # Output: 64 channels
            resnet.layer2,  # Output: 128 channels
            resnet.layer3   # Output: 256 channels
        )
        
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_output_shape(self, input_size: int) -> tuple:
        return (self.out_channels, input_size // 16, input_size // 16)