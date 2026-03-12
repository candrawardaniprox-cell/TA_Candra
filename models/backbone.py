import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    def __init__(self, channels=None, kernel_size=None, padding=None):
        super().__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.out_channels = 256

        # --- TAMBAHKAN 2 BARIS INI UNTUK MEMBEKUKAN RESNET ---
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        # -----------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_output_shape(self, input_size: int) -> tuple:
        return (self.out_channels, input_size // 16, input_size // 16)