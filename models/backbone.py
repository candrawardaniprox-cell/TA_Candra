import torch
import torch.nn as nn
import torchvision.models as models

class DynamicCNNBackbone(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == 'none':
            # =================================================================
            # EKSPERIMEN 1: TANPA BACKBONE CNN (Pure Vision Transformer Style)
            # =================================================================
            # Gambar mentah hanya diiris menjadi 'Patch' tanpa lapisan CNN dalam
            self.c3_channels = 128
            self.c4_channels = 256
            
            # Linear Projection Stride 8 (Menggantikan output blok C3 ResNet)
            self.patch_embed_c3 = nn.Conv2d(3, self.c3_channels, kernel_size=8, stride=8)
            # Linear Projection Stride 16 (Menggantikan output blok C4 ResNet)
            self.patch_embed_c4 = nn.Conv2d(self.c3_channels, self.c4_channels, kernel_size=2, stride=2)
            
        elif self.backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.c3_channels = 128
            self.c4_channels = 256
            
            self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer2 = resnet.layer2 
            self.layer3 = resnet.layer3 
            
        elif self.backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.c3_channels = 512
            self.c4_channels = 1024
            
            self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer2 = resnet.layer2 
            self.layer3 = resnet.layer3 
            
        else:
            raise ValueError(f"Backbone {backbone_name} belum didukung.")

    def forward(self, x):
        if self.backbone_name == 'none':
            # Bypass CNN: Gambar mentah murni diproyeksikan jadi grid (token)
            c3 = self.patch_embed_c3(x)
            c4 = self.patch_embed_c4(c3)
            return c3, c4
        else:
            # Gunakan CNN (Hybrid Architecture)
            x = self.layer1(x)
            c3 = self.layer2(x)
            c4 = self.layer3(c3)
            return c3, c4