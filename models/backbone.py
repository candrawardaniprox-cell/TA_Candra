import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


def _extract_backbone_state_dict(payload):
    """Ekstrak state_dict backbone dari checkpoint umum atau checkpoint detector."""
    if not isinstance(payload, dict):
        return payload

    state_dict = (
        payload.get('backbone_state_dict')
        or payload.get('model_state_dict')
        or payload.get('state_dict')
        or payload
    )
    if not isinstance(state_dict, dict):
        return state_dict

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('module.backbone.'):
            new_key = new_key[len('module.backbone.'):]
        elif new_key.startswith('backbone.'):
            new_key = new_key[len('backbone.'):]
        elif new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        cleaned[new_key] = value
    return cleaned


def _resolve_pretrain_mode(pretrained=True, pretrain_source='imagenet'):
    source = str(pretrain_source or 'imagenet').lower()
    if source == 'none':
        return False, source
    return bool(pretrained), source

class DynamicCNNBackbone(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, pretrain_source='imagenet', custom_weights_path=None):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.pretrained, self.pretrain_source = _resolve_pretrain_mode(pretrained, pretrain_source)
        self.custom_weights_path = str(custom_weights_path) if custom_weights_path else None
        
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
            weights = models.ResNet18_Weights.DEFAULT if (self.pretrained and self.pretrain_source == 'imagenet') else None
            resnet = models.resnet18(weights=weights)
            self._maybe_load_custom_weights(resnet)
            self.c3_channels = 128
            self.c4_channels = 256
            
            self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer2 = resnet.layer2 
            self.layer3 = resnet.layer3 
            
        elif self.backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if (self.pretrained and self.pretrain_source == 'imagenet') else None
            resnet = models.resnet50(weights=weights)
            self._maybe_load_custom_weights(resnet)
            self.c3_channels = 512
            self.c4_channels = 1024
            
            self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
            self.layer2 = resnet.layer2 
            self.layer3 = resnet.layer3 
            
        else:
            raise ValueError(f"Backbone {backbone_name} belum didukung.")

    def _maybe_load_custom_weights(self, resnet: nn.Module) -> None:
        if not self.pretrained or self.pretrain_source == 'imagenet':
            return

        if self.pretrain_source not in {'agriculture', 'internal'}:
            raise ValueError(
                f"BACKBONE_PRETRAIN_SOURCE '{self.pretrain_source}' belum didukung. "
                "Gunakan 'imagenet', 'agriculture', 'internal', atau 'none'."
            )
        if not self.custom_weights_path:
            raise ValueError(
                f"BACKBONE_PRETRAIN_SOURCE='{self.pretrain_source}' membutuhkan BACKBONE_CUSTOM_WEIGHTS_PATH."
            )

        weights_path = Path(self.custom_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint backbone kustom tidak ditemukan: {weights_path}")

        payload = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = _extract_backbone_state_dict(payload)
        missing, unexpected = resnet.load_state_dict(state_dict, strict=False)

        critical_missing = [key for key in missing if key.startswith(('conv1', 'bn1', 'layer1', 'layer2', 'layer3'))]
        if critical_missing:
            raise RuntimeError(
                "Checkpoint backbone kustom tidak kompatibel dengan arsitektur "
                f"{self.backbone_name}. Missing keys penting: {critical_missing[:8]}"
            )
        self.loaded_custom_weight_report = {
            'source': self.pretrain_source,
            'path': str(weights_path),
            'missing_keys': missing,
            'unexpected_keys': unexpected,
        }

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
