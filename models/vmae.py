#VMAE Model
import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224


class VideoMAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vit_base_patch16_224()
        self.encoder.head = nn.Identity()

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # treat each frame as a separate image
        feats = self.encoder(x)  # [B*T, D]
        return feats.view(B, T, -1)  # [B, T, D]

class VideoMAEWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VideoMAEEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded