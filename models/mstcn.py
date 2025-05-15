#MS-TCN

import torch
import torch.nn as nn

class MSTCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, input_dim, 3, padding=1, groups=input_dim),
                nn.ReLU(),
                nn.Conv1d(input_dim, input_dim, 1),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Conv1d(input_dim, num_classes, 1)

    def forward(self, x):  # [B, T, D]
        x = x.permute(0, 2, 1)  # [B, D, T]
        for layer in self.layers:
            x = x + layer(x)
        out = self.classifier(x)  # [B, num_classes, T]
        return out.permute(0, 2, 1)  # [B, T, num_classes]
