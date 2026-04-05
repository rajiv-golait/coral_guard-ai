"""CoralCNN from Colab `mdm.py` — EfficientNet-B3 backbone + custom classifier head."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class CoralCNN(nn.Module):
    """Matches Phase 4 training in `mdm.py` / MDM.ipynb."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.embed_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        emb = self.backbone(x)
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits
