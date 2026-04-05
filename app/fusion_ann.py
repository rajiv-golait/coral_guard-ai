"""FusionANN from Phase 6 in `mdm.py` — tabular + DBSCAN one-hot -> Percent_Bleaching."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionANN(nn.Module):
    """Matches `FusionANN` in `mdm.py`; forward(tab, clust) -> scalar % bleaching."""

    def __init__(self, n_tab: int, n_clust: int, dropout: float = 0.3) -> None:
        super().__init__()
        input_dim = n_tab + n_clust
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.regression_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, tab: torch.Tensor, clust: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tab, clust], dim=1)
        feat = self.net(x)
        return self.regression_head(feat) * 100
