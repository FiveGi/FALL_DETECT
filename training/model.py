"""Lightweight temporal CNN classifier for fall detection from pose keypoint windows.

An ST-GCN variant (graph convolutions over the actual skeleton edges) was tried and
underperformed this simpler model (val F1 0.582 vs 0.615) despite having ~17x more
parameters -- see training notes. Reverted to this architecture as the best result so far.

Input:  (batch, window_size=30, 85) -- 17 landmarks x (x, y, confidence, vx, vy), per frame
Output: (batch,) raw logit -- pass through sigmoid for fall probability.
"""
import torch
import torch.nn as nn

NUM_LANDMARKS = 17  # unified COCO-17 keypoint set (see dataset.py)
FEATURES_PER_FRAME = NUM_LANDMARKS * 5  # x, y, confidence, vx, vy


class FallClassifier(nn.Module):
    def __init__(self, in_features=FEATURES_PER_FRAME, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, T, F) -> (batch, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)  # (batch, hidden)
        return self.head(x).squeeze(-1)  # (batch,) logits
