"""1D-CNN from Note S2, Fig. S19 and Tables S4-S5 of Wang et al. (2026)."""

import torch
from torch import nn


class TengCNN(nn.Module):
    """C1 (6) -> pool -> C2 (12) -> pool -> C3 (60) -> FC1 (60) -> FC2 (6), kernel 5, stride 1, ReLU.

    Softmax is applied by the cross-entropy loss during training and explicitly at inference.
    """

    def __init__(self, length=1024, n_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5, stride=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(6, 12, kernel_size=5, stride=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(12, 60, kernel_size=5, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, length)).numel()
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 60), nn.ReLU())
        self.fc2 = nn.Linear(60, n_classes)

    def embed(self, x):
        """FC1 activations, used for the t-SNE plot (Fig. 6n)."""
        return self.fc1(self.features(x))

    def forward(self, x):
        return self.fc2(self.embed(x))
