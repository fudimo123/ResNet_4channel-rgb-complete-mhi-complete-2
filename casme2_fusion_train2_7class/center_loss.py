import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        batch_centers = self.centers[labels]
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss

