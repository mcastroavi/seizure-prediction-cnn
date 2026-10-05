"""Training loss shared by every encoder and the context model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSeizureLoss(nn.Module):
    """alpha * MSE(sigmoid(logit), soft risk) + (1 - alpha) * BCE(logit, hard label).

    The soft risk rises from 0.10 at the start of the 30-min preictal period to 1.00 at onset
    (by time to onset), so windows close to a seizure are pushed harder than early ones.
    """

    def __init__(self, alpha: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits, risk, hard):
        logits = logits.view(-1).float()
        mse = F.mse_loss(torch.sigmoid(logits), risk.view(-1).float())
        bce = F.binary_cross_entropy_with_logits(logits, hard.view(-1).float(), pos_weight=self.pos_weight)
        return self.alpha * mse + (1 - self.alpha) * bce
