# losses.py (MultiTaskLoss simplified wrapper)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, recon_logits, true_tokens, labels):

        loss_cls = F.cross_entropy(logits, labels)
        loss_rec = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)), true_tokens.view(-1))
        total = self.alpha * loss_cls + self.beta * loss_rec
        return total, loss_cls, loss_rec
