"""Coarse Segmentation Head and its supervised loss in OSS-CAEA."""
import torch
from torch import nn
from torch.nn import functional as F


class CoarseSegmentationHead(nn.Module):
    """Coarse Segmentation Head, PDF Figure 2 and Eq. (10)."""
    def __init__(self, channels, text_dim, temperature=50.):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.embedding = nn.Conv2d(channels, text_dim, 1)
        self.prior_proj = nn.Conv2d(text_dim, channels, 1)
        self.temperature = temperature

    def forward(self, x, text, output_size):
        latent = F.interpolate(self.features(x), output_size, mode='bilinear', align_corners=False)
        embedding = F.normalize(self.embedding(latent), dim=1, eps=1e-6)
        text = F.normalize(text, dim=-1, eps=1e-6).to(embedding.dtype)
        # Dynamic 1x1 category kernels permit a different vocabulary at test time.
        logits = self.temperature * torch.einsum('bchw,kc->bkhw', embedding, text)
        semantic_prior = torch.einsum('bkhw,kc->bchw', logits.softmax(dim=1), text)
        return latent + self.prior_proj(semantic_prior), logits


def coarse_segmentation_loss(logits, target, ignore_index=255, alpha=0.4):
    if target.ndim == 4:
        target = target.squeeze(1)
    logits = F.interpolate(logits.float(), target.shape[-2:], mode='bilinear', align_corners=False)
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index)
    safe_target = target.masked_fill(~valid, 0)
    truth = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float()
    truth = truth * valid[:, None]
    prob = logits.softmax(1) * valid[:, None]
    intersection = (prob * truth).sum((0, 2, 3))
    denominator = (prob + truth).sum((0, 2, 3))
    dice = (1 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return alpha * ce + (1 - alpha) * dice
