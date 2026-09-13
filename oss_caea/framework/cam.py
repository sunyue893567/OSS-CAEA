"""Collaborative Attention Module (CAM), PDF Figure 3 and Eqs. (4)-(7).

GeoText Prompts are projected by text_proj and averaged into a semantic vector.
Signed cosine weighting preserves the spatial token sequence.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CAM(nn.Module):
    """Collaborative Attention Module (CAM)."""
    def __init__(self, visual_dim, depth_dim, text_dim, channels):
        super().__init__()
        self.visual_proj = nn.Conv2d(visual_dim, channels, 1)
        self.depth_proj = nn.Conv2d(depth_dim, channels, 1)
        self.text_proj = nn.Linear(text_dim, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, visual, depth, text):
        # Figure 2: VFM + DA V2. Spatial/channel alignment precedes Eq. (4).
        depth = F.interpolate(depth, visual.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.visual_proj(visual) + self.depth_proj(depth)
        tokens = self.norm(fused.flatten(2).transpose(1, 2))
        geotext_prompts = self.text_proj(text)
        global_geotext_prompt = geotext_prompts.mean(dim=0)
        weights = F.cosine_similarity(tokens.float(), global_geotext_prompt.float()[None, None], dim=-1, eps=1e-6)
        # Retain signed cosine weights and every spatial token (Eq. 7).
        result = tokens * weights.to(tokens.dtype).unsqueeze(-1)
        return result.transpose(1, 2).reshape_as(fused)
