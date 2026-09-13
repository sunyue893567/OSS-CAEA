"""Query-to-image embedding alignment used by PSAH."""
import torch
from torch import nn
from torch.nn import functional as F


class UnifiedEmbeddingAlignment(nn.Module):
    """Embedding alignment for PSAH.

    Research attribution and Apache-2.0 terms: docs/THIRD-PARTY-NOTICES.md.

    Query MLP -> multihead cross attention to CLIP memory -> residual/norm ->
    L2-normalized text dot product, temperature 50; zero no-object embedding.
    """
    def __init__(self, query_dim, clip_dim, heads=8, temperature=50.):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(query_dim, clip_dim // 2), nn.ReLU(),
                                    nn.Linear(clip_dim // 2, clip_dim))
        # MultiheadAttention contains independent Q/K/V linear projections.
        self.cross_attention = nn.MultiheadAttention(clip_dim, heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(clip_dim)
        self.temperature = temperature

    def forward(self, queries, clip_memory, text):
        q = self.linear(queries)
        aligned = self.cross_attention(q, clip_memory.to(q.dtype), clip_memory.to(q.dtype), need_weights=False)[0]
        aligned = self.norm(q + aligned)
        aligned = self.temperature * F.normalize(aligned, dim=-1, eps=1e-6)
        text = F.normalize(text, dim=-1, eps=1e-6).to(aligned.dtype)
        scores = torch.einsum('bqc,kc->bqk', aligned, text)
        return torch.cat((scores, scores.new_zeros(*scores.shape[:-1], 1)), dim=-1)
