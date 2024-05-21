# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

import torch
import torch.nn as nn

class MLPFeatureProjModel(nn.Module):
    """SD model with feature prompt"""

    # torch.Size([1, 49, 1536]) -> torch.Size([1, 49, 768])
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=1536):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, id_embeds):
        feature_tokens = self.proj(id_embeds)

        return feature_tokens