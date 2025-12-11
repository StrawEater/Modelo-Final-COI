import torch
import torch.nn as nn


class SimpleEmbeddingEncoder(nn.Module):
    def __init__(self, embed_dim=768, latent_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, latent_dim)
        )
        self.latent_dim = latent_dim

    def forward(self, emb):
        return self.net(emb)
    

class DeepMLPEmbeddingEncoder(nn.Module):
    def __init__(self, embed_dim=768, latent_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, emb):
        return self.net(emb)