import torch
import torch.nn as nn

class EmbeddingDecoder(nn.Module):
    def __init__(self, latent_dim=128, embedding_dim=768, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, embedding_dim)
        )

    def forward(self, z):
        return self.net(z)