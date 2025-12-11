import torch
import torch.nn as nn
from src.encoders_model.DNABERT_Embedder import DNABERTEmbedder


class SimpleEncoder(nn.Module):
  def __init__(self, latent_dim=128, dropout=0.0, dnabert_path=None, max_length=512):

    super().__init__()

    self.embedder = DNABERTEmbedder(dnabert_path, max_length=max_length)
    embed_dim = self.embedder.get_embedding_dim()

    self.net = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, latent_dim),
        )

    self.embed_dim = embed_dim
    self.latent_dim = latent_dim


  def forward(self, sequences):
    embeddings = self.embedder(sequences)

    latent = self.net(embeddings)

    return latent

class DeepMLPEncoder(nn.Module):
    def __init__(self, dnabert_path=None, latent_dim=128, max_length=512, dropout=0.0):
        super().__init__()
        self.embedder = DNABERTEmbedder(dnabert_path, max_length=max_length)
        embed_dim = self.embedder.get_embedding_dim()

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

    def forward(self, sequences):
        embeddings = self.embedder(sequences)
        return self.net(embeddings)

class ResidualMLPEncoder(nn.Module):
    def __init__(self, dnabert_path=None, latent_dim=128, max_length=512):
        super().__init__()
        self.embedder = DNABERTEmbedder(dnabert_path, max_length=max_length)
        embed_dim = self.embedder.get_embedding_dim()

        self.fc1 = nn.Linear(embed_dim, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)

        self.fc3 = nn.Linear(512, latent_dim)
        self.ln3 = nn.LayerNorm(latent_dim)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()

        self.latent_dim = latent_dim

    def forward(self, sequences):
        x = self.embedder(sequences)


        residual = self.fc1(x)
        x = self.ln1(residual)
        x = self.activation(x)
        x = self.dropout(x)


        x_res = self.fc2(x)
        x = self.ln2(x + x_res)
        x = self.activation(x)
        x = self.dropout(x)


        x = self.fc3(x)
        x = self.ln3(x)

        return x