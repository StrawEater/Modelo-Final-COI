import torch
import torch.nn as nn

class SequenceDecoder(nn.Module):
    def __init__(self, latent_dim=128, seq_len=657, vocab_size=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size   # <<--- asignar aquí

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, seq_len * vocab_size)
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, self.seq_len, self.vocab_size)  # ahora self.vocab_size existe
