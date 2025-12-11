import torch
import torch.nn as nn
from src.evaluators.linear_evaluator import LinearProbe

class CombinedModel(nn.Module):
    """
    Combined model with frozen DNABERT encoder and trainable linear probe
    """
    def __init__(self, encoder, probe):
        super().__init__()
        self.encoder = encoder
        self.probe = probe


    def forward(self, sequences):
        latent = self.encoder(sequences)  
        return self.probe(latent)
    
class DNABERTWithProbe(nn.Module):

    def __init__(self, dnabert_embedder, probe):
        super().__init__()
        self.encoder = dnabert_embedder
        self.probe = probe

    def forward(self, sequences):
        with torch.no_grad():              
            latent = self.encoder(sequences)

        return self.probe(latent)