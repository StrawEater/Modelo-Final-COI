import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):

    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)
    

class SimpleMLPProbe(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
    


class BetterMLPProbe(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)



class CosineClassifier(nn.Module):
    """
    Usa similitud coseno en lugar de producto interno
    Mejor para high-dimensional spaces y muchas clases
    """
    def __init__(self, latent_dim, num_classes, scale=20.0):
        super().__init__()
        # Embeddings de prototipos por clase
        self.weight = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale  # Factor de escala para softmax
    
    def forward(self, x):
        # Normalizar entrada
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Normalizar pesos (prototipos)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Similitud coseno
        logits = self.scale * torch.matmul(x_norm, w_norm.t())
        
        return logits