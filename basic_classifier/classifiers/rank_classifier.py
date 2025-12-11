import torch
import torch.nn as nn
from torch.nn import functional as F

class RankClassiferEnd(nn.Module):
    
    def __init__(self, in_features, num_classes):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):        
        return self.classifier(x)

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.scale = scale  # Temperature parameter (learnable or fixed)
    
    def forward(self, x):
        # Normalize features
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Normalize weights
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity * scale
        logits = F.linear(x_norm, w_norm) * self.scale
        return logits

class RankClassiferCosine(nn.Module):
    
    def __init__(self, in_features, num_classes):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.LayerNorm(256),
            CosineClassifier(256, num_classes)
        )

    def forward(self, x):        
        return self.classifier(x)


class RankClassifer():
    def __init__(self, classification_in_features, classification_end, num_classes, loss_weight):
        
        self.classification_end = classification_end
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.classification_in_features = classification_in_features
    
    def calculate_class_probability(self):
        return self.classification_end 
    
