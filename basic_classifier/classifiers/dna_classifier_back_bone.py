import torch
import torch.nn as nn
import math
from src.basic_classifier.dnabert_embedder import DNABERTEmbedder



class ResBlock(nn.Module):
    
    def __init__(self, channels, tmp_channels):
        super().__init__()
        
        channels = int(channels)

        self.gelu  = nn.GELU()
        self.conv1 = nn.Conv1d(channels, tmp_channels, kernel_size=1)
        self.bn1   = nn.BatchNorm1d(tmp_channels)
        
        self.conv2 = nn.Conv1d(tmp_channels, tmp_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(tmp_channels)
        
        self.conv3 = nn.Conv1d(tmp_channels, channels, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        
        return self.gelu(out)

class CNNBackbone(nn.Module):
    
    def __init__(self, configuration):
        super().__init__()
        
        first_channel_size = configuration["first_channel_size"] 
        self.deepness = configuration["deepness"]
        tmp_channels = configuration["tmp_channels"]

        assert self.deepness >= 0

        backbone_list = nn.ModuleList()

        # Primer bloque
        backbone_list.append(nn.Conv1d(1, first_channel_size, kernel_size=3, padding=1))
        backbone_list.append(nn.GELU())
        backbone_list.append(ResBlock(first_channel_size, tmp_channels))
        channel_size = first_channel_size

        for d in range(self.deepness):
            backbone_list.append(nn.MaxPool1d(kernel_size=2, stride=2)) #divido por 2 el size
            backbone_list.append(nn.Conv1d(channel_size, channel_size * 2, kernel_size=3, padding=1))
            backbone_list.append(nn.GELU())
            backbone_list.append(ResBlock(channel_size * 2, tmp_channels))

            channel_size = channel_size * 2

        self.backbone = nn.Sequential(*backbone_list)

        self.backbone_channels = channel_size

    def forward(self, x):
        assert (x.shape[1] % (2**self.deepness)) == 0
        
        x = x.unsqueeze(1)
        return self.backbone(x)
        
    def get_backbone_length(self, initial_length):
        assert (initial_length % (2**self.deepness)) == 0
        
        return initial_length / (2**self.deepness)

    def get_backbone_channels(self):
        return self.backbone_channels


        