import torch
import torch.nn as nn
from torch.nn import functional as F
from src.basic_classifier.classifiers.dna_classifier_back_bone import ResBlock

class DNAClassifier(nn.Module):
    def __init__(self, embedder, rank_classifiers, config):
        super().__init__()
        
        self.embedder = embedder
        self.rank_classifiers = rank_classifiers

        tmp_channel = config["tmp_channel"]
        deepness = config["deepness"]
        embbeder_size = embedder.get_embedding_dim()

        cant_features = embbeder_size

        #Pre procesamiento propio de cada rank
        self.rank_classifiers_pre_process = nn.ModuleList()
        for rank_classifier in rank_classifiers:
            pre_process = nn.ModuleList()
            
            for i in range(deepness):
                pre_process.append(ResBlock(1, tmp_channel))
            
            pre_process.append(nn.Flatten())
            self.rank_classifiers_pre_process.append(nn.Sequential(*pre_process)) 


        #Clasificacion propia de cada rank
        self.rank_classifiers_end = nn.ModuleList()
        for rank_classifier in rank_classifiers:
            end = nn.Sequential(
                                nn.Linear(cant_features, rank_classifier.classification_in_features),
                                rank_classifier.classification_end
                               )
            
            self.rank_classifiers_end.append(end)
            

    def forward(self, sequences):
        
        features = self.embedder(sequences)
        
        pre_processing = []
        for rank_preprocess in self.rank_classifiers_pre_process:
            pre_processing.append(rank_preprocess(features))

        # Use amplified features for predictions
        predictions = []
        for idx, rank_end in enumerate(self.rank_classifiers_end):
            predictions.append(rank_end(pre_processing[idx]))
        
        return predictions

    def get_loss_weight(self, rank_index):
        return self.rank_classifiers[rank_index].loss_weight
    
    def get_ranks(self):
        return len(self.rank_classifiers)

    def get_dynamic_loss_weights(self, current_epoch, total_epochs):
        
        num_ranks = len(self.rank_classifiers_end)
        progress = min(current_epoch / total_epochs, 1.0)
        
        def cuadratic_f(x, offset, roots):
            x -= offset
            result =  1 - (x/roots) * (x/roots) 
            return max(0, result)
        
        # var = (1 - progress) + 0.001
        var = 1
        offset = min(1 , (progress * 1.3))

        weights = []
        for rank_idx in range(num_ranks):

            norm_rank = rank_idx / (num_ranks - 1) # 0 a 1
            coeff = cuadratic_f(norm_rank, offset, var) * 2
            coeff = max(0.2, coeff)
            
            base_weight = self.get_loss_weight(rank_idx)

            weights.append(base_weight * coeff)

        return weights