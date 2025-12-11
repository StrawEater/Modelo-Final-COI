import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedModelWithDecoder(nn.Module):

    def __init__(self, dnabert, encoder, decoders_dict, classifiers_dict, global_decoder):
        super().__init__()
        self.dnabert = dnabert
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders_dict)
        self.classifiers = nn.ModuleDict(classifiers_dict)
        self.global_decoder = global_decoder

    def forward(self, sequences):
        with torch.no_grad():
            emb = self.dnabert(sequences)

        z = self.encoder(emb)

        logits_out = {}
        recon_out  = {}

        for taxon in self.classifiers:
            logits_out[taxon] = self.classifiers[taxon](z)
            recon_out[taxon]  = self.decoders[taxon](z)

        global_rec = self.global_decoder(z)

        return {
            "z": z,
            "logits": logits_out,
            "recon": recon_out,
            "global_recon": global_rec
        }


class HierarchicalCombinedModel(nn.Module):
    """
    Modelo que respeta la jerarquía taxonómica:
    - Cada nivel usa las predicciones del nivel anterior
    - Previene inconsistencias (ej: species de un genus que no existe)
    """

    def __init__(self, dnabert, encoder, decoders_dict, classifiers_dict, 
                 global_decoder, taxonomy_hierarchy):
        """
        Args:
            taxonomy_hierarchy: Dict[str, Dict[int, List[int]]]
                Ej: {
                    'order': {0: [0,1,2], 1: [3,4]},  # phylum 0 tiene orders 0,1,2
                    'family': {0: [0,1], 1: [2,3,4]}, # order 0 tiene families 0,1
                    ...
                }
        """
        super().__init__()
        self.dnabert = dnabert
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders_dict)
        self.classifiers = nn.ModuleDict(classifiers_dict)
        self.global_decoder = global_decoder
        self.taxonomy_hierarchy = taxonomy_hierarchy
        
        # Orden de predicción
        self.taxon_order = ['phylum', 'order', 'family', 'genus', 'species']

    def forward(self, sequences, use_hierarchy=True):
        """
        Args:
            use_hierarchy: Si True, aplica máscaras jerárquicas
        """
        with torch.no_grad():
            emb = self.dnabert(sequences)

        z = self.encoder(emb)

        logits_out = {}
        recon_out = {}
        
        if not use_hierarchy:
            # Modo legacy: clasificadores independientes
            for taxon in self.classifiers:
                logits_out[taxon] = self.classifiers[taxon](z)
                recon_out[taxon] = self.decoders[taxon](z)
        else:
            # Modo jerárquico
            parent_preds = None
            parent_taxon = None
            
            for taxon in self.taxon_order:
                if taxon not in self.classifiers:
                    continue
                
                # Obtener logits sin restricción
                logits = self.classifiers[taxon](z)
                
                # Aplicar máscara jerárquica
                if parent_preds is not None and taxon in self.taxonomy_hierarchy:
                    logits = self._apply_hierarchical_mask(
                        logits, parent_preds, parent_taxon, taxon
                    )
                
                logits_out[taxon] = logits
                recon_out[taxon] = self.decoders[taxon](z)
                
                # Guardar predicción para siguiente nivel
                parent_preds = torch.argmax(logits, dim=1)
                parent_taxon = taxon

        global_rec = self.global_decoder(z)

        return {
            "z": z,
            "logits": logits_out,
            "recon": recon_out,
            "global_recon": global_rec,
            "emb": emb
        }
    
    def _apply_hierarchical_mask(self, logits, parent_preds, parent_taxon, current_taxon):
        """
        Aplica máscara para que solo sean válidas las clases consistentes
        con la predicción del nivel padre.
        
        Args:
            logits: [B, num_classes] - logits del nivel actual
            parent_preds: [B] - predicciones del nivel padre
            parent_taxon: str - nombre del taxon padre
            current_taxon: str - nombre del taxon actual
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # Crear máscara (True = clase válida)
        mask = torch.zeros(batch_size, num_classes, 
                          dtype=torch.bool, device=logits.device)
        
        # Para cada ejemplo en el batch
        for i in range(batch_size):
            parent_class = parent_preds[i].item()
            
            # Obtener clases válidas para este padre
            valid_classes = self.taxonomy_hierarchy.get(current_taxon, {}).get(parent_class, [])
            
            if valid_classes:
                mask[i, valid_classes] = True
            else:
                # Si no hay info, permitir todas
                mask[i, :] = True
        
        # Aplicar máscara: -inf para clases inválidas
        masked_logits = logits.clone()
        masked_logits[~mask] = float('-inf')
        
        return masked_logits


class HierarchicalCombinedModelFixed(nn.Module):
    """
    Versión mejorada que evita loss = inf causado por máscaras jerárquicas
    """
    
    def __init__(self, dnabert, encoder, decoders_dict, classifiers_dict, 
                 global_decoder, taxonomy_hierarchy=None):
        super().__init__()
        self.dnabert = dnabert
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders_dict)
        self.classifiers = nn.ModuleDict(classifiers_dict)
        self.global_decoder = global_decoder
        self.taxonomy_hierarchy = taxonomy_hierarchy
        
        # ⭐ NUEVO: Parámetro para controlar cuán restrictiva es la máscara
        self.mask_value = -100  # En lugar de -inf
        
        print("✅ Modelo jerárquico creado con máscaras suaves (no -inf)")

    def apply_hierarchical_mask_soft(self, logits_dict, parent_predictions):
        """
        Aplica máscaras jerárquicas SIN usar -inf
        
        Args:
            logits_dict: Diccionario de logits {taxon: tensor}
            parent_predictions: Predicciones del nivel padre
            
        Returns:
            logits_dict con máscaras aplicadas
        """
        if self.taxonomy_hierarchy is None:
            return logits_dict
        
        taxon_order = ['phylum', 'order', 'family', 'genus', 'species']
        masked_logits = {}
        
        masked_logits['phylum'] = logits_dict['phylum']  # Phylum no tiene máscara
        
        for i in range(1, len(taxon_order)):
            parent_taxon = taxon_order[i-1]
            child_taxon = taxon_order[i]
            
            if child_taxon not in logits_dict:
                continue
            
            child_logits = logits_dict[child_taxon].clone()
            
            # Obtener predicción del padre
            parent_pred = parent_predictions[parent_taxon]  # [batch_size]
            
            # Para cada ejemplo en el batch
            batch_size = child_logits.shape[0]
            n_child_classes = child_logits.shape[1]
            
            # Crear máscara (True = válido, False = inválido)
            mask = torch.zeros_like(child_logits, dtype=torch.bool)
            
            for b in range(batch_size):
                parent_class = parent_pred[b].item()
                
                # Obtener clases válidas del hijo
                if child_taxon in self.taxonomy_hierarchy:
                    if parent_class in self.taxonomy_hierarchy[child_taxon]:
                        valid_children = self.taxonomy_hierarchy[child_taxon][parent_class]
                        
                        # Marcar clases válidas
                        for valid_child in valid_children:
                            if valid_child < n_child_classes:
                                mask[b, valid_child] = True
                
                # Si no hay clases válidas, permitir todas (fallback)
                if not mask[b].any():
                    mask[b] = True
            
            # ✅ CLAVE: Usar valor muy negativo en lugar de -inf
            child_logits[~mask] = self.mask_value  # -1e9 en lugar de -inf
            
            masked_logits[child_taxon] = child_logits
        
        return masked_logits

    def forward(self, sequences, use_hierarchy=True, true_labels=None):
        """
        Forward pass con opción de desactivar jerarquía
        
        Args:
            sequences: Lista de secuencias de DNA
            use_hierarchy: Si False, no aplica máscaras jerárquicas
        """
        # DNABERT embeddings (congelado)
        with torch.no_grad():
            emb = self.dnabert(sequences)
        
        # Encoder
        z = self.encoder(emb)
        
        # Classifiers (sin máscaras primero)
        logits_out = {}
        for taxon in self.classifiers:
            logits_out[taxon] = self.classifiers[taxon](z)
        
        # Decoders
        recon_out = {}
        for taxon in self.decoders:
            recon_out[taxon] = self.decoders[taxon](z)
        
        # Global decoder
        global_rec = self.global_decoder(z)
        
        # ⭐ Aplicar máscaras jerárquicas (OPCIONAL)
        if use_hierarchy and self.taxonomy_hierarchy is not None:
            
            # ✅ CLAVE: Usar true_labels si están disponibles
            if true_labels is not None:
                # TRAINING: Usar ground truth
                parent_predictions = {}
                taxon_order = ['phylum', 'order', 'family', 'genus', 'species']
                
                for taxon in taxon_order:
                    if taxon in true_labels:
                        parent_predictions[taxon] = true_labels[taxon]
                    elif taxon in logits_out:
                        parent_predictions[taxon] = torch.argmax(
                            logits_out[taxon], dim=1
                        )
            else:
                # INFERENCE: Usar predicciones del modelo
                parent_predictions = {}
                with torch.no_grad():
                    for taxon in logits_out:
                        parent_predictions[taxon] = torch.argmax(
                            logits_out[taxon], dim=1
                        )
            
            # Aplicar máscaras usando parent_predictions
            logits_out = self.apply_hierarchical_mask_soft(
                logits_out, parent_predictions
            )
        
        return {
            "z": z,
            "logits": logits_out,
            "recon": recon_out,
            "global_recon": global_rec,
            "emb": emb
        }



