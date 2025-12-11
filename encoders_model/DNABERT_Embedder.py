import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DNABERTEmbedder(nn.Module):
    """
    A PyTorch module that uses DNABERT to generate frozen embeddings.
    The DNABERT weights are not trainable - only use for feature extraction.
    """
    def __init__(self, model_name, max_length=750, device="cuda"):

        """
        Args:
            model_name: HuggingFace model identifier for DNABERT
            pooling: How to pool token embeddings ("mean", "cls", or "max")
        """

        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)

        self.load_from_route()
        self.freeze_network()
        self.model.to(self.device)

    def load_from_route(self):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

    def freeze_network(self):
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.model.config.hidden_size


    def forward(self, sequences):
        """
        Args:
            sequences: List of DNA sequences (strings) or batch of sequences

        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        # Ensure we're in eval mode and not computing gradients
        with torch.no_grad():

            # Tokenize sequences
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length # Maxima longitud de las secuencias de entrenamiento (VER QUE ONDA)
            ).to(self.device)

            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Get DNABERT outputs
            outputs = self.model(**inputs)
            emb = outputs[0].mean(dim=1)   # shape [B, H]

            return emb

    def get_embedding_dim(self):
        """Returns the dimension of the output embeddings"""
        return self.embedding_dim
