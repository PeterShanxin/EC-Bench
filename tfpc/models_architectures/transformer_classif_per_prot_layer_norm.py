"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm


class TransformerClassifPerProtLayerNorm(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(
        self, transformer_embedder, num_classes, dropout, train_only_last_layer
    ):
        super(TransformerClassifPerProtLayerNorm, self).__init__()
        self.transformer_embedder = transformer_embedder
        self.train_only_last_layer = train_only_last_layer
        if self.train_only_last_layer:
            self.transformer_embedder = self.transformer_embedder.eval()
            for param in self.transformer_embedder.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.norm1 = LayerNorm(num_classes)
        self.classe_embedding = nn.Linear(int(self.d_model), int(num_classes))

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output[:, 0])
        output = self.classe_embedding(output)
        output = self.norm1(output)
        return output
