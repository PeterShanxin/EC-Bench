"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
import torch


class TransformerClassifPerProtNoCls(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(self, transformer_embedder, num_classes, dropout):
        super(TransformerClassifPerProtNoCls, self).__init__()
        self.transformer_embedder = transformer_embedder

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.classe_embedding = nn.Linear(int(self.d_model), int(num_classes))

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output[:, 1:])
        output = self.classe_embedding(output)
        output = torch.mean(output, dim=1)
        return output
