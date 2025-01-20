"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn


class TransformerClassifPerProtMoreExpressive(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(self, transformer_embedder, num_classes, dropout, num_inter):
        super(TransformerClassifPerProtMoreExpressive, self).__init__()
        self.transformer_embedder = transformer_embedder

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.project_intern = nn.Linear(int(self.d_model), num_inter)
        self.relu = nn.ReLU()
        self.classe_embedding = nn.Linear(num_inter, int(num_classes))

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output[:, 0])
        output = self.project_intern(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.classe_embedding(output)
        return output
