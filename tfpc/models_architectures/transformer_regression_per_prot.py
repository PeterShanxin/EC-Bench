"""
This module create TransformerRegressionPerProt
"""
# pylint: disable=W0223
import torch.nn as nn


class TransformerRegressionPerProt(nn.Module):
    """
    This class define a Transformer model that predict a number for an entire protein
    """

    def __init__(self, transformer_embedder, dropout):
        super(TransformerRegressionPerProt, self).__init__()
        self.transformer_embedder = transformer_embedder

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.classe_embedding = nn.Linear(int(self.d_model), 1)

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output[:, 0])
        output = self.classe_embedding(output)
        return output
