"""
This module create TransformerStructurePrediction
"""
# pylint: disable=W0223
import torch.nn as nn


class TransformerStructurePrediction(nn.Module):
    """
    This class define a Transformer model that predict 3 angles for each amino acid of a protein
    """

    def __init__(self, transformer_embedder, dropout):
        super(TransformerStructurePrediction, self).__init__()
        self.transformer_embedder = transformer_embedder

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.classe_embedding = nn.Linear(int(self.d_model), 3)

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output)
        output = self.classe_embedding(output)
        return output
