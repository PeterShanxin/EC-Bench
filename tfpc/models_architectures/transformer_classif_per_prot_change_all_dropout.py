"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import logging
import torch.nn as nn
from torch.nn.modules.dropout import Dropout


class TransformerClassifPerProtChangeDropout(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(self, transformer_embedder, num_classes, dropout):
        super(TransformerClassifPerProtChangeDropout, self).__init__()
        self.transformer_embedder = transformer_embedder

        # We redifine the dropout of the Transformer
        for layer in self.transformer_embedder.transformer_encoder.layers:
            logging.debug(
                "Le dropout actuel est de %s et je vais le changé en %s",
                layer.dropout.p,
                dropout,
            )
            layer.dropout = Dropout(dropout)
            layer.dropout1 = Dropout(dropout)
            layer.dropout2 = Dropout(dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.classe_embedding = nn.Linear(int(self.d_model), int(num_classes))

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output[:, 0])
        output = self.classe_embedding(output)
        return output
