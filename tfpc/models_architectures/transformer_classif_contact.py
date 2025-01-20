"""
This module create TransformerClassifContact
"""
# pylint: disable=W0223
import torch.nn as nn


class TransformerClassifContact(nn.Module):
    """
    This class define a Transformer model that classify each couple of amino acid of a protein
    """

    def __init__(self, transformer_embedder, dropout):
        super(TransformerClassifContact, self).__init__()
        self.transformer_embedder = transformer_embedder

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.classe_embedding = nn.Linear(int(self.d_model), 256)
        self.proj_embed = nn.Linear(256, 2)

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output)
        output = self.classe_embedding(output)
        # We multiply the embedding of each amino acid between each other to test the L^2 contact
        prod = output[:, :, None, :] * output[:, None, :, :]
        prod_embed = self.proj_embed(prod)
        return prod_embed
