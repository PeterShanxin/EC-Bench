"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
import torch


class TransformerClassifPerProtLikeEnsemble(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(self, transformer_embedder, num_classes, dropout, nb_paquet):
        super(TransformerClassifPerProtLikeEnsemble, self).__init__()
        self.transformer_embedder = transformer_embedder

        self.nb_paquet = nb_paquet
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.classe_embedding = nn.Linear(
            int(self.d_model), nb_paquet * int(num_classes)
        )

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = self.dropout(output[:, 0])
        output = self.classe_embedding(output)
        bs = output.shape[0]
        new_shape = (bs, self.nb_paquet, self.num_classes)
        output = output.view(new_shape)
        output = torch.mean(output, dim=1)
        return output
