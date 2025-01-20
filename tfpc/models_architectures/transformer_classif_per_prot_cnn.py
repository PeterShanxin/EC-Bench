"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm


class TransformerClassifPerProtConv(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(
        self,
        transformer_embedder,
        num_classes,
        dropout,
        nb_reduce,
        nb_of_conv_map,
        train_only_last_layer,
    ):
        super(TransformerClassifPerProtConv, self).__init__()

        self.transformer_embedder = transformer_embedder

        self.d_model = self.transformer_embedder.d_model
        self.nb_of_conv_map = nb_of_conv_map
        self.train_only_last_layer = train_only_last_layer
        self.nb_reduce = nb_reduce
        self.reduce_dim = nn.Linear(int(self.d_model), int(self.nb_reduce))
        # non-square kernels and unequal stride and with padding and dilation
        self.convLayer = nn.Conv1d(
            in_channels=self.nb_reduce,
            out_channels=self.nb_of_conv_map,
            kernel_size=200,
            stride=1,
            padding="same",
        )
        self.classe_embedding = nn.Linear(int(self.nb_of_conv_map), int(num_classes))

    def forward(self, src):
        if self.train_only_last_layer:
            self.transformer_embedder = self.transformer_embedder.eval()
        output = self.transformer_embedder(src)
        current_shape = output.shape
        output = output.reshape((current_shape[0] * current_shape[1], current_shape[2]))
        output = self.reduce_dim(output)
        output = output.reshape((current_shape[0], current_shape[1], self.nb_reduce))
        output = torch.transpose(output, 1, 2)
        output = self.convLayer(output)
        output = torch.transpose(output, 1, 2)
        output, _ = torch.max(output, dim=1)
        output = self.classe_embedding(output)
        return output
