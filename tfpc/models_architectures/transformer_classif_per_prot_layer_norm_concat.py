"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm


class TransformerClassifPerProtLayerNormConcat(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(
        self,
        transformer_embedder,
        num_classes,
        dropout,
        nb_reduce,
        max_size,
        train_only_last_layer,
    ):
        super(TransformerClassifPerProtLayerNormConcat, self).__init__()

        self.transformer_embedder = transformer_embedder
        self.train_only_last_layer = train_only_last_layer
        if self.train_only_last_layer:
            # print("Parameter requires_grad to false")
            self.transformer_embedder = self.transformer_embedder.eval()
            for param in self.transformer_embedder.parameters():
                param.requires_grad = False
        self.max_size = max_size
        self.d_model = self.transformer_embedder.d_model
        self.nb_reduce = nb_reduce
        self.reduce_dim = nn.Linear(int(self.d_model), int(self.nb_reduce))
        self.norm1 = LayerNorm(num_classes)
        self.classe_embedding = nn.Linear(
            int(self.max_size * self.nb_reduce), int(num_classes)
        )

    def forward(self, src):
        if self.train_only_last_layer:
            self.transformer_embedder = self.transformer_embedder.eval()
        output = self.transformer_embedder(src)
        current_shape = output.shape
        output = output.reshape((current_shape[0] * current_shape[1], current_shape[2]))
        output = self.reduce_dim(output)
        output = output.reshape((current_shape[0], current_shape[1] * self.nb_reduce))
        output = nn.functional.pad(
            input=output,
            pad=(0, self.max_size * self.nb_reduce - output.shape[1]),
            mode="constant",
            value=0,
        )
        output = self.classe_embedding(output)
        output = self.norm1(output)
        return output
