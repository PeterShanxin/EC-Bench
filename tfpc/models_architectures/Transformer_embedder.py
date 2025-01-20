"""
This module create TransformerEmbedder
"""
# pylint: disable=W0223
import torch.nn as nn
import torch


class TransformerEmbedder(nn.Module):
    """
    This class define the Transformer embedder that is the core of all other Transformer
    for the different task.

    This take an input protein and give a representation for each amino acid of the sequence
    """

    def __init__(self, originalModel, pad_index):
        originalModel = originalModel.module
        super(TransformerEmbedder, self).__init__()
        self.d_model = originalModel.d_model
        self.encoder = originalModel.encoder
        self.pos_encoder = originalModel.pos_encoder
        self.transformer_encoder = originalModel.transformer_encoder
        self.pad_index = pad_index

    def forward(self, src):
        mask_pad_token = src == self.pad_index
        output = self.encoder(src)
        output = output.transpose(0, 1)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output, src_key_padding_mask=mask_pad_token)
        output = output.transpose(0, 1)
        return output
