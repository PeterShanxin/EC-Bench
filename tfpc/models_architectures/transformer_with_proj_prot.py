"""
This module create TransformerWithProjProt
"""
# pylint: disable=W0223
import torch.nn as nn
import torch


class TransformerWithProjProt(nn.Module):
    """
    This class allow to load the pretrained weight for the projection in order
    to predict amino acid from corrupted sequence.
    transformer lm retrain bias and is for a specific task, this class is for the
    generic pre training model.
    """

    def __init__(self, transformer_embedder, originalModel):
        super(TransformerWithProjProt, self).__init__()
        self.transformer_embedder = transformer_embedder
        self.decoder_bias = originalModel.module.decoder_bias

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = (
            torch.nn.functional.linear(output, self.transformer_embedder.encoder.weight)
            + self.decoder_bias
        )
        return output
