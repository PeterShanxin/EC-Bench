"""
This module create TransformerLM
"""
# pylint: disable=W0223
import torch.nn as nn
import torch


class TransformerLM(nn.Module):
    """
    This class try to predict true token in the ouput
    """

    def __init__(self, transformer_embedder):
        super(TransformerLM, self).__init__()
        self.transformer_embedder = transformer_embedder
        ntoken = self.transformer_embedder.encoder.num_embeddings
        self.decoder_bias = nn.Parameter(torch.zeros(ntoken))

    def forward(self, src):
        output = self.transformer_embedder(src)
        output = (
            torch.nn.functional.linear(output, self.transformer_embedder.encoder.weight)
            + self.decoder_bias
        )
        return output
