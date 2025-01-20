"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn


class TransformerClassifPerProtMultipleLayer(nn.Module):
    def __init__(self, originalModel):
        super(TransformerClassifPerProtMultipleLayer, self).__init__()
        # Partie multi level embedding
        NB_LAYER = len(originalModel.transformer_encoder.layers)
        self.layers = []
        for k in range(NB_LAYER):
            self.layers.append(originalModel.transformer_encoder.layers[k])

        self.combine = nn.Parameter(torch.rand(NB_LAYER))
        """
        self.combine = nn.Parameter(
            torch.tensor([0.01 for _ in range(NB_LAYER - 1)] + [0.99])
        )
        """

    def forward(self, output):
        output_layers = None
        for k in range(len(self.layers)):
            if k == 0:
                preced_output = output
            else:
                preced_output = output_layers[k - 1]
            output_layer_i = self.layers[k](preced_output)
            output_layer_i = output_layer_i.unsqueeze(0)
            if output_layers is None:
                output_layers = output_layer_i
            else:
                output_layers = torch.cat((output_layers, output_layer_i))
        softmax_combine = torch.nn.Softmax()(self.combine)
        total = output_layers[0] * softmax_combine[0]
        nb_layer = output_layers.shape[0]
        for indice_element in range(1, nb_layer):
            total += softmax_combine[indice_element] * output_layers[indice_element]
        return total
