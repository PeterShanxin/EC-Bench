"""
This module define a 
"""
# pylint: disable=W0223
import torch
import logging
from utils.utils import select_output_lm
import torch


class SumLossJustWeights(
    torch.nn.modules.loss._Loss
):  # pylint: disable=protected-access
    """
    This class define a specific loss :
    """

    def __init__(self):
        super(SumLossJustWeights, self).__init__()

    def forward(self, output, target):
        loss = torch.tensor(0.0)
        for num_batch, t in enumerate(target):
            loss += torch.cat(
                (output[num_batch][:t], output[num_batch][t + 1 :])
            ).mean()
            loss -= output[num_batch][t]
        return loss
