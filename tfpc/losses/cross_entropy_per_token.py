"""
This module define a CrossEntropyPerToken
"""
# pylint: disable=W0223
import torch
from utils.utils import select_output_lm


class CrossEntropyPerToken(
    torch.nn.modules.loss._Loss
):  # pylint: disable=protected-access
    """
    This class define a specific loss : crossentropy loss for each token in input
    """

    def __init__(self, weight=None):
        super(CrossEntropyPerToken, self).__init__()
        if weight is not None:
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        mask = target != -1
        target = target[mask]
        output = output[mask]
        loss = self.cross_entropy(output, target)
        return loss
