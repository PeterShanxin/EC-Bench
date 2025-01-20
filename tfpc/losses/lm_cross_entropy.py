"""
This module define a LMCrossEntropy
"""
# pylint: disable=W0223
import torch
from utils.utils import select_output_lm


class LMCrossEntropy(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    """
    This class define a specific loss for language model, that do the cross
        entropy loss only on mask token in input
    """

    def __init__(self, weight=None):
        super(LMCrossEntropy, self).__init__()
        if weight is not None:
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        true_target, indice_mask = target
        output = select_output_lm(output, indice_mask)
        loss = self.cross_entropy(output, true_target)
        return loss
