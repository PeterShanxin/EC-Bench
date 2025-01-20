"""
This module define a CrossEntropyPerToken
"""
# pylint: disable=W0223
import torch
import logging
from utils.utils import select_output_lm


class CrossEntropyPerProtWeighted(
    torch.nn.modules.loss._Loss
):  # pylint: disable=protected-access
    """
    This class define a specific loss : crossentropy loss for each token in input
    """

    def __init__(self):
        super(CrossEntropyPerProtWeighted, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, target):
        true_target, weight = target
        logging.info(output.shape)
        loss = torch.matmul(weight, self.cross_entropy(output, true_target))
        return loss
