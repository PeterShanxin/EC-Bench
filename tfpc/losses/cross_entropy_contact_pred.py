"""
This module define a CrossEntropyContactPred
"""
# pylint: disable=W0223
import torch
from utils.utils import select_output_lm
from utils.utils import square_pad_sequence


class CrossEntropyContactPred(
    torch.nn.modules.loss._Loss
):  # pylint: disable=protected-access
    """
    This class define a specific loss : crossentropy loss for each token in input
    """

    def __init__(self):
        super(CrossEntropyContactPred, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        target = square_pad_sequence(target, padding_value=-1)
        batch_size, max_prot_len1, max_prot_len2, nb_class = output.shape
        output = output.view(batch_size * max_prot_len1 * max_prot_len2, nb_class)
        target = target.view(batch_size * max_prot_len1 * max_prot_len2)
        mask = target != -1
        target = target[mask]
        output = output[mask]
        loss = self.cross_entropy(output, target)
        return loss
