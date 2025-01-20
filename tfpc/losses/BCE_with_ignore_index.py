"""
This module define a BCEWithIgnoreIndex
"""
# pylint: disable=W0223
import torch
from utils.utils import select_output_lm
from utils.utils import square_pad_sequence


class BCEWithIgnoreIndex(
    torch.nn.modules.loss._Loss
):  # pylint: disable=protected-access
    """
    This class define a specific loss : crossentropy loss for each token in input
    """

    def __init__(self):
        super(BCEWithIgnoreIndex, self).__init__()
        self.binary_cross_entropy = torch.nn.BCELoss()

    def forward(self, output, indices_only_leaves_and_all_labels):
        # Here target is the list of indice that need to be at proba one
        batch_size, nb_class = output.shape
        target = torch.zeros(batch_size, nb_class)
        indices_target = indices_only_leaves_and_all_labels[
            1
        ]  # Get all labels leaves + appended parents
        for num_batch in range(batch_size):
            ind_to_select = indices_target[num_batch]
            ind_to_select = ind_to_select[ind_to_select != -1]
            target[num_batch, ind_to_select] = 1.0

        loss = self.binary_cross_entropy(output, target)
        return loss
