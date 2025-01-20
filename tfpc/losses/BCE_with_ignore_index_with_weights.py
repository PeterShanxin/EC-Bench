"""
This module define a BCEWithIgnoreIndex
"""
# pylint: disable=W0223
import torch
from utils.utils import select_output_lm
from utils.utils import square_pad_sequence


class BCEWithIgnoreIndexWithWeights(
    torch.nn.modules.loss._Loss
):  # pylint: disable=protected-access
    """
    This class define a specific loss : crossentropy loss for each token in input
    """

    def __init__(self, weight_go_class):
        super(BCEWithIgnoreIndexWithWeights, self).__init__()
        self.binary_cross_entropy = torch.nn.BCELoss(weight=weight_go_class)

    def forward(self, output, target):
        # Here target is the list of indice that need to be at proba one
        batch_size, nb_class = output.shape
        target = target[0]
        all_new_target = None
        for num_batch in range(batch_size):
            ind_to_select = target[num_batch]
            # ind_to_select = ind_to_select[ind_to_select != -1]
            one_target = torch.zeros(nb_class)
            one_target[ind_to_select] = 1.0
            one_target = one_target.unsqueeze(0)
            # output_selected = output[num_batch, ind_to_select]
            if all_new_target is None:
                # all_selected = output_selected
                all_new_target = one_target
            else:
                # all_selected = torch.cat((all_selected, output_selected))
                all_new_target = torch.cat((all_new_target, one_target))
        # artificial_target = torch.ones(len(all_selected))
        # artificial_target = artificial_target.double()
        all_new_target = all_new_target.double()
        loss = self.binary_cross_entropy(output, all_new_target)
        return loss
