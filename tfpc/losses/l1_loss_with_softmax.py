"""
This module define a L1 loss with min max norm
"""
# pylint: disable=W0223
import torch
import logging
from utils.utils import select_output_lm


class L1LossSoftmax(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    """
    This class define a specific loss : L1 loss with a min max normalisation of the outputs
    """

    def __init__(self, nb_labels):
        super(L1LossSoftmax, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.softmax = torch.nn.Softmax()
        self.nb_labels = nb_labels

    def forward(self, output, target):
        batch_size = output.shape[0]
        target_onehot = torch.FloatTensor(batch_size, self.nb_labels)
        target_onehot.zero_()
        for num_batch, ind_t in enumerate(target):
            target_onehot[num_batch, ind_t] = 1.0
        logging.debug("A revérifier")
        output = self.softmax(output)
        loss = self.l1_loss(target_onehot, output)
        return loss
