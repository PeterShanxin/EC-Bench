"""
This module define a MSEStructure
"""
# pylint: disable=W0223
import torch


class MSEStructure(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    """
    This class define a specific loss : MSE loss for each token in input
    """

    def __init__(self):
        super(MSEStructure, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, output, target):
        true_target, mask = target
        batch_size = output.shape[0]
        sum_loss = 0
        for num_batch in range(batch_size):
            mask_one_prot = mask[num_batch]
            output_one_prot = output[num_batch]
            target_one_prot = true_target[num_batch]
            # We take only AA with correct target that is defined by the mask
            output_one_prot = output_one_prot[mask_one_prot]
            sum_loss += self.mse(output_one_prot, target_one_prot)
        return sum_loss
