"""
This module define a converter
"""
import torch
from converter.generic_converter import GenericConverter
from utils.utils import select_output_lm


class ConverterL5MostLikelyMediumAndLong(GenericConverter):
    """
    This class select L/5 most likely contact pred
    """

    def convert_output_and_target(self, output, target):
        target_pad = torch.nn.utils.rnn.pad_sequence(
            target, padding_value=-1, batch_first=True
        )
        valid_mask = target_pad != -1
        seqpos = torch.arange(valid_mask.size(1))
        x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
        valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)

        final_target = None
        final_pred = None
        for pred, lab, mask in zip(output, target, valid_mask):
            len_prot = lab.shape[0]
            pred = pred[:len_prot]
            masked_prob = (pred * mask).view(-1)
            most_likely_indice = masked_prob.topk(len_prot // 5, sorted=False).indices
            most_likely = masked_prob[most_likely_indice].view(-1)
            target_likely = lab[most_likely_indice]
            if final_target is None:
                final_target = target_likely
                final_pred = most_likely
            else:
                final_target = torch.concatenate((final_target, target_likely))
                final_pred = torch.concatenate((final_pred, most_likely))

        return final_pred, final_target
