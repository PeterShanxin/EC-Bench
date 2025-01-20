"""
This module define a converter
"""
import logging
import torch
from converter.generic_converter import GenericConverter
from utils.utils import square_pad_sequence


class ConverterMultiLabelProba(GenericConverter):
    """
    This class convert proba output to prediction
    """

    def convert_output_and_target(self, output, target):
        bs = output.shape[0]
        mask = target != -1
        new_target = []
        for num_batch in range(bs):
            new_target.append(target[num_batch][mask[num_batch]].tolist())
        return output.tolist(), new_target
