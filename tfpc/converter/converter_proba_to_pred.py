"""
This module define a converter
"""
import logging
import torch
from converter.generic_converter import GenericConverter
from utils.utils import square_pad_sequence


class ConverterProbaToPred(GenericConverter):
    """
    This class convert proba output to prediction
    """

    def convert_output_and_target(self, output, target):
        if isinstance(target, list):
            target = square_pad_sequence(target, padding_value=-1)
        prediction = torch.argmax(output, axis=-1)
        prediction = prediction.view(-1)
        target = target.view(-1)
        mask = target != -1
        target = target[mask]
        prediction = prediction[mask]
        return prediction, target
