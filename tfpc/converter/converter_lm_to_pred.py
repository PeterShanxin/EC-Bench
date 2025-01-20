"""
This module define a converter
"""
import torch
from converter.generic_converter import GenericConverter
from utils.utils import select_output_lm


class ConverterLMToPred(GenericConverter):
    """
    This class convert language model output to prediction
    """

    def convert_output_and_target(self, output, target):
        true_target, indice_mask = target
        prediction = select_output_lm(output, indice_mask)
        prediction = torch.argmax(prediction, axis=1)
        return prediction, true_target
