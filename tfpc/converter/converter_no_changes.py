"""
This module define a converter
"""
import torch
from converter.generic_converter import GenericConverter
from utils.utils import select_output_lm


class ConverterNoChanges(GenericConverter):
    """
    This class doesn't convert
    """

    def convert_output_and_target(self, output, target):
        return output, target
