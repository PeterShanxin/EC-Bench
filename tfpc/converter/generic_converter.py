"""
This module implement the generic class for the converter
"""
from abc import abstractmethod


class GenericConverter:
    """
    This class define what is a converter
    """

    @abstractmethod
    def convert_output_and_target(
        self,
        output,
        target,
    ):
        raise NotImplementedError