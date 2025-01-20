"""
This module define GenericMetric
"""
from abc import abstractmethod


class GenericMetric:
    """
    This class define an interface for the metric class
    """

    @abstractmethod
    def update_metric(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_value(self):
        raise NotImplementedError

    @abstractmethod
    def reset_and_init_value(self):
        raise NotImplementedError
