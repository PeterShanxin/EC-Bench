"""
This module define MetricAccuracy
"""
import logging
from metrics.generic_metric import GenericMetric


class MetricAccuracy(GenericMetric):
    """
    This class define the accuracy metric
    """

    def __init__(self):
        self.reset_and_init_value()

    def update_metric(self, prediction, target):
        self.total_correct += (prediction == target)[target != -1].sum().item()
        self.total += sum(sum([target != -1])).item()

    def get_value(self):
        if self.total == 0:
            return 0
        else:
            acc = 100 * self.total_correct / self.total
        return acc

    def reset_and_init_value(self):
        logging.debug("Reset metric")
        self.total_correct = 0
        self.total = 0
