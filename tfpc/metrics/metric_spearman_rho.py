"""
This module define MetricSpearmanRho
"""
import logging
import scipy
from scipy import stats
import numpy as np
from metrics.generic_metric import GenericMetric


class MetricSpearmanRho(GenericMetric):
    """
    This class define the Spearman's rho metric
    """

    def __init__(self):
        self.reset_and_init_value()

    def update_metric(self, prediction, target):
        prediction = prediction.detach().cpu().numpy().tolist()
        target = target.detach().cpu().numpy().tolist()

        self.all_preds += prediction
        self.all_true_value += target

    def get_value(self):
        spearman_corr = scipy.stats.spearmanr(self.all_preds, self.all_true_value)[0]
        return 100 * np.abs(spearman_corr)

    def reset_and_init_value(self):
        logging.debug("Reset metric")
        self.all_preds = []
        self.all_true_value = []
