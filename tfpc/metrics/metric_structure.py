"""
This module define MetricStructure
"""
import logging
import scipy
from scipy import stats
import numpy as np
from metrics.generic_metric import GenericMetric


class MetricStructure(GenericMetric):
    """
    This class define the MetricStructure
    """

    def __init__(self):
        self.reset_and_init_value()

    def update_metric(self, prediction, target):
        mse_list = []
        true_target, mask = target
        batch_size = prediction.shape[0]
        for num_batch in range(batch_size):
            mask_one_prot = mask[num_batch]
            output_one_prot = prediction[num_batch]
            target_one_prot = true_target[num_batch]
            # We take only AA with correct target that is defined by the mask
            output_one_prot = output_one_prot[mask_one_prot]
            mse = ((output_one_prot - target_one_prot) ** 2).mean()
            mse_list.append(mse.item())

        self.all_preds += mse_list

    def get_value(self):
        mse = np.mean(self.all_preds)
        return mse

    def reset_and_init_value(self):
        logging.debug("Reset metric")
        self.all_preds = []