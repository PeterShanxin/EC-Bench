"""
This module define loss logger
"""
import numpy as np
import logging


class LossLogger:
    """
    This class keep track of the loss during the training process
    We have two function in order to add train loss, the first one is add_train_loss_tmp() that is executed at each epoch and
    calc_mean_train_loss() that avreage the tmp loss and is executed at each optimizer step/each theorical batch size.

    For the tmp loss, we log in a regular basis and we add_dev_loss() at each batch in the develpment set and
    we execute finish_add_dev_loss() to avreage the loss and give only one result for a specific batch seen number of train data.
    """

    def __init__(self):
        self.tmp_train_loss = []
        self.tmp_dev_loss = []
        self.tmp_test_loss = []
        self.train_loss = []
        self.dev_loss = []
        self.test_loss = None

    def add_loss_tmp(self, typeLoss, loss):
        if typeLoss == "train":
            self.tmp_train_loss.append(loss)
        elif typeLoss == "dev":
            logging.debug("Loss en dev : %s", loss)
            self.tmp_dev_loss.append(loss)
        elif typeLoss == "test":
            self.tmp_test_loss.append(loss)
        else:
            raise RuntimeError(
                "Loss type unknown in add_loss_tmp in the LossLogger class"
            )

    def mean_train_loss(self, nb_batch_seen):
        train_loss = np.mean(self.tmp_train_loss)
        self.tmp_train_loss = []
        self.train_loss.append([nb_batch_seen, train_loss])

    def finish_add_loss_tmp(self, typeLoss, nb_batch_seen):
        if typeLoss == "dev":
            self.dev_loss.append([nb_batch_seen, np.mean(self.tmp_dev_loss)])
            self.tmp_dev_loss = []
        elif typeLoss == "test":
            self.test_loss = np.mean(self.tmp_test_loss)
            self.tmp_test_loss = []
        else:
            raise RuntimeError(
                "Loss type unknown in finish_add_loss_tmp in the LossLogger class"
            )

    def get_loss(self, typeLoss):
        if typeLoss == "train":
            return self.train_loss
        elif typeLoss == "dev":
            return self.dev_loss
        elif typeLoss == "test":
            return self.test_loss
