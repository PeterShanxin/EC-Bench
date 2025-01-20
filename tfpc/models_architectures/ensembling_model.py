"""
This module define the ensembling model class
"""
import torch
import torch.nn as nn
import logging


class EnsemblingModel(nn.Module):
    """
    This class allow you to make ensembling model
    """

    def __init__(self, list_of_model):
        super(EnsemblingModel, self).__init__()
        self.list_of_model = list_of_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, src):
        pred = self.softmax(self.list_of_model[0](src))
        if pred.shape[1] == 65:
            pred_ext = torch.cuda.FloatTensor(pred.shape[0], 99986).fill_(0)
            pred_ext[:, :65] = pred
            pred = pred_ext
        all_preds = pred
        logging.debug("Shape output one model is %s", all_preds.shape)
        nb_pred = 1
        for indice_model in range(1, len(self.list_of_model)):
            pred = self.softmax(self.list_of_model[indice_model](src))
            if pred.shape[1] == 65:
                pred_ext = torch.cuda.FloatTensor(pred.shape[0], 99986).fill_(0)
                pred_ext[:, :65] = pred
                pred = pred_ext
            all_preds += pred
            nb_pred += 1
        mean_pred = all_preds / nb_pred
        logging.debug("Shape output merge model is %s", mean_pred.shape)
        return mean_pred
