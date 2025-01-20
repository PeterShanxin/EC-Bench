"""
This module define MetricAccuracy
"""
import logging
from metrics.generic_metric import GenericMetric
import numpy as np
import pickle as pkl
import torch


class MetricFMax(GenericMetric):
    """
    This class define the accuracy metric
    """

    def __init__(self, data_for_metric):
        self.number_of_threshold = 100
        self.thresholds_values = np.linspace(0, 1, self.number_of_threshold)
        self.reset_and_init_value()
        class_vocab = torch.load(data_for_metric["path_class_vocab"])
        self.inverse_class_vocab = {value: key for key, value in class_vocab.items()}
        self.global_class_vocab_GO_MF_2016_08 = self.get_global_vocab()
        self.inverse_global_class_vocab_GO_MF_2016_08 = {
            value: key for key, value in self.global_class_vocab_GO_MF_2016_08.items()
        }
        self.path_dico_leaves_childs = "data/GO_embedings/dico_childs_which_are_leaves.pkl"  # dico_leaves_childs.pkl"
        fichier = open(self.path_dico_leaves_childs, "rb")
        self.dico_leaves_childs = pkl.load(fichier)
        self.dico_ancestors = pkl.load(
            open("data/GO_embedings/dico_ancestor_2016_08_MF.pkl", "rb")
        )

    def get_global_vocab(self):
        global_list_GO_term_MF_2016_08 = pkl.load(
            open("data/GO_embedings/global_list_GO_term_MF_2016_08.pkl", "rb")
        )
        global_class_vocab_GO_MF_2016_08 = {
            GO_term: num for num, GO_term in enumerate(global_list_GO_term_MF_2016_08)
        }
        return global_class_vocab_GO_MF_2016_08

    def append_ancestors(self, one_prediction):
        propagated_prediction_to_ancestor = []
        for one_leaf in one_prediction:
            propagated_prediction_to_ancestor += self.dico_ancestors[one_leaf]
        return list(set(propagated_prediction_to_ancestor))

    def update_metric(self, prediction, target):
        target = target[1]  # Get all target, also non leaves ones
        batch_size = len(target)
        for threshold_number in range(self.number_of_threshold):
            threshold_val = self.thresholds_values[threshold_number]
            for num_batch in range(batch_size):
                TP, FP, FN = self.get_confusion_matrix(
                    prediction[num_batch], target[num_batch], threshold_val
                )
                self.TP[threshold_number] += TP
                self.FP[threshold_number] += FP
                self.FN[threshold_number] += FN

    def get_value(self):
        all_f1_values = self.get_all_f1()
        # We replace nan values by zero, that can be the case when the model make no prediction, there are 0 positive label or precision_at_t + recall_at_t=0
        all_f1_values[np.isnan(all_f1_values)] = 0.0
        argmax = np.argmax(all_f1_values)
        threshold_value = self.thresholds_values[argmax]
        logging.debug("all_f1_values: %s", str(all_f1_values))
        logging.info(
            "The best f1 value was for the threshold value of %f", threshold_value
        )
        fmax = np.max(all_f1_values)
        return fmax

    def reset_and_init_value(self):
        logging.debug("Reset metric")
        self.TP = np.zeros(self.number_of_threshold)
        self.FP = np.zeros(self.number_of_threshold)
        self.FN = np.zeros(self.number_of_threshold)

    def get_all_f1(self):
        all_f1 = []
        for threshold_number in range(self.number_of_threshold):
            precision_at_t = self.get_precision(
                self.TP[threshold_number], self.FP[threshold_number]
            )
            recall_at_t = self.get_recall(
                self.TP[threshold_number], self.FN[threshold_number]
            )

            f1_at_t = (2 * precision_at_t * recall_at_t) / (
                precision_at_t + recall_at_t
            )
            all_f1.append(f1_at_t)
        return np.array(all_f1)

    def get_precision(self, TP, FP):
        number_of_prediction = TP + FP
        return TP / number_of_prediction

    def get_recall(self, TP, FN):
        number_of_label_positive = TP + FN
        return TP / number_of_label_positive

    def get_confusion_matrix(self, proba_pred, target, threshold):
        target = target.tolist()
        target = [self.inverse_global_class_vocab_GO_MF_2016_08[t] for t in target]
        prediction = (
            proba_pred > threshold
        ).nonzero()  # Indice of GO term in the vocab
        prediction = prediction.reshape(-1)
        prediction = prediction.tolist()
        prediction = [
            self.inverse_class_vocab[p] for p in prediction
        ]  # List of GO term ID
        prediction = self.append_ancestors(prediction)  # Append non-leaf GO term ID
        # print("Prediction:", prediction)
        # print("Target:", target)
        TP = len(set(prediction).intersection(set(target)))
        FP = len(prediction) - TP
        FN = len(set(target) - set(prediction))
        return TP, FP, FN
