"""
This module define MetricAccuracy
"""
import imp
import logging

import torch
from metrics.generic_metric import GenericMetric
import numpy as np
import pickle as pkl
import os
import pandas as pd
from collections import Counter
import math
from collections import deque


class MetricSmin(GenericMetric):
    """
    This class define the accuracy metric
    """

    def __init__(self, data_for_metric, with_rels=False):
        filename = data_for_metric["path_go_ontology"]
        class_vocab = torch.load(data_for_metric["path_class_vocab"])
        self.inverse_class_vocab = {value: key for key, value in class_vocab.items()}
        self.label_col_names = data_for_metric["label_col_names"]
        self.data_path = data_for_metric["data_path"]
        self.ont = self.load(filename, with_rels)
        self.global_class_vocab_GO_MF_2016_08 = self.get_global_vocab()
        self.inverse_global_class_vocab_GO_MF_2016_08 = {
            value: key for key, value in self.global_class_vocab_GO_MF_2016_08.items()
        }
        self.number_of_threshold = 100
        self.thresholds_values = np.linspace(0, 1, self.number_of_threshold)
        self.compute_GO_IC()
        self.order_ic = self.align_order_ic_and_global_vocab()
        self.reset_and_init_value()
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
        for one_go_term in one_prediction:
            propagated_prediction_to_ancestor += self.dico_ancestors[one_go_term]
        return list(set(propagated_prediction_to_ancestor))

    def update_metric(self, prediction, target):
        target = target[1]  # Get all target, also non leaves ones

        batch_size = len(target)
        self.number_of_prediction += batch_size
        for threshold_number in range(self.number_of_threshold):
            threshold_val = self.thresholds_values[threshold_number]
            for num_batch in range(batch_size):
                ru_at_tau, mi_at_tau = self.get_ru_and_mi(
                    prediction[num_batch], target[num_batch], threshold_val
                )
                self.ru[threshold_number] += ru_at_tau
                self.mi[threshold_number] += mi_at_tau

    def get_value(self):
        all_smin_values = self.get_all_smin()
        all_smin_values[np.isnan(all_smin_values)] = 1000.0
        argmin = np.argmin(all_smin_values)
        threshold_value = self.thresholds_values[argmin]
        logging.info(
            "The best smin value was for the threshold value of %f", threshold_value
        )
        smin = np.min(all_smin_values)
        return smin

    def reset_and_init_value(self):
        logging.debug("Reset metric")
        self.ru = np.zeros(self.number_of_threshold)
        self.mi = np.zeros(self.number_of_threshold)
        self.number_of_prediction = 0

    def get_all_smin(self):
        all_smin = []
        for threshold_number in range(self.number_of_threshold):
            mean_ru = self.ru[threshold_number] / self.number_of_prediction
            mean_mi = self.mi[threshold_number] / self.number_of_prediction
            smin_at_tau = np.sqrt(np.power(mean_ru, 2) + np.power(mean_mi, 2))
            all_smin.append(smin_at_tau)
        return np.array(all_smin)

    def get_ru_and_mi(self, proba_pred, target, threshold):
        indice_model_vocab_prediction = (
            proba_pred > threshold
        ).nonzero()  # Indice of GO term in the vocab
        indice_model_vocab_prediction = indice_model_vocab_prediction.reshape(-1)
        indice_model_vocab_prediction = indice_model_vocab_prediction.tolist()
        term_prediction = [
            self.inverse_class_vocab[p] for p in indice_model_vocab_prediction
        ]  # List of GO term ID
        term_prediction = self.append_ancestors(
            term_prediction
        )  # Append non-leaf GO term ID
        indice_global_vocab_prediction = [
            self.global_class_vocab_GO_MF_2016_08[p] for p in term_prediction
        ]
        prediction = np.zeros((len(self.global_class_vocab_GO_MF_2016_08)))
        prediction[indice_global_vocab_prediction] = 1
        prediction = prediction.astype(np.bool)
        # prediction = proba_pred > threshold
        # prediction = self.append_ancestors(prediction)
        target_one_hot = np.zeros(len(prediction))
        target_one_hot[target] = 1
        target_one_hot = target_one_hot.astype(np.bool)
        assert prediction.dtype is np.dtype(np.bool)
        assert target_one_hot.dtype is np.dtype(np.bool)
        indicatrice_ru = np.invert(prediction) * target_one_hot
        indicatrice_mi = prediction * np.invert(target_one_hot)
        # Need the vocabulary to map indice in prediction vectors and GO_term -> That will allow us to get the correspondint IC
        ru_at_tau = np.sum(self.order_ic * indicatrice_ru)
        mi_at_tau = np.sum(self.order_ic * indicatrice_mi)

        return ru_at_tau, mi_at_tau

    def get_annots(self, json_path):
        df = pd.read_json(json_path)
        annotations = df[self.label_col_names].values
        annotations = list(
            map(
                lambda list_go_term: self.go_term_with_all_ancestors(list_go_term),
                annotations,
            )
        )
        annotations = list(map(lambda x: set(x), annotations))
        return annotations

    def go_term_with_all_ancestors(self, list_go_term):
        list_go_term = list(list_go_term)
        list_go_term_completed = []
        for go_term in list_go_term:
            list_go_term_completed += self.get_anchestors(go_term)
        return list_go_term_completed

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]["is_a"]:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return list(term_set)

    def align_order_ic_and_global_vocab(self):
        size_vocab = len(self.inverse_global_class_vocab_GO_MF_2016_08)
        order_ic = []
        for id_vocab in range(size_vocab):
            go_term_associated = self.inverse_global_class_vocab_GO_MF_2016_08[id_vocab]
            if go_term_associated in self.ic.keys():
                order_ic.append(self.ic[go_term_associated])
            else:  # No example in the training/validation/testing set
                order_ic.append(0)
        return np.array(order_ic)

    # Code inspire from UDSMProt function calculate_ic from UDSMProt/code/utils/evaluate_deepgoplus.py
    def compute_GO_IC(self):
        logging.info(
            "I use data/dataset/CAFA3/ dataset to compute Information Content (IC) of the GO term"
        )
        train_annotations = self.get_annots(self.data_path + "_train.json")
        valid_annotations = self.get_annots(self.data_path + "_valid.json")
        test_annotations = self.get_annots(self.data_path + "_test.json")
        annots = train_annotations + valid_annotations + test_annotations

        cnt = Counter()
        for x in annots:
            cnt.update(x)

        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]["is_a"]:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = dict()
                    obj["is_a"] = list()
                    obj["part_of"] = list()
                    obj["regulates"] = list()
                    obj["alt_ids"] = list()
                    obj["is_obsolete"] = False
                    continue
                elif line == "[Typedef]":
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == "id":
                        obj["id"] = l[1]
                    elif l[0] == "alt_id":
                        obj["alt_ids"].append(l[1])
                    elif l[0] == "namespace":
                        obj["namespace"] = l[1]
                    elif l[0] == "is_a":
                        obj["is_a"].append(l[1].split(" ! ")[0])
                    elif with_rels and l[0] == "relationship":
                        it = l[1].split()
                        # add all types of relationships
                        obj["is_a"].append(it[1])
                    elif l[0] == "name":
                        obj["name"] = l[1]
                    elif l[0] == "is_obsolete" and l[1] == "true":
                        obj["is_obsolete"] = True
        if obj is not None:
            ont[obj["id"]] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]["alt_ids"]:
                ont[t_id] = ont[term_id]
            if ont[term_id]["is_obsolete"]:
                del ont[term_id]
        for term_id, val in ont.items():
            if "children" not in val:
                val["children"] = set()
            for p_id in val["is_a"]:
                if p_id in ont:
                    if "children" not in ont[p_id]:
                        ont[p_id]["children"] = set()
                    ont[p_id]["children"].add(term_id)
        return ont
