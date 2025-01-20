"""
This module define a specific dataset
"""
import logging
import torch
from datasets_pre_processing.generic_dataset import GenericDataset


class ClassificationPerAminoAcidDataset(GenericDataset):
    """
    This class is a dataset where we have one label per protein/example
    """

    def __init__(
        self,
        dataframe,
        limit_size_input_prot,
        config,
        col_name_input,
        col_name_output,
        indice_label,
    ):
        super().__init__(
            dataframe, limit_size_input_prot, config, col_name_input, col_name_output
        )
        self.indice_label = indice_label

    @staticmethod
    def construct_label_indice(train_label, dev_label):
        all_unique_label = dict()
        compteur = 0
        for lab_list in train_label:
            for lab in lab_list:
                if lab not in all_unique_label.keys():
                    all_unique_label[lab] = compteur
                    compteur += 1
        nb_class_not_in_train = 0
        for lab_list in dev_label:
            for lab in lab_list:
                if lab not in all_unique_label.keys():
                    all_unique_label[lab] = compteur
                    nb_class_not_in_train += 1
                    compteur += 1

        return all_unique_label

    def sequence_encoding(self, seq):
        return [self.vocab[amine] for amine in seq][: self.size_limit]

    def label_encoding(self, label):
        return [self.indice_label[l] for l in label]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def collate(self, batch):
        data = [torch.LongTensor(self.sequence_encoding(item[0])) for item in batch]
        labels = [torch.LongTensor(self.label_encoding(item[1])) for item in batch]
        # p permet d'encoder le pad token
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1),
        )
