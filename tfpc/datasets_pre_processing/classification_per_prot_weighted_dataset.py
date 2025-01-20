"""
This module define a specific dataset
"""
import logging
import torch
from datasets_pre_processing.generic_dataset import GenericDataset


class ClassificationPerProtWeightedDataset(GenericDataset):
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
        if "weight" in dataframe.columns:
            self.weight = list(dataframe["weight"])
        else:
            self.weight = [1 for _ in range(len(dataframe))]

    @staticmethod
    def construct_label_indice(train_label, dev_label):
        all_unique_label = dict()
        compteur = 0
        for lab in train_label:
            if lab not in all_unique_label.keys():
                all_unique_label[lab] = compteur
                compteur += 1
        nb_class_not_in_train = 0
        for lab in dev_label:
            if lab not in all_unique_label.keys():
                all_unique_label[lab] = compteur
                nb_class_not_in_train += 1
                compteur += 1

        logging.info(
            "Il y a %d classes qui ne sont pas représenté dans l'ensemble de train mais sont \
                présente dans l'ensemble de validation.",
            (nb_class_not_in_train),
        )
        logging.info(
            "La taille de l'ensemble de validation est de %d", (len(dev_label))
        )
        logging.info(
            "Il y a %s labels différents dans ce dataset", len(all_unique_label)
        )
        return all_unique_label

    def sequence_encoding(self, seq):
        return [self.vocab["c"]] + [self.vocab[amine] for amine in seq][
            : self.size_limit
        ]

    def label_encoding(self, label):
        return self.indice_label[label]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.weight[index]

    def collate(self, batch):
        data = [torch.LongTensor(self.sequence_encoding(item[0])) for item in batch]
        labels = [self.label_encoding(item[1]) for item in batch]
        weights = [item[2] for item in batch]
        # p permet d'encoder le pad token
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            [torch.LongTensor(labels), torch.FloatTensor(weights)],
        )
