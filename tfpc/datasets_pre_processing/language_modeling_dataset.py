"""
This module the language modeling dataset
"""
import logging
import torch
import numpy as np
from datasets_pre_processing.generic_dataset import GenericDataset


class LanguageModelingDataset(GenericDataset):
    """
    class that define language modeling dataset
    """

    def __init__(
        self,
        dataframe,
        limit_size_input_prot,
        config,
        col_name_input,
        col_name_output,
        prop_mask,
        vocab,
    ):
        super().__init__(
            dataframe, limit_size_input_prot, config, col_name_input, col_name_output
        )
        self.prop_mask = prop_mask
        self.limit_size_input_prot = limit_size_input_prot
        self.data = list(dataframe[col_name_input])
        self.list_possible_random = [
            "I",
            "W",
            "Y",
            "F",
            "D",
            "E",
            "R",
            "K",
            "H",
            "Q",
            "G",
            "A",
            "S",
            "T",
            "N",
            "P",
            "C",
            "V",
            "M",
            "L",
            "F",
            "O",
            "B",
            "U",
            "Z",
            "X",
            "m",
            "p",
            "c",
        ]

        nb_seq = len(self.data)
        logging.info("Il y a %d séquences/domaines dans le dataset.", (nb_seq))
        if vocab is None:
            self.vocab = self.construct_vocab()
        else:
            self.vocab = vocab

        logging.debug("The vocab is %s", self.vocab)
        self.list_possible_random = self.list_possible_random[:-3]
        self.list_nb_random = np.array(
            [self.vocab[c] for c in self.list_possible_random]
        )

    def construct_vocab(self):
        dict_charac = {y: x for x, y in enumerate(self.list_possible_random)}
        logging.debug("There is %s letter in dict charac", len(dict_charac))
        return dict_charac

    def corrupt_input_optim(self, sequence):
        # pylint: disable=not-callable
        taille_seq = sequence.shape[0]
        if taille_seq > self.limit_size_input_prot:
            max_ind = taille_seq - self.limit_size_input_prot
            ind_debut = torch.randint(max_ind, (1,))[0]
            sequence = sequence[ind_debut : ind_debut + self.limit_size_input_prot]
            taille_seq = self.limit_size_input_prot
        nb_mask = int(self.prop_mask * taille_seq)

        nb_keep = int(nb_mask * 0.1)
        nb_random = int(nb_mask * 0.1)
        nb_true_m = nb_mask - nb_keep - nb_random

        perm = torch.randperm(taille_seq)
        indices_mask = perm[:nb_mask]
        target_final = sequence[indices_mask].clone()
        sequence[indices_mask[:nb_true_m]] = self.vocab["m"]
        choice = torch.tensor(np.random.choice(self.list_nb_random, nb_random))
        sequence[indices_mask[nb_true_m : nb_random + nb_true_m]] = choice

        return sequence, target_final, indices_mask

    def sequence_encoding(self, seq):
        # pylint: disable=not-callable
        return torch.tensor([self.vocab[amine] for amine in seq])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, batch):
        # m est le token pour le mask
        # p est le token pour le PAD
        data = []
        targets = None
        all_indices_mask = []
        for sequence in batch:
            corrupt_seq, target, indice_mask = self.corrupt_input_optim(
                self.sequence_encoding(sequence)
            )
            if targets is None:
                targets = target
            else:
                targets = torch.cat((targets, target))
            data.append(corrupt_seq)
            all_indices_mask.append(indice_mask)
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            [targets, all_indices_mask],
        )
