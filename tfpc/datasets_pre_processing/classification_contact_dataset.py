"""
This module define the class ContactDataset_per_amino_acid to predict contact
"""
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from datasets_pre_processing.generic_dataset import GenericDataset


class ContactDataset_per_amino_acid(GenericDataset):
    """
    FINE TUNNING ON CONTACT
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
        self.size_limit = limit_size_input_prot
        self.data = dataframe[col_name_input].tolist()
        self.labels = dataframe[col_name_output].tolist()
        self.coord_validity = dataframe["valid_mask"].tolist()
        self.indice_label = indice_label

    @staticmethod
    def construct_label_indice(train_label, dev_label):
        # For the compatibility with the rest of the code
        return {"1": 1, "0": 0}

    def sequence_encoding(self, seq):
        return [self.vocab[amine] for amine in seq][: self.size_limit]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mask = np.array(self.coord_validity[index][: self.size_limit]).reshape(1, -1)
        validity_mask = np.invert(np.matmul(mask.T, mask))
        yind, xind = np.indices(validity_mask.shape)
        validity_mask |= np.abs(yind - xind) < 6
        contact_map = np.less(
            squareform(pdist(self.labels[index][: self.size_limit])), 8.0
        ).astype(np.int64)
        contact_map[validity_mask] = -1
        return (
            self.data[index],
            contact_map,
        )

    def collate(self, batch):
        data = [torch.LongTensor(self.sequence_encoding(item[0])) for item in batch]
        labels = [torch.LongTensor(item[1]) for item in batch]
        # p permet d'encoder le pad token
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            labels,
        )
