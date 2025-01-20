"""
This module define the class StructurePredictionDataset
"""
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from datasets_pre_processing.generic_dataset import GenericDataset


class StructurePredictionDataset(GenericDataset):
    """
    FINE TUNNING ON Structure
    """

    def __init__(
        self,
        dataframe,
        config,
        limit_size_input_prot,
        col_name_input,
        col_name_output,
    ):
        super().__init__(
            dataframe, limit_size_input_prot, config, col_name_input, col_name_output
        )
        self.size_limit = limit_size_input_prot
        self.data = dataframe[col_name_input].tolist()
        self.labels = dataframe[col_name_output].tolist()
        self.coord_validity = dataframe["valid_mask"].tolist()

    def sequence_encoding(self, seq):
        return [self.vocab[amine] for amine in seq][: self.size_limit]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mask = torch.tensor(self.coord_validity[index][: self.size_limit])
        lab = torch.tensor(self.labels[index][: self.size_limit])
        lab = lab[mask]
        return self.data[index], lab, mask

    def collate(self, batch):
        data = [torch.LongTensor(self.sequence_encoding(item[0])) for item in batch]
        labels = [item[1] for item in batch]
        all_masks = torch.nn.utils.rnn.pad_sequence(
            [item[2] for item in batch], batch_first=True, padding_value=False
        )
        # p permet d'encoder le pad token
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            [labels, all_masks],
        )
