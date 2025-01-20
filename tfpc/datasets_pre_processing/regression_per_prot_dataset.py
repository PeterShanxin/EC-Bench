import torch
import numpy as np
from datasets_pre_processing.generic_dataset import GenericDataset


# FINE TUNNING ON Regression
class RegressionPerProtDataset(GenericDataset):
    def __init__(
        self, dataframe, config, limit_size_input_prot, col_name_input, col_name_output
    ):
        super().__init__(
            dataframe, limit_size_input_prot, config, col_name_input, col_name_output
        )
        self.limit_size_input_prot = limit_size_input_prot
        self.data = dataframe[col_name_input].tolist()
        self.labels = dataframe[col_name_output].tolist()

    def sequence_encoding(self, seq):
        return [self.vocab["c"]] + [self.vocab[amine] for amine in seq][
            : self.limit_size_input_prot
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def collate(self, batch):
        data = [torch.LongTensor(self.sequence_encoding(item[0])) for item in batch]
        labels = [item[1][0] for item in batch]
        # p permet d'encoder le pad token
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            torch.FloatTensor(labels),
        )
