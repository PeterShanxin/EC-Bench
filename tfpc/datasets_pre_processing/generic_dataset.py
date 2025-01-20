"""
This module define an interface for the differents datasets
"""
import logging
import torch


class GenericDataset(torch.utils.data.Dataset):
    """
    This class is an interface for the datasets
    """

    def __init__(
        self,
        dataframe,
        limit_size_input_prot,
        config,
        col_name_input,
        col_name_output,
    ):
        logging.debug("J'éxécute le constructeur de Generic_dataset")
        self.size_limit = limit_size_input_prot
        self.col_name_input = col_name_input
        self.col_name_output = col_name_output
        self.vocab = config.vocab
        self.data = dataframe[col_name_input].tolist()
        logging.debug("All cols names are %s", dataframe.columns)
        if col_name_output is not False:
            self.labels = dataframe[col_name_output].tolist()

    def sequence_encoding(self, seq):
        return [self.vocab[amine] for amine in seq][: self.size_limit]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
