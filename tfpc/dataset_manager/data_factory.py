"""
This module define the data factory
"""
import logging
import torch
import pandas as pd
import collections
from datasets_pre_processing.classification_per_prot_dataset import (
    ClassificationPerProtDataset,
)
from datasets_pre_processing.classification_per_prot_GO_term import (
    ClassificationPerProtGOTerm,
)
from datasets_pre_processing.classification_per_prot_with_apriori_dataset import (
    ClassificationPerProtWithAprioriDataset,
)
from datasets_pre_processing.classification_per_prot_weighted_dataset import (
    ClassificationPerProtWeightedDataset,
)
from datasets_pre_processing.language_modeling_dataset import LanguageModelingDataset
from datasets_pre_processing.classification_per_amino_acid_dataset import (
    ClassificationPerAminoAcidDataset,
)
from datasets_pre_processing.classification_contact_dataset import (
    ContactDataset_per_amino_acid,
)
from datasets_pre_processing.regression_per_prot_dataset import RegressionPerProtDataset
from datasets_pre_processing.structure_prediction_dataset import (
    StructurePredictionDataset,
)
from datasets_pre_processing.language_modeling_dataset_weighted_mask import (
    LanguageModelingDatasetWeightMask,
)
from datasets_pre_processing.language_modeling_dataset_inverse_weighted_mask import (
    LanguageModelingDatasetInverseWeightMask,
)


class DataFactory:
    """
    This class is a datafactory
    """

    def __init__(self, config, task_param):
        self.config = config
        self.unique_task_name = task_param.unique_task_name
        self.dataset_type_name = task_param.dataset_type["name"]
        self.dataset_type_params = task_param.dataset_type["params"]
        self.col_name_input = task_param.col_name_input
        if hasattr(task_param, "col_name_output"):
            self.col_name_output = task_param.col_name_output
        else:
            self.col_name_output = False

        self.limit_size_input_prot = task_param.limit_size_input_prot

        if self.dataset_type_name == "Classification_per_prot_dataset":
            self.task_type = "classification_per_prot"
            self.specific_dataset_class = ClassificationPerProtDataset
        elif self.dataset_type_name == "Classification_per_prot__with_apriori_dataset":
            self.task_type = "classification_per_prot"
            self.specific_dataset_class = ClassificationPerProtWithAprioriDataset
        elif self.dataset_type_name == "Classification_per_prot_weighted_dataset":
            self.task_type = "classification_per_prot_instance_weighting"
            self.specific_dataset_class = ClassificationPerProtWeightedDataset
        elif self.dataset_type_name == "Language_modeling_dataset":
            self.task_type = "language_modeling"
            self.specific_dataset_class = LanguageModelingDataset
        elif self.dataset_type_name == "Language_modeling_dataset_weight_mask":
            self.task_type = "language_modeling"
            self.specific_dataset_class = LanguageModelingDatasetWeightMask
        elif self.dataset_type_name == "Language_modeling_dataset_inverse_weight_mask":
            self.task_type = "language_modeling"
            self.specific_dataset_class = LanguageModelingDatasetInverseWeightMask
        elif self.dataset_type_name == "Classification_per_amino_acid_dataset":
            self.task_type = "classification_per_amino_acid"
            self.specific_dataset_class = ClassificationPerAminoAcidDataset
        elif self.dataset_type_name == "classification_contact_per_amino_acid":
            self.task_type = "classification_contact_per_amino_acid"
            self.specific_dataset_class = ContactDataset_per_amino_acid
        elif self.dataset_type_name == "regression_per_prot":
            self.task_type = "regression"
            self.specific_dataset_class = RegressionPerProtDataset
        elif self.dataset_type_name == "structure_prediction":
            self.task_type = "structure"
            self.specific_dataset_class = StructurePredictionDataset
        elif self.dataset_type_name == "Classif_GO_term_poincare_ball":
            self.task_type = "classification_per_prot"
            self.specific_dataset_class = ClassificationPerProtGOTerm
        else:
            raise RuntimeError("unknown dataset_type_name")

        if "classification" in self.task_type:
            self.indice_label = None

        if self.task_type == "language_modeling":
            self.prop_mask = self.dataset_type_params[0]
            if config.starting_model.type == "from_scratch":
                self.vocab = None
            else:
                self.vocab = config.vocab

    def create_dataset_from(
        self,
        dataframe: pd.DataFrame,
    ) -> torch.utils.data.Dataset:
        if self.task_type == "language_modeling":
            dataset = (
                self.specific_dataset_class(  # pylint: disable=too-many-function-args
                    dataframe,
                    self.limit_size_input_prot,
                    self.config,
                    self.col_name_input,
                    self.col_name_output,
                    self.prop_mask,
                    self.vocab,
                )
            )
            self.vocab = dataset.vocab
            self.config.vocab = dataset.vocab
        elif "classification" in self.task_type:
            logging.debug(
                "J'essaye de créer un dataset de type : %s",
                (self.specific_dataset_class),
            )
            dataset = self.specific_dataset_class(
                dataframe,
                self.limit_size_input_prot,
                self.config,
                self.col_name_input,
                self.col_name_output,
                self.indice_label,
            )
            logging.debug(
                "Liste des argument disponible de dataset : %s",
                (dataset.__dict__.keys()),
            )
            self.indice_label = dataset.indice_label
            self.save_label_vocab()
        elif self.task_type == "regression" or "structure":
            dataset = (
                self.specific_dataset_class(  # pylint: disable=no-value-for-parameter
                    dataframe,
                    self.config,
                    self.limit_size_input_prot,
                    self.col_name_input,
                    self.col_name_output,
                )
            )
        return dataset

    def save_label_vocab(self):
        # We save the vocabulary
        vocab_saving_path = (
            self.config.path_folder_json + self.unique_task_name + "_vocab.pth"
        )
        torch.save(self.indice_label, vocab_saving_path)

    def create_label_indice_if_classif_task(self, data_train, data_dev):
        if "classification" in self.task_type:
            self.indice_label = self.specific_dataset_class.construct_label_indice(
                data_train[self.col_name_output], data_dev[self.col_name_output]
            )

    def set_label_indice_if_classif_task(self, vocab):
        if "classification" in self.task_type:
            self.indice_label = vocab

    def create_dataloader_from(self, dataset, batch_size, balanced_dataset):
        if balanced_dataset:
            dico_occ_each_class = self.get_dict_nb_label_per_class(dataset)
            sample_weights = self.get_balanced_weights(dataset, dico_occ_each_class)
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=dataset.collate,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=dataset.collate,
            )
        return dataloader

    def get_dict_nb_label_per_class(self, dataset):
        all_labels = [label for _, label in dataset]
        count = collections.Counter(all_labels)
        dico_count = dict(count)
        for key in dico_count.keys():
            dico_count[key] = 1 / dico_count[key]
        return dico_count

    def get_balanced_weights(self, dataset, dico_occ_each_class):
        sample_weights = [0] * len(dataset)
        for idx, (_, label) in enumerate(dataset):
            class_weight = dico_occ_each_class[label]
            sample_weights[idx] = class_weight
        return sample_weights