"""
This module define the training dataset manager
"""
import logging
from collections import Counter
import pandas as pd
import torch
from dataset_manager.data_factory import DataFactory
from datasets_pre_processing.classification_contact_dataset import (
    ContactDataset_per_amino_acid,
)
from utils.json_loader import load_json_into_pandas_dataframe
from goatools.obo_parser import GODag
from tqdm import tqdm


class TrainingDatasetManager:
    """
    Class to create(from a dataset) and manage dataloader
    """

    def __init__(self, config, no_train=False, dico_loaded_vocab=None):
        self.no_train = no_train
        self.dataloader_train = []
        self.dataloader_dev = []
        self.dataloader_test = []
        self.tasks_types = []
        self.task_vocabs = []
        for task_param in config.tasks:
            # Loading of the dataset
            path_json_dataset = (
                config.root_datasets + task_param.data_path + "/" + task_param.data_path.split("/")[-1]
            )
            if not no_train:
                path_train = path_json_dataset + "_train.json"
                logging.debug("J'essaye de read le json suivant : %s", (path_train))
                train = load_json_into_pandas_dataframe(path_train)
            path_valid = path_json_dataset + "_valid.json"
            logging.debug("J'essaye de read le json suivant : %s", (path_valid))
            dev = load_json_into_pandas_dataframe(path_valid)
            path_test = path_json_dataset + "_test.json"
            logging.debug("J'essaye de read le json suivant : %s", (path_test))
            test = load_json_into_pandas_dataframe(path_test)

            data_factory = DataFactory(config, task_param)
            if no_train and "classification" in data_factory.task_type:
                try:
                    data_factory.set_label_indice_if_classif_task(
                        dico_loaded_vocab[task_param.unique_task_name]
                    )
                except:
                    logging.debug(
                        "Attention fichier vocab non trouvé donc on charge à l'ancienne."
                    )
                    path_train = path_json_dataset + "_train.json"
                    logging.debug("J'essaye de read le json suivant : %s", (path_train))
                    train = load_json_into_pandas_dataframe(path_train)
                    data_factory.create_label_indice_if_classif_task(train, dev)

            if not no_train:
                if "classification" in data_factory.task_type:
                    data_factory.create_label_indice_if_classif_task(train, dev)
                dataset_train = data_factory.create_dataset_from(train)
            dataset_dev = data_factory.create_dataset_from(dev)
            dataset_test = data_factory.create_dataset_from(test)

            if not no_train:
                self.dataloader_train.append(
                    data_factory.create_dataloader_from(
                        dataset_train,
                        task_param.batch_size,
                        task_param.balanced_dataset,
                    )
                )
            self.dataloader_dev.append(
                data_factory.create_dataloader_from(
                    dataset_dev,
                    task_param.batch_size,
                    False,
                )
            )

            self.dataloader_test.append(
                data_factory.create_dataloader_from(
                    dataset_test,
                    task_param.batch_size,
                    False,
                )
            )

            self.tasks_types.append(data_factory.task_type)
            if "classification" in data_factory.task_type:
                self.task_vocabs.append(data_factory.indice_label)
            else:
                self.task_vocabs.append(None)

        if hasattr(config, "auxiliary_evaluation"):
            params = config.auxiliary_evaluation
            path_json_dataset = (
                config.root_datasets + params.data_path + "/" + params.data_path
            )
            path_dev = path_json_dataset + "_train.json"
            dev = load_json_into_pandas_dataframe(path_dev)

            dataset = ContactDataset_per_amino_acid(
                dev,
                params.limit_size_input_prot,
                config,
                params.col_name_input,
                params.col_name_output,
                {"1": 1, "0": 0},
            )
            self.auxiliary_dataloader_dev = torch.utils.data.DataLoader(
                dataset,
                batch_size=params.batch_size,
                shuffle=True,
                collate_fn=dataset.collate,
            )

    def get_nb_labels(self, task_index):
        train_dataloader = self.dataloader_train[task_index]
        dev_dataloader = self.dataloader_dev[task_index]

        all_labels = self.get_target_from_dataloader(
            train_dataloader, task_index
        ).copy()

        all_labels += self.get_target_from_dataloader(dev_dataloader, task_index).copy()

        logging.debug("All label are %s", list(set(all_labels)))
        nb_uniq_label = max(len(list(set(all_labels))), max(all_labels) + 1)
        logging.debug(
            "Il y a %s labels dans le dataset pour la tâche numero %s",
            nb_uniq_label,
            task_index,
        )
        return nb_uniq_label

    def get_target_from_dataloader(self, dataloader, task_index):
        task_type = self.tasks_types[task_index]
        if task_type == "classification_per_prot":
            all_labels = []
            for _, target_batch in dataloader:
                for t in target_batch:
                    label = t.item()
                    all_labels.append(label)
        elif task_type == "classification_per_prot_instance_weighting":
            all_labels = []
            for _, target_batch in dataloader:
                for t in target_batch:
                    label = t[0].item()
                    all_labels.append(label)
        elif (
            task_type == "classification_per_amino_acid"
            or task_type == "classification_contact_per_amino_acid"
        ):
            all_labels = []
            for _, target_batch in dataloader:
                for seq_lab in target_batch:
                    for t in seq_lab:
                        label = t.item()
                        all_labels.append(label)
        else:
            logging.debug("The task type is %s", task_type)
            raise RuntimeError("Not possible to get all label from this task type")
        return list(set(all_labels))

    def get_weight(self, task_index):
        logging.info(
            "Be Carefull loss not comparable between test and train in weighting because weight are not calculated in the same way. Because we don't reload the training dataset and so we don't have the frequency of class."
        )
        if self.no_train:
            nb_class = int(input("How many classes ?"))
            logging.debug("Create uniform weight of 1")
            tenseur = torch.ones((nb_class))
        else:
            train_dataloader = self.dataloader_train[task_index]
            all_labels = self.get_target_from_dataloader(train_dataloader, task_index)
            nb_uniq_label = len(list(set(all_labels)))

            compteur = Counter(all_labels)
            tenseur = torch.tensor([compteur[k] for k in range(nb_uniq_label)])
            tenseur = tenseur.float()
            tenseur = 1 / tenseur

        return tenseur

    def get_GO_weights(self, task_index):
        if self.no_train:
            nb_class = int(input("How many classes ?"))
            logging.debug("Create uniform weight of 1")
            tenseur = torch.ones((nb_class))
        else:
            vocab = self.task_vocabs[task_index]
            inverse_vocab = {value: key for key, value in vocab.items()}

            dico_is_leaf = self.get_all_go_leaf()
            tenseur = []
            for k in range(len(inverse_vocab)):
                if dico_is_leaf[inverse_vocab[k]]:
                    weight = 1
                else:
                    weight = 0.1
                tenseur.append(weight)
            tenseur = torch.tensor(tenseur)
        return tenseur

    def get_all_go_leaf(self):
        logging.info("Creating class weights for the loss")
        godag = GODag("data/GO_embedings/go_2016_08.obo")
        dico_is_leaf = dict()
        for go_term in tqdm(godag.values(), total=44949):
            is_leaf = not go_term.children
            dico_is_leaf[go_term.id] = is_leaf
        return dico_is_leaf

    def get_vocab(self, ind_task):
        return self.task_vocabs[ind_task]

    def get_pred_type(self, ind_task):
        return self.tasks_types[ind_task]

    def get_nb_batch(self, task_index):
        return len(self.dataloader_train[task_index])

    def get_train_dataloader(self, index):
        return self.dataloader_train[index]

    def get_dev_dataloader(self, index):
        return self.dataloader_dev[index]

    def get_test_dataloader(self, index):
        return self.dataloader_test[index]
