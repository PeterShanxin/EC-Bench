"""
This module define how to manage configuration.
Classes :
ConfigManager
"""

from collections import namedtuple
import json
import torch
from utils.utils import function_not_implemented


class ConfigManager:
    """
    A class to manage the config
    """

    def __init__(self, json_path: str):
        # Fixed attribute of config
        self.root_models = "data/models/"
        self.root_datasets = "data/datasets/"
        self.json_path = json_path
        self.path_folder_json = "/".join(json_path.split("/")[:-1]) + "/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # To avoid pylint error
        namedTupleTmp = namedtuple("starting_model", "type path_vocab")
        self.starting_model = namedTupleTmp(None, None)

        # Load other attribute from config file
        fichier = open(json_path)
        dict_config = json.load(fichier)

        for key, value in dict_config.items():
            setattr(self, key, value)

        for task in self.tasks:
            # The default value for all task of this attribute is False, to know if we sample equally for each class (OverSampling for minority class)
            if "balanced_dataset" not in task.keys():
                task["balanced_dataset"] = False

        # Convert dict attribute to nametupple
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                new = self.convert_dict_to_nametupple(key, value)
                setattr(self, key, new)
            # Also convert list of dict to list of nametupple
            if isinstance(value, list) and isinstance(value[0], dict):
                new_list = []
                for element in value:
                    new_list.append(self.convert_dict_to_nametupple(key, element))
                setattr(self, key, new_list)

        # Load vocab
        if self.starting_model.type == "from_pre_trained":
            self.vocab = torch.load(self.root_models + self.starting_model.path_vocab)
        elif self.starting_model.type == "from_scratch":
            self.vocab = None
        else:
            raise RuntimeError("Starting model type unknown")

        if hasattr(self, "seed_weight_init"):
            torch.random.manual_seed(self.seed_weight_init)

    def convert_dict_to_nametupple(self, name, dico):
        return namedtuple(name, dico.keys())(*dico.values())
