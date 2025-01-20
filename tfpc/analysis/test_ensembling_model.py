"""
This module allow you to test ensembling model
"""
import torch
import os
import logging
from analysis.training_analyser import TrainingAnalyser
from utils.utils import (
    calc_metric_and_loss_on_data,
)
from models_architectures.ensembling_model import EnsemblingModel


class TestEnsemblingModel:
    """
    This class test emsemble model
    """

    def __init__(self, list_of_xp_path, ensembling_name):
        self.ensembling_name = ensembling_name
        self.path_save_ensemble = (
            "data/models/ensembling_models/" + ensembling_name + "/"
        )
        self.list_of_xp_path = list_of_xp_path
        self.list_of_analyser = []
        for path in list_of_xp_path:
            self.list_of_analyser.append(TrainingAnalyser(path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.create_ensembling_model()

    def create_ensembling_model(self):
        list_of_model = []
        for ind, analyser in enumerate(self.list_of_analyser):
            list_tasks = analyser.list_all_tasks()
            if len(list_tasks) > 1:
                for indi, task in enumerate(list_tasks):
                    print(indi, ":", task)
                indice_task = int(input("Choose an indice of a task(q to quit) : "))
            else:
                indice_task = 0
            complete_path_model = (
                self.list_of_xp_path[ind] + "/" + list_tasks[indice_task] + ".pth"
            )
            logging.debug("Path of the model we load is %s", complete_path_model)
            model = torch.load(complete_path_model, map_location=self.device)
            model = model.eval()
            if self.device == "cuda":
                model = torch.nn.DataParallel(model)
            list_of_model.append(model)
        self.model_ens = EnsemblingModel(list_of_model)

    def eval_model(self, ind_model, ind_task):
        analyser = self.list_of_analyser[ind_model]

        calc_metric_and_loss_on_data(
            analyser.dataset_manager.get_test_dataloader(ind_task),
            self.model_ens,
            analyser.all_losses_logger[ind_task],
            analyser.all_metrics_logger[ind_task],
            analyser.weights_modifiers,
            analyser.metrics_manager,
            ind_task,
            analyser.nb_batch_seen[ind_task],
            "test",
        )

        dict_test_metric = analyser.get_all_test_metric()
        logging.debug("This is the dict_test_metric value : %s", dict_test_metric)
        self.save_model_and_metric(dict_test_metric)
        return dict_test_metric

    def list_all_tasks(self, num_model):
        return self.list_of_analyser[num_model].list_all_tasks()

    def save_model_and_metric(self, dict_test_metric):
        if not os.path.isdir(self.path_save_ensemble):
            os.mkdir(self.path_save_ensemble)
        torch.save(dict_test_metric, self.path_save_ensemble + "test_metrics.pkl")
        torch.save(self.list_of_xp_path, self.path_save_ensemble + "list_xp.pkl")
