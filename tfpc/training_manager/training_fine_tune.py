"""
This module define TrainingFineTune
"""
import logging
import torch
from dataset_manager.training_dataset_manager import TrainingDatasetManager
from model_manager.model_factory import ModelFactory
from model_manager.weight_modifier import WeightModifier
from config_manager.config_manager import ConfigManager
from utils.utils import function_not_implemented, move_tensor_to_device
from logger.train_logger_and_saver import TrainLoggerAndSaver
from metrics.metrics_manager import MetricsManager


class TrainingFineTune:
    """
    This class manage all the training at the higher level. It's an orchestrator
     for the training process.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.nb_task = len(config.tasks)
        self.dataset_manager = TrainingDatasetManager(config)
        model_factory = ModelFactory(config, self.dataset_manager)
        self.models = model_factory.create_all_models()
        self.weights_modifiers = WeightModifier(
            config, self.models, self.dataset_manager
        )

        self.metric_manager = MetricsManager(config)

        self.train_logger_and_saver = TrainLoggerAndSaver(
            self.config,
            self.dataset_manager,
            self.models,
            self.weights_modifiers,
            self.metric_manager,
        )

    def train(self):
        logging.info(
            "Je lance un entrainement de type %s", (self.config.training_strategy)
        )
        if self.config.training_strategy == "classic":
            self.train_classique()
        elif self.config.training_strategy == "crossover":
            self.train_crossover()
        else:
            function_not_implemented()

        logging.info("J'ai finit le train correctement")
        self.train_logger_and_saver.finished_XP = True
        self.train_logger_and_saver.export_to_file()

    def train_crossover(self):
        stop_training = False
        # Put all model on GPU
        for ind_task in range(self.nb_task):
            self.put_model_i_on_gpu(ind_task)

        # While all the stoping criteria for all task are not satisfied
        while not stop_training:
            for ind_task in range(self.nb_task):
                train_dataloader = self.dataset_manager.get_train_dataloader(ind_task)
                input_batch, target_batch = next(iter(train_dataloader))
                self.train_one_batch(ind_task, input_batch, target_batch)

                # If we have seen the whole dataset for this task
                if (
                    self.train_logger_and_saver.nb_batch_seen[ind_task]
                    % len(train_dataloader)
                    == 0
                    and self.train_logger_and_saver.nb_batch_seen[ind_task] != 0
                ):
                    logging.debug("WE HAVE SEEN A COMPLETE DATASET")
                    self.train_logger_and_saver.one_epoch_finish(ind_task)
                    mean_met = self.train_logger_and_saver.mean_metric_last_epoch(
                        ind_task
                    )
                    self.weights_modifiers.one_epoch_finish(ind_task, mean_met)

                    # We save the model
                    self.train_logger_and_saver.save_generic_model(ind_task)
                    self.train_logger_and_saver.save_task_model(ind_task)
                    logging.info(
                        "We save the generic and specific model of the task num %s",
                        ind_task,
                    )
                    # We update the stoping criteria just when an epoch is completed
                    stop_training = self.stoping_criteria_multiple()

        # Eject all the model from GPU
        for ind_task in range(self.nb_task):
            self.eject_model_i_from_gpu(ind_task)

    def train_classique(self):
        for ind_task in range(self.nb_task):
            logging.info(
                "Je lance l'entrainement sur la tâche %s",
                (self.config.tasks[ind_task].unique_task_name),
            )
            self.train_task_i(ind_task)
            logging.info("Je viens de finir la tâche numero %s", ind_task)

    def train_task_i(self, ind_task):
        self.put_model_i_on_gpu(ind_task)

        while not self.stoping_criteria(ind_task):
            logging.info(
                "Je lance une epoch sur la tâche %s",
                (self.config.tasks[ind_task].unique_task_name),
            )
            self.train_on_epoch_i(ind_task)

        self.eject_model_i_from_gpu(ind_task)

    def train_on_epoch_i(self, ind_task):
        train_dataloader = self.dataset_manager.get_train_dataloader(ind_task)
        self.set_seed_for_data_shuffle()

        for input_batch, target_batch in train_dataloader:
            self.set_seed_for_dropout()
            self.train_one_batch(ind_task, input_batch, target_batch)

        self.train_logger_and_saver.one_epoch_finish(ind_task)
        mean_met = self.train_logger_and_saver.mean_metric_last_epoch(ind_task)
        self.weights_modifiers.one_epoch_finish(ind_task, mean_met)

        # We save the model
        self.train_logger_and_saver.save_generic_model(ind_task)
        self.train_logger_and_saver.save_task_model(ind_task)
        logging.info(
            "We save the generic and specific model of the task num %s", ind_task
        )

    def set_seed_for_data_shuffle(self):
        if hasattr(self.config, "seed_data_shuffle"):
            torch.random.manual_seed(self.config.seed_data_shuffle)

    def set_seed_for_dropout(self):
        if hasattr(self.config, "seed_dropout"):
            torch.random.manual_seed(self.config.seed_dropout)

    def train_one_batch(self, ind_task, input_batch, target_batch):

        output_batch = self.models[ind_task](input_batch)

        loss = self.weights_modifiers.one_training_batch(
            ind_task, output_batch, target_batch
        )
        # Free GPU memory as soon as we can
        del input_batch
        self.train_logger_and_saver.step(ind_task, loss, output_batch, target_batch)

    def put_model_i_on_gpu(self, indice_model):
        # Put the model on GPU if available
        # Embedding of the model in a dataPrallel object to exploit multiple GPU
        self.models[indice_model] = self.models[indice_model].to(self.config.device)
        self.models[indice_model] = torch.nn.DataParallel(self.models[indice_model])
        logging.info(
            "Au debut du train_task_i le model est sur %s",
            next(self.models[indice_model].parameters()).device,
        )

    def eject_model_i_from_gpu(self, indice_model):
        self.models[indice_model] = self.models[indice_model].module
        self.models[indice_model] = self.models[indice_model].to(torch.device("cpu"))
        logging.info(
            "A la fin du train_task_i le model est sur %s",
            next(self.models[indice_model].parameters()).device,
        )

    def stoping_criteria(self, ind_task):
        if self.config.tasks[ind_task].stoping_criteria["name"] == "no":
            logging.warning("Attention stoping criteria to NO, infinite loop")
            return False
        elif self.config.tasks[ind_task].stoping_criteria["name"] == "yes":
            logging.warning("Attention stoping criteria to YES, stop after one epoch")
            if len(self.train_logger_and_saver.num_batch_each_epoch[ind_task]) != 0:
                return True
            else:
                return False
        elif (
            self.config.tasks[ind_task].stoping_criteria["name"]
            == "slope_on_first_metric"
        ):
            nb_finish_epoch = self.train_logger_and_saver.get_nb_finish_epoch(ind_task)
            nb_epoch_mini = self.config.tasks[ind_task].stoping_criteria["params"][1]
            if nb_finish_epoch < nb_epoch_mini:
                return False
            slope_metric = self.train_logger_and_saver.get_dev_slope_first_metric(
                ind_task
            )
            slope_mini = self.config.tasks[ind_task].stoping_criteria["params"][0]
            if slope_metric < slope_mini:
                return True
            else:
                return False
        elif self.config.tasks[ind_task].stoping_criteria["name"] == "nb_epoch":
            nb_finish_epoch = self.train_logger_and_saver.get_nb_finish_epoch(ind_task)
            nb_epoch_to_reach = self.config.tasks[ind_task].stoping_criteria["params"][
                0
            ]
            if nb_finish_epoch == nb_epoch_to_reach:
                return True
            else:
                return False
        else:
            function_not_implemented()

    def stoping_criteria_multiple(self):
        stop = True
        for ind_task in range(self.nb_task):
            stop = stop and self.stoping_criteria(ind_task)
        return stop
