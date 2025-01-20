"""
This module define the logger and saver
"""
import logging
import time
import torch
import numpy as np
from logger.loss_logger import LossLogger
from logger.metrics_logger import MetricsLogger
from utils.utils import (
    get_material_and_infos,
    get_slope,
    get_first_metric_dev_value,
    calc_metric_and_loss_on_data,
    get_attribute_to_save_train_logger_and_save,
    calc_metric_on_auxiliary_data,
)


class TrainLoggerAndSaver:
    """
    This class use some lossLogger and MetricsLogger to save all loss
    and metrics during the training process
    """

    def __init__(
        self,
        config,
        dataset_manager,
        models,
        weights_modifiers,
        metrics_manager,
    ):
        tasks = config.tasks
        log_and_save = config.log_and_save
        self.list_all_save_even_if_worse = [t.save_even_if_worse for t in tasks]
        self.best_metric = []
        self.config = config
        self.metrics_manager = metrics_manager
        self.dataset_manager = dataset_manager
        self.models = models
        self.weights_modifiers = weights_modifiers
        self.frequency_train_metrics = log_and_save.frequency_train_metrics
        self.task_names = []
        self.all_nb_batch_log = []
        self.all_nb_batch_save = []
        self.nb_batch_seen = []
        self.num_batch_each_epoch = []
        self.time_tracker = []
        self.lr_tracker = []
        self.all_losses_logger = []
        self.all_metrics_logger = []
        self.finished_XP = False

        # Save on which material we run the training
        self.material = get_material_and_infos()

        for (index, _) in enumerate(tasks):
            self.task_names.append(tasks[index].unique_task_name)
            nb_batch_one_epoch = dataset_manager.get_nb_batch(index)
            logging.debug(
                "Il y a %s batch dans une epoch de la tache %s",
                nb_batch_one_epoch,
                tasks[index].unique_task_name,
            )
            logging.debug("log_and_save object : %s", (log_and_save))
            if log_and_save.type == "nb_batch":
                nb_batch_log = log_and_save.log
                self.all_nb_batch_log.append(nb_batch_log)
                nb_batch_save = log_and_save.save
                self.all_nb_batch_save.append(nb_batch_save)

            elif log_and_save.type == "nb_time_per_epoch":
                log_time_per_epoch = log_and_save.log
                save_time_per_epoch = log_and_save.save
                nb_batch_log = int(nb_batch_one_epoch / log_time_per_epoch)
                nb_batch_save = int(nb_batch_one_epoch / save_time_per_epoch)
                self.all_nb_batch_log.append(nb_batch_log)
                self.all_nb_batch_save.append(nb_batch_save)
            else:
                raise RuntimeError("log and save type is unknonw")
            logging.debug("self.all_nb_batch_log  : %s", str(self.all_nb_batch_log))
            logging.debug("self.all_nb_batch_save  : %s", str(self.all_nb_batch_save))

            if nb_batch_log > nb_batch_one_epoch or nb_batch_save > nb_batch_one_epoch:
                raise RuntimeError(
                    "The frequency of logging is two low to allow at least one log by epoch"
                )
            if nb_batch_save > nb_batch_one_epoch:
                raise RuntimeError(
                    "The frequency of save is two low to allow at least one log by epoch"
                )

            self.nb_batch_seen.append(0)
            self.best_metric.append(0)
            self.num_batch_each_epoch.append([])
            self.all_losses_logger.append(LossLogger())

            if hasattr(self.config, "auxiliary_evaluation"):
                self.auxiliary_metrics_logger = []

            self.all_metrics_logger.append(MetricsLogger(tasks[index].metrics))

            self.time_tracker.append([])
            self.lr_tracker.append([])
            logging.debug("Longueur tasks : %d", len(tasks))

        logging.debug("Longueur all_nb_batch_log : %d", len(self.all_nb_batch_log))
        self.train_metrics_counter = [0 for _ in range(len(tasks))]
        # We take the min in the folowing because instead we can have some loging step where we have no training example record
        self.train_metrics_each = [
            min(
                int(1 / self.frequency_train_metrics),
                self.all_nb_batch_log[i],
            )
            for i in range(len(tasks))
        ]

    def step(self, ind_task, train_loss, output_batch, target_batch):
        self.nb_batch_seen[ind_task] += 1
        # On ajoute le train loss au logger
        self.all_losses_logger[ind_task].add_loss_tmp("train", train_loss)
        if (
            self.nb_batch_seen[ind_task]
            % self.weights_modifiers.all_nb_steps_to_complete_theorical_batch[ind_task]
            == 0
        ):
            self.all_losses_logger[ind_task].mean_train_loss(
                self.nb_batch_seen[ind_task]
            )
        # On garde en memoire un partie des exemples de train pour pouvoir estimer la métrique
        # selon le paramettre de configuration frequency_train_metrics
        self.sample_calc_metric_train(ind_task, output_batch.cpu(), target_batch)

        # Free GPU/CPU memory usage
        del output_batch
        del target_batch

        # In a regular basis we calculate the mean loss and mean metrics on dev and test
        if self.nb_batch_seen[ind_task] % self.all_nb_batch_log[ind_task] == 0:
            logging.info("%d batchs on été effectués", (self.nb_batch_seen[ind_task]))
            logging.info("Au temps %d", time.time())
            self.get_train_metric_values(ind_task)
            self.eval_in_dev(ind_task)
            self.export_to_file()
            self.time_tracker[ind_task].append(
                [self.nb_batch_seen[ind_task], time.time()]
            )
            self.lr_tracker[ind_task].append(
                [
                    self.nb_batch_seen[ind_task],
                    self.weights_modifiers.optimizers[ind_task].param_groups[0]["lr"],
                ]
            )

        if self.nb_batch_seen[ind_task] % self.all_nb_batch_save[ind_task] == 0:
            logging.info("I save the model")
            logging.info("%d batchs on été effectués", (self.nb_batch_seen[ind_task]))
            logging.info("Au temps %d", time.time())
            self.save_generic_model(ind_task)
            self.save_task_model(ind_task)

    def get_train_metric_values(self, ind_task):
        metric_logger = self.all_metrics_logger[ind_task]
        for metric_name in metric_logger.dev_metrics.keys():
            value_metric = self.metrics_manager.get_metric_value(
                "train", ind_task, metric_name
            )
            logging.info(
                "%s on train on task number %s is %s",
                metric_name,
                ind_task,
                str(value_metric),
            )
            metric_logger.add_metric(
                "train", metric_name, self.nb_batch_seen[ind_task], value_metric
            )
            logging.debug(
                "First train metric list is  : %s",
                str(get_first_metric_dev_value(self.all_metrics_logger[ind_task])),
            )
        self.metrics_manager.reset_all_metrics("train", ind_task)

    def sample_calc_metric_train(self, ind_task, output_batch, target_batch):
        self.train_metrics_counter[ind_task] += 1
        if self.train_metrics_counter[ind_task] == self.train_metrics_each[ind_task]:
            self.metrics_manager.update_metrics(
                "train", ind_task, output_batch, target_batch
            )
            self.train_metrics_counter[ind_task] = 0

    def one_epoch_finish(self, ind_task):
        self.num_batch_each_epoch[ind_task].append(self.nb_batch_seen[ind_task])
        logging.info(
            "%d epochs on été effectués", (len(self.num_batch_each_epoch[ind_task]))
        )

    def mean_metric_last_epoch(self, ind_task):
        first_metric_name = self.all_metrics_logger[ind_task].get_all_metric_names()[0]
        metric_values = self.all_metrics_logger[ind_task].get_metric_values(
            "dev", first_metric_name
        )
        nb_value_per_epoch = int(
            self.num_batch_each_epoch[ind_task][0] / self.all_nb_batch_log[ind_task]
        )
        return np.mean(metric_values[-nb_value_per_epoch:])

    def eval_in_dev(self, ind_task):
        if hasattr(self.config, "auxiliary_evaluation"):
            calc_metric_on_auxiliary_data(
                self.dataset_manager.auxiliary_dataloader_dev,
                self.models[ind_task],
                self.auxiliary_metrics_logger,
                self.nb_batch_seen[ind_task],
                "dev",
            )
        calc_metric_and_loss_on_data(
            self.dataset_manager.get_dev_dataloader(ind_task),
            self.models[ind_task],
            self.all_losses_logger[ind_task],
            self.all_metrics_logger[ind_task],
            self.weights_modifiers,
            self.metrics_manager,
            ind_task,
            self.nb_batch_seen[ind_task],
            "dev",
        )

    def export_to_file(self):
        attribute_to_save = get_attribute_to_save_train_logger_and_save()
        if hasattr(self.config, "auxiliary_evaluation"):
            attribute_to_save += ["auxiliary_metrics_logger"]
        dict_to_save = self.create_dict(attribute_to_save)
        # open a file, where you ant to store the data
        fichier = open(
            self.config.path_folder_json + "Train_logger_and_saver.pkl", "wb"
        )
        # dump information to that file
        torch.save(dict_to_save, fichier)
        # close the file
        fichier.close()

    def create_dict(self, attribute_to_save):
        dico = dict()
        for att in attribute_to_save:
            dico[att] = getattr(self, att)
        return dico

    def get_nb_finish_epoch(self, ind_task):
        return len(self.num_batch_each_epoch[ind_task])

    def get_dev_slope_first_metric(self, ind_task):
        metric_val = get_first_metric_dev_value(self.all_metrics_logger[ind_task])
        return get_slope(metric_val)

    def save_generic_model(self, ind_task):
        if self.we_have_to_save(ind_task):
            generic_model = self.models[ind_task].module.transformer_embedder
            saving_path = self.config.path_folder_json + "transformer_embedder.pth"
            logging.info("We save the generic model")
            torch.save(generic_model, saving_path)

    def save_task_model(self, ind_task):
        if self.we_have_to_save(ind_task):
            # We save the model
            model_to_save = self.models[ind_task].module
            model_saving_path = (
                self.config.path_folder_json
                + self.config.tasks[ind_task].unique_task_name
                + ".pth"
            )
            logging.info("We save the model of the task num %d", ind_task)
            logging.debug("%s is the saving path", model_saving_path)
            torch.save(model_to_save, model_saving_path)
            if "classification" in self.dataset_manager.get_pred_type(ind_task):
                # We save the vocabulary
                vocab_saving_path = (
                    self.config.path_folder_json
                    + self.config.tasks[ind_task].unique_task_name
                    + "_vocab.pth"
                )
                vocab_to_save = self.dataset_manager.get_vocab(ind_task)
                torch.save(vocab_to_save, vocab_saving_path)

    def we_have_to_save(self, ind_task):
        first_metric_name = self.all_metrics_logger[ind_task].get_all_metric_names()[0]
        all_metric_values = self.all_metrics_logger[ind_task].get_metric_values(
            "dev", first_metric_name
        )
        if len(all_metric_values) != 0:
            last_metric_value = all_metric_values[0]
            if last_metric_value > self.best_metric[ind_task]:
                better_metric = True
            else:
                better_metric = False

            return self.list_all_save_even_if_worse[ind_task] or better_metric
        else:
            logging.info(
                "We don't have metric value for now, so we don't know if the model it's better than the last save, so we save."
            )
            return True
