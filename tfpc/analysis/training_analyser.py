"""
This module define class in order to analyse experiement result
"""
# pylint: disable=no-member
import logging
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from config_manager.config_manager import ConfigManager
from dataset_manager.training_dataset_manager import TrainingDatasetManager
from model_manager.weight_modifier import WeightModifier
from logger.metrics_logger import MetricsLogger
from torchmetrics.functional import accuracy
from metrics.metrics_manager import MetricsManager
from utils.utils import (
    get_slope,
    get_first_metric_dev_value,
    calc_metric_and_loss_on_data,
    return_pretty_time,
    get_attribute_to_save_train_logger_and_save,
    save_fig,
    calc_metric_on_auxiliary_data,
)

sys.path.append("models_architectures/")
# from model_ENCODER import EncoderModel


class TrainingAnalyser:
    """
    This class allow the user to explore the result pickle file of an experiement
    list of attribute inside the pickle file are in the get_attribute_to_save_train_logger_and_save in the utils.py file
    """

    def __init__(self, path_folder_xp, batch_size=2, load_train=False):
        # Save the path of xp
        self.path_folder_xp = path_folder_xp

        # Loading of the config file link to this xp
        self.config = ConfigManager(self.path_folder_xp + "/config.json")

        # Change batch_size if we need
        for k in range(len(self.config.tasks)):
            self.config.tasks[k] = self.config.tasks[k]._replace(batch_size=batch_size)

        # Get XP name
        self.xp_name = path_folder_xp.split("/")[-1]
        logging.info("Xp name : %s", self.xp_name)

        # Load dict that contain info on the training procedure and put them on attribute of this object
        file_log = open(path_folder_xp + "/Train_logger_and_saver.pkl", "rb")
        print(path_folder_xp + "/Train_logger_and_saver.pkl")
        dict_content = torch.load(file_log, map_location=torch.device("cpu"))
        logging.debug("All content key are %s", dict_content.keys())
        self.set_dict_to_class_atribute(dict_content)

        # Loading some needed value
        self.frequency_train_metrics = self.config.log_and_save.frequency_train_metrics
        self.task_names = [t.unique_task_name for t in self.config.tasks]

        # Load dataset manager
        self.dico_loaded_vocab = dict()
        for k in range(len(self.config.tasks)):
            try:
                vocab = torch.load(
                    self.path_folder_xp + "/" + self.task_names[k] + "_vocab.pth",
                    map_location=self.config.device,
                )
                self.dico_loaded_vocab[self.task_names[k]] = vocab
            except:
                pass
        if load_train:
            self.dataset_manager = TrainingDatasetManager(
                self.config, no_train=False, dico_loaded_vocab=self.dico_loaded_vocab
            )
        else:
            self.dataset_manager = TrainingDatasetManager(
                self.config, no_train=True, dico_loaded_vocab=self.dico_loaded_vocab
            )

        # Load model and re create weight modifier object
        self.models = self.load_all_models()
        self.weights_modifiers = WeightModifier(
            self.config, self.models, self.dataset_manager
        )

        # Transfer the loss on the available device
        self.weights_modifiers.losses = [
            l.to(self.config.device) for l in self.weights_modifiers.losses
        ]

        # Loading metric manager to be able to calculate metric on some new dataset
        self.metrics_manager = MetricsManager(self.config)

    def load_all_models(self):
        all_models = []
        for uniq_model_name in self.task_names:
            try:
                logging.info(
                    "Path of the model is %s",
                    self.path_folder_xp + "/" + uniq_model_name + ".pth",
                )
                logging.info("Map location is %s", self.config.device)
                model = torch.load(
                    self.path_folder_xp + "/" + uniq_model_name + ".pth",
                    map_location=self.config.device,
                )
                all_models.append(model)
            except:
                all_models.append(all_models[0])
                logging.info("Some model are not yet available")
        return all_models

    def set_dict_to_class_atribute(self, dico):
        for key, value in dico.items():
            setattr(self, key, value)

    def list_all_tasks(self):
        return self.task_names

    def print_general_infos(self):
        print("#" * 89)
        for key, value in self.material.items():
            print(key, " : ", value)
        self.time_per_epoch_per_task()
        print("#" * 89)

    def time_per_epoch_per_task(self):
        epoch_time = dict()
        for num_task, _ in enumerate(self.task_names):
            all_epochs = self.num_batch_each_epoch[num_task]
            if len(all_epochs) == 0:
                print(
                    "Pas d'epoch encore effectué sur la tache",
                    self.task_names[num_task],
                )
                continue

            num_batch_per_epoch = all_epochs[0]

            timer_tracker_i = self.time_tracker[num_task]
            duration_per_batch = []
            for num_each_time in range(0, len(self.time_tracker[num_task]) - 1):
                num_batch_x = timer_tracker_i[num_each_time][0]
                time_x = timer_tracker_i[num_each_time][1]
                num_batch_xp1 = timer_tracker_i[num_each_time + 1][0]
                time_xp1 = timer_tracker_i[num_each_time + 1][1]

                dur = (time_xp1 - time_x) / (num_batch_xp1 - num_batch_x)
                duration_per_batch.append(dur)
            mean_time_per_batch = np.mean(duration_per_batch)
            epoch_time[self.task_names[num_task]] = (
                mean_time_per_batch * num_batch_per_epoch
            )

        for key, value in epoch_time.items():
            print(
                "Une epoch de la tache",
                key,
                "prend en moyenne",
                return_pretty_time(value),
            )

    def trace_double_graph(self, ind_task):
        # Cut the window in two graph
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle(
            "Xp name :" + self.xp_name + ", task name :" + self.task_names[ind_task]
        )
        # Set title on the graph
        ax[0].set_title(
            "Loss on last batch and loss on the\n validation dataset during training"
        )
        ax[1].set_title(
            "Accuracy on last examples and metric\n on the validation dataset during training"
        )
        # Name the axes
        ax[0].set_ylabel("Loss value")
        ax[0].set_xlabel("Number of batch")
        ax[1].set_ylabel("Metrics values")
        ax[1].set_xlabel("Number of batch")

        # Trace loss
        df_all_loss = self.get_df_loss(ind_task)
        sns.lineplot(
            x="nb_batch",
            y="loss",
            hue="source",
            data=df_all_loss,
            ax=ax[0],
            legend=True,
        )
        # Trace metrics
        df_all_metrics = self.get_df_metric(ind_task, "")
        sns.lineplot(
            x="nb_batch",
            y="metric",
            hue="source",
            data=df_all_metrics,
            ax=ax[1],
            legend=True,
        )

        # Delimitate epoch
        # Plot line to delimitate epochs
        self.trace_epoch_line(ax, self.num_batch_each_epoch[ind_task])

        save_fig(
            "double_graph_task_" + self.task_names[ind_task] + "_xp_" + self.xp_name,
            "data/saved_figures/" + self.xp_name + "/",
        )
        plt.show()

    def trace_epoch_line(self, axis, num_batch_each_epoch):
        print(num_batch_each_epoch)
        for ax in axis:
            for num_batch in num_batch_each_epoch:
                ax.axvline(
                    num_batch,
                    0,
                    10000,
                    alpha=0.4,
                    color="black",
                    label="Epoch",
                    linestyle="--",
                )

    def get_df_metric(self, ind_task, prefix):
        list_of_all = []
        metric_logger = self.all_metrics_logger[ind_task]
        for _, metric_name in enumerate(metric_logger.all_metrics_names):
            train_met = metric_logger.train_metrics[metric_name]
            dev_met = metric_logger.dev_metrics[metric_name]
            df_train = self.create_df(
                "nb_batch", "metric", prefix + "train_" + metric_name, train_met
            )
            df_dev = self.create_df(
                "nb_batch", "metric", prefix + "dev_" + metric_name, dev_met
            )
            list_of_all.append(df_train)
            list_of_all.append(df_dev)

        df_all = pd.concat(list_of_all)
        return df_all

    def get_df_loss(self, ind_task):
        train_loss = self.all_losses_logger[ind_task].get_loss("train")
        df_train = self.create_df("nb_batch", "loss", "train", train_loss)
        dev_loss = self.all_losses_logger[ind_task].get_loss("dev")
        df_dev = self.create_df("nb_batch", "loss", "dev", dev_loss)
        df_all = pd.concat([df_train, df_dev])
        return df_all

    def create_df(self, name_x, name_y, source, data):
        dico = {
            name_x: [el[0] for el in data],
            name_y: [el[1] for el in data],
            "source": [source for _ in range(len(data))],
        }
        df = pd.DataFrame(data=dico)
        df["source"] = df["source"].astype("category")
        return df

    def get_dev_slope_first_metric(self, ind_task):
        metric_val = get_first_metric_dev_value(self.all_metrics_logger[ind_task])
        if len(metric_val) > 3:
            return get_slope(metric_val)
        else:
            return "not available"

    def check_eval_model(self, ind_task):
        eval_test_exist = self.all_losses_logger[ind_task].get_loss("test")
        choice2 = input("Do you want to override the calc metric ?(Y/N)")
        if choice2 == "Y":
            eval_test_exist = None
        if eval_test_exist is None:
            self.eval_model(ind_task)
            choice = input(
                "Do you want to replace the saving in order to include the test metric result(Y/N)"
            )
            if choice == "Y":
                self.overwrite_save_file()

    def calc_and_print_metric(self, ind_task):
        loss, all_metric_result = self.return_loss_and_metrics_value(ind_task)
        print("Loss :", loss)
        for key, value in all_metric_result.items():
            print("Metric ", key, ":", value)

    def overwrite_save_file(self):
        """
        overwrite the pickle dict with all parameter to add test metric in the saving
        """
        attribute_to_save = get_attribute_to_save_train_logger_and_save()
        dict_to_save = self.create_dict(attribute_to_save)
        # open a file, where you ant to store the data
        fichier = open(self.path_folder_xp + "/Train_logger_and_saver.pkl", "wb")
        # dump information to that file
        torch.save(dict_to_save, fichier)
        # close the file
        fichier.close()

    def create_dict(self, attribute_to_save):
        dico = dict()
        for att in attribute_to_save:
            dico[att] = getattr(self, att)
        return dico

    def eval_model(self, ind_task):
        task = self.config.tasks[ind_task]
        complete_path_model = self.path_folder_xp + "/" + task.unique_task_name + ".pth"
        model = torch.load(complete_path_model, map_location=self.config.device)
        if self.config.device == "cuda":
            model = torch.nn.DataParallel(model)
        calc_metric_and_loss_on_data(
            self.dataset_manager.get_test_dataloader(ind_task),
            model,
            self.all_losses_logger[ind_task],
            self.all_metrics_logger[ind_task],
            self.weights_modifiers,
            self.metrics_manager,
            ind_task,
            self.nb_batch_seen[ind_task],
            "test",
        )
        if self.config.device == "cuda":
            model = model.module

    def return_loss_and_metrics_value(self, ind_task):
        """
        return the loss in test for the task num ind_task and return a dict with as key the name a the matric and in value
        the value of the metric on the test set
        """
        self.check_eval_model(ind_task)
        loss = self.all_losses_logger[ind_task].get_loss("test")
        dico_metric = dict()
        metrics_names = self.all_metrics_logger[ind_task].get_all_metric_names()
        for metric_name in metrics_names:
            dico_metric[metric_name] = self.all_metrics_logger[
                ind_task
            ].get_metric_values("test", metric_name)

        return loss, dico_metric

    def get_cluster(self):
        if self.material["pwd"] == "/home/genouest/dyliss/nbuton/tfpc":
            return "genouest"
        elif self.material["pwd"] == "/udd/nbuton/Documents/tfpc":
            return "igrida"
        else:
            return "uknown cluster"

    def get_all_test_metric(self):
        dict_all_metrics_all_task = dict()
        for ind_task, task_name in enumerate(self.task_names):
            dict_all_metrics_all_task[task_name] = dict()
            metrics_names = self.all_metrics_logger[ind_task].get_all_metric_names()
            for metric_name in metrics_names:
                value = self.all_metrics_logger[ind_task].get_metric_values(
                    "test", metric_name
                )
                if value is None:
                    dict_all_metrics_all_task[task_name][metric_name] = "Not calculated"
                else:
                    dict_all_metrics_all_task[task_name][metric_name] = value

        return dict_all_metrics_all_task

    @torch.no_grad()
    def accuracy_model_on_new_data(
        self, ind_task, dataloader, model_name, dataset_name, vocab, predict_class_zero
    ):
        task = self.config.tasks[ind_task]
        complete_path_model = self.path_folder_xp + "/" + task.unique_task_name + ".pth"
        model = torch.load(complete_path_model, map_location=self.config.device)
        if self.config.device == "cuda":
            model = torch.nn.DataParallel(model)
        model_prediction_and_target = open(
            "data/predictions/model_"
            + model_name
            + "_on_dataset_"
            + dataset_name
            + "_with_class_zero_"
            + str(predict_class_zero)
            + ".csv",
            "w",
        )
        inverse_vocab = {value: key for key, value in vocab.items()}
        model_prediction_and_target.write("prediction,target\n")
        model = model.eval()
        for input_batch, target_batch in tqdm(dataloader):
            output_batch = model(input_batch)
            if not predict_class_zero:
                output_batch[:, 2] = -100
            pred_int = output_batch.argmax(axis=1)
            for k in range(len(pred_int)):
                pred_ec = inverse_vocab[int(pred_int[k])]
                target_ec = inverse_vocab[int(target_batch[k])]
                model_prediction_and_target.write(pred_ec + "," + target_ec + "\n")
        model_prediction_and_target.close()
