"""
This module define MetricsManager
"""
from metrics.metrics_factory import MetricsFactory
from converter.converters_factory import ConvertersFactory
from utils.utils import move_tensor_to_device


class MetricsManager:
    """
    This class manage which metric is for each task and save the values durring training
    """

    def __init__(self, config):
        tasks = config.tasks
        metric_factory = MetricsFactory()
        converter_factory = ConvertersFactory()
        self.all_metrics_train_all_tasks = []
        self.all_metrics_dev_all_tasks = []
        self.all_metrics_test_all_tasks = []
        self.all_converters_all_tasks = []
        for task_param in tasks:
            all_metric_train_task_i = dict()
            all_metric_dev_task_i = dict()
            all_metric_test_task_i = dict()
            all_converter_task_i = dict()
            for metric in task_param.metrics:
                data_for_metric = self.get_additional_data(metric, config, task_param)

                met_train = metric_factory.get_metrics(metric["name"], data_for_metric)
                all_metric_train_task_i[metric["name"]] = met_train
                met_dev = metric_factory.get_metrics(metric["name"], data_for_metric)
                all_metric_dev_task_i[metric["name"]] = met_dev
                met_test = metric_factory.get_metrics(metric["name"], data_for_metric)
                all_metric_test_task_i[metric["name"]] = met_test
                converter = converter_factory.get_converter(metric["converter"])
                all_converter_task_i[metric["name"]] = converter
            self.all_metrics_train_all_tasks.append(all_metric_train_task_i)
            self.all_metrics_dev_all_tasks.append(all_metric_dev_task_i)
            self.all_metrics_test_all_tasks.append(all_metric_test_task_i)
            self.all_converters_all_tasks.append(all_converter_task_i)

    def get_additional_data(self, metric, config, task_param):
        data_for_metric = dict()
        if "get_path_class_vocab" in metric.keys() and metric["get_path_class_vocab"]:
            path_class_vocab = (
                config.path_folder_json + task_param.unique_task_name + "_vocab.pth"
            )
            data_for_metric["path_class_vocab"] = path_class_vocab
        if "get_label_col_names" in metric.keys() and metric["get_label_col_names"]:
            data_for_metric["label_col_names"] = task_param.col_name_output
        if "get_data_path" in metric.keys() and metric["get_data_path"]:
            data_for_metric["data_path"] = (
                config.root_datasets + task_param.data_path + "/" + task_param.data_path
            )
        return {**data_for_metric, **metric}

    def get_nb_metric_task_i(self, ind_task):
        return len(self.all_metrics_train_all_tasks[ind_task])

    def update_metrics(self, typeMetric, ind_task, output_batch, target_batch):
        # On s'assure qu'a ce moment c'est bien sur le cpu
        # output_batch = output_batch.to("cpu")
        # target_batch = move_tensor_to_device(target_batch, "cpu")

        if typeMetric == "train":
            all_metric = list(self.all_metrics_train_all_tasks[ind_task].values())
        elif typeMetric == "dev":
            all_metric = list(self.all_metrics_dev_all_tasks[ind_task].values())
        elif typeMetric == "test":
            all_metric = list(self.all_metrics_test_all_tasks[ind_task].values())
        else:
            raise RuntimeError(
                "typeMetric unknown, only train/dev or test value are accepted"
            )
        all_converter = list(self.all_converters_all_tasks[ind_task].values())

        for converter, metric in zip(all_converter, all_metric):
            converted_output, converted_target = converter.convert_output_and_target(
                output_batch, target_batch
            )
            metric.update_metric(converted_output, converted_target)

    def reset_all_metric(self, all_metrics):
        for metric in all_metrics:
            metric.reset_and_init_value()

    def reset_all_metrics(self, typeMetric, ind_task):
        if typeMetric == "train":
            all_metrics = list(self.all_metrics_train_all_tasks[ind_task].values())
            self.reset_all_metric(all_metrics)
        elif typeMetric == "dev":
            all_metrics = list(self.all_metrics_dev_all_tasks[ind_task].values())
            self.reset_all_metric(all_metrics)
        elif typeMetric == "test":
            all_metrics = list(self.all_metrics_test_all_tasks[ind_task].values())
            self.reset_all_metric(all_metrics)
        else:
            raise RuntimeError(
                "Unknown typeMetric, only train/dev and test are accepted"
            )

    def get_metric_value(self, typeMetric, ind_task, metric_name):
        if typeMetric == "train":
            return self.all_metrics_train_all_tasks[ind_task][metric_name].get_value()
        elif typeMetric == "dev":
            return self.all_metrics_dev_all_tasks[ind_task][metric_name].get_value()
        elif typeMetric == "test":
            return self.all_metrics_test_all_tasks[ind_task][metric_name].get_value()
        else:
            raise RuntimeError(
                "Unknown typeMetric, only train/dev and test are accepted"
            )
