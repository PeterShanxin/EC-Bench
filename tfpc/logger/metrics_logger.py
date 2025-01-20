"""
This module define metric logger
"""


class MetricsLogger:
    """
    This class keep track of all the metrics for one task
    """

    def __init__(self, all_metrics):
        self.all_metrics_names = [a["name"] for a in all_metrics]
        self.train_metrics = dict()
        self.dev_metrics = dict()
        self.test_metrics = dict()
        for metric in self.all_metrics_names:
            self.train_metrics[metric] = []
            self.dev_metrics[metric] = []
            self.test_metrics[metric] = None

    def add_metric(self, typeMetric, metric_name, nb_batch_seen, value_metric):
        if typeMetric == "train":
            self.train_metrics[metric_name].append([nb_batch_seen, value_metric])
        elif typeMetric == "dev":
            self.dev_metrics[metric_name].append([nb_batch_seen, value_metric])
        elif typeMetric == "test":
            self.test_metrics[metric_name] = value_metric
        else:
            raise RuntimeError("Unknown typeMetric train/dev or test only accepted")

    def get_metric_values(self, typeMetric, metric_name):
        if typeMetric == "train":
            return [d[1] for d in self.train_metrics[metric_name]]
        elif typeMetric == "dev":
            return [d[1] for d in self.dev_metrics[metric_name]]
        elif typeMetric == "test":
            return self.test_metrics[metric_name]
        else:
            raise RuntimeError("Only typeMetric value train/dev/test are accepted")

    def get_metric_couple(self, typeMetric, metric_name):
        if typeMetric == "train":
            return self.train_metrics[metric_name]
        elif typeMetric == "dev":
            return self.dev_metrics[metric_name]
        else:
            raise RuntimeError(
                "Only typeMetric value train/dev are accepted, test don't have num_batch because its only at the end of the trainning"
            )

    def get_all_metric_names(self):
        return self.all_metrics_names
