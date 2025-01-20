"""
This module define MetricsFactory
"""
from metrics.metric_accuracy import MetricAccuracy
from metrics.metric_spearman_rho import MetricSpearmanRho
from metrics.metric_structure import MetricStructure
from metrics.metric_fmax import MetricFMax
from metrics.metric_smin import MetricSmin


class MetricsFactory:
    """
    This class define a factory for the metric class that heritate from genericMetric
    """

    def get_metrics(self, metric_type, data_for_metric):
        if metric_type == "accuracy":
            return MetricAccuracy()
        elif metric_type == "spearman_rho":
            return MetricSpearmanRho()
        elif metric_type == "metric_structure":
            return MetricStructure()
        elif metric_type == "metric_fmax":
            return MetricFMax(data_for_metric)
        elif metric_type == "metric_smin":
            return MetricSmin(data_for_metric)
        else:
            raise RuntimeError("Metrics type unknown.")
