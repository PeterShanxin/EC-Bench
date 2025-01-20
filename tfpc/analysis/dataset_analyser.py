"""
This module allow to analyse dataset
"""
from config_manager.config_manager import ConfigManager
from dataset_manager.training_dataset_manager import TrainingDatasetManager


class DatasetAnalyser:
    """
    This class analyse a dataset
    """

    def __init__(self, path_vocab):
        config = ConfigManager(path_vocab)
        self.dataset_manager = TrainingDatasetManager(config)
