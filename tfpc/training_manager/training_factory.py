"""
This module define TrainingFactory
"""
from training_manager.training_fine_tune import TrainingFineTune
from utils.utils import function_not_implemented
from config_manager.config_manager import ConfigManager


class TrainingFactory:
    """
    this class define a factory for the training object
    """

    def __init__(self):
        pass

    def get_training_manager(self, config: ConfigManager) -> TrainingFineTune:
        if config.type == "pre_training":
            function_not_implemented()
        elif config.type == "fine_tuning":
            return TrainingFineTune(config)
        else:
            raise RuntimeError("training type unknown.")
