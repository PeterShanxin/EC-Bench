"""
This module is the main script to launch the training process
"""
import argparse
import logging
import torch
from training_manager.training_factory import TrainingFactory
from config_manager.config_manager import ConfigManager

if torch.cuda.is_available():
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.getLogger().setLevel(logging.DEBUG)

# Specification of the argument to specify when you launch the script
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="indicate the config file path")
args = parser.parse_args()

config = ConfigManager(args.config_path)
# test_config_bool(config)

training_factory = TrainingFactory()
training_manager = training_factory.get_training_manager(config)

training_manager.train()
