import os
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

from tensorflow import keras

from . import conv_and_global_attention_model
from .model_generation import load_pretrained_model_from_dump

# 100: epoch_59720_sample_2410000.pkl: 1 week; epoch_122100_sample_21830000: 2 weeks; epoch_183350_sample_4810000: 3 weeks; epoch_249440_sample_18760000: 4 weeks
# 30: epoch_59310_sample_29140000.pkl: 1 week; epoch_119020_sample_2420000: 2 weeks; epoch_183290_sample_4880000: 3 weeks; epoch_245950_sample_26250000: 4 weeks
# 100-go: epoch_56510_sample_69710000.pkl: week 1; epoch_113850_sample_36900000.pkl: week 2; epoch_159360_sample_29360000.pkl: week 3; epoch_212760_sample_77290000.pkl: week 4
# 30-go: epoch_56210_sample_67860000.pkl: week 1; epoch_114050_sample_38420000.pkl: week 2; epoch_159200_sample_28590000.pkl: week 3; epoch_213250_sample_81050000.pkl: week 4
# proteinbert: epoch_92400_sample_23500000
DEFAULT_LOCAL_MODEL_DUMP_DIR = 'proteinbert_models/cluster-30-new/go'
DEFAULT_LOCAL_MODEL_DUMP_FILE_NAME = 'epoch_159200_sample_28590000.pkl'

def load_pretrained_model(local_model_dump_dir=DEFAULT_LOCAL_MODEL_DUMP_DIR,
                          local_model_dump_file_name=DEFAULT_LOCAL_MODEL_DUMP_FILE_NAME,
                          download_model_dump_if_not_exists=True,
                          validate_downloading=True,
                          create_model_function=conv_and_global_attention_model.create_model, create_model_kwargs={},
                          optimizer_class=keras.optimizers.Adam, lr=2e-04,
                          other_optimizer_kwargs={}, annots_loss_weight=1, load_optimizer_weights=False):
    local_model_dump_dir = os.path.expanduser(local_model_dump_dir)
    dump_file_path = os.path.join(local_model_dump_dir, local_model_dump_file_name)


    return load_pretrained_model_from_dump(dump_file_path, create_model_function,
                                           create_model_kwargs=create_model_kwargs, optimizer_class=optimizer_class,
                                           lr=lr,
                                           other_optimizer_kwargs=other_optimizer_kwargs,
                                           annots_loss_weight=annots_loss_weight,
                                           load_optimizer_weights=load_optimizer_weights)
