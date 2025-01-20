import os
import random
import re

import numpy as np
import pandas as pd
import h5py
from IPython.display import display

import tensorflow as tf

from proteinbert import load_pretrained_model_from_dump
from proteinbert.conv_and_global_attention_model import create_model
from proteinbert.pretraining import DEFAULT_EPISODE_SETTINGS, EpochGenerator, DatasetHandler

# Define the cluster number variable
CLUSTER_NUM = 100

MODEL_DUMP_DIRS = [f'proteinbert_models/cluster-{CLUSTER_NUM}/go']
H5_DATASET_FILE_PATH = f'data/cluster-{CLUSTER_NUM}/pretrain_go_final.h5'

def create_epoch(seq_len, batch_size):
    np.random.seed(0)
    epoch_generator = EpochGenerator(episode_settings = [(seq_len, batch_size)])
    
    with h5py.File(H5_DATASET_FILE_PATH, 'r') as h5f:
        epoch_generator.setup(DatasetHandler(h5f))
        return epoch_generator.create_next_epoch()
    
#assert len(tf.config.list_physical_devices('GPU')) > 0

dump_file_paths = sorted([os.path.join(dump_dir, file_name) for dump_dir in MODEL_DUMP_DIRS for file_name in os.listdir(dump_dir)])
random.seed(0)
random.shuffle(dump_file_paths)
    
seq_len_to_epoch = {seq_len: create_epoch(seq_len, batch_size) for seq_len, batch_size in DEFAULT_EPISODE_SETTINGS}
results = []

if os.path.exists(f'fine_tuning_results/cluster-{CLUSTER_NUM}/go/training_loss_{CLUSTER_NUM}.csv'):
    final_df = pd.read_csv(f'fine_tuning_results/cluster-{CLUSTER_NUM}/go/training_loss_{CLUSTER_NUM}.csv')
else:
    final_df = pd.DataFrame(results, columns = ['n_pretraining_epochs', 'n_pretraining_samples', 'seq_len', 'seq_loss', 'annots_loss'])

list_epochs = final_df.n_pretraining_epochs.values
for dump_file_path in dump_file_paths:
    
    print('Dump file: %s' % dump_file_path)
    n_pretraining_epochs, = map(int, re.findall(r'epoch_(\d+)', dump_file_path))
    n_pretraining_samples, = map(int, re.findall(r'sample_(\d+)', dump_file_path))
    if n_pretraining_epochs in list_epochs:
        continue
    else:
        if os.path.exists(dump_file_path):
            model_generator, _ = load_pretrained_model_from_dump(dump_file_path, create_model)
            for seq_len, batch_size in DEFAULT_EPISODE_SETTINGS:
                print('Seq len: %d' % seq_len)
                model = model_generator.create_model(seq_len)
                X, Y, sample_weights = seq_len_to_epoch[seq_len]
                seq_w, annots_w = sample_weights
                _, seq_loss, annots_loss = model.evaluate(X, Y, sample_weight = sample_weights, batch_size = batch_size)
                seq_loss /= seq_w.mean()
                annots_loss /= annots_w.mean()
        
                results.append([n_pretraining_epochs, n_pretraining_samples, seq_len, seq_loss, annots_loss])
        else:
            continue
            
del seq_len_to_epoch
new_results = pd.DataFrame(results, columns = ['n_pretraining_epochs', 'n_pretraining_samples', 'seq_len', 'seq_loss', 'annots_loss'])
final_df = final_df.append(new_results, ignore_index=True)
display(final_df)

print('Done.')
final_df.to_csv(f'fine_tuning_results/cluster-{CLUSTER_NUM}/go/training_loss_{CLUSTER_NUM}.csv', index = False)
