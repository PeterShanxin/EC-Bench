import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Define the cluster number variable
CLUSTER_NUM = 100

loss_training = pd.read_csv(f'fine_tuning_results/cluster-{CLUSTER_NUM}/go/training_loss_{CLUSTER_NUM}.csv')
epoch_and_sample_pairs = []

for i in range(len(loss_training)):
    epoch_and_sample_pairs.append((int(loss_training['n_pretraining_epochs'][i]), int(loss_training['n_pretraining_samples'][i])))

epoch_and_sample_pairs = pd.DataFrame(epoch_and_sample_pairs, columns=['epoch', 'sample']).sort_values('epoch').reset_index(drop=True)
display(epoch_and_sample_pairs)

epoch_and_sample_diffs = epoch_and_sample_pairs.diff().dropna()
epoch_and_sample_diffs = epoch_and_sample_diffs[epoch_and_sample_diffs['sample'] > 0]
samples_per_epochs = epoch_and_sample_diffs['sample'] / epoch_and_sample_diffs['epoch']

print('Average # of samples per epoch: %d' % samples_per_epochs.mean())

N_SAMPLES_PER_EPOCH = samples_per_epochs.mean()

LOSS_NAME_MAPPING = {
    # original_name: (full_name, metric_name)
    'seq_loss': ('Sequence loss', 'categorical cross-entropy'),
    'annots_loss': ('Annotation loss', 'binary cross-entropy'),
}

max_samples = N_SAMPLES_PER_EPOCH * loss_training['n_pretraining_epochs'].max()
print('Max samples: %d' % max_samples)

fig, axes = plt.subplots(ncols=2, figsize=(12, 3))
fig.subplots_adjust(wspace=0.3)

for i, (ax, loss) in enumerate(zip(axes, loss_training.columns[[-2, -1]])):
    
    loss_full_name, loss_metric_name = LOSS_NAME_MAPPING[loss] 
    
    for seq_len, seq_len_results in loss_training.groupby('seq_len'):
        seq_len_results = seq_len_results.sort_values('n_pretraining_epochs')
        ax.plot(N_SAMPLES_PER_EPOCH * seq_len_results['n_pretraining_epochs'], seq_len_results[loss], label='Length = %d' % seq_len, alpha=0.5)
        
    legend = ax.legend()
    
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    
    ax.set_title(loss_full_name, fontsize=14)
    ax.set_xlim((0, max_samples))
    ax.set_xlabel('# processed samples', fontsize=12)
    ax.set_ylabel(loss_metric_name, fontsize=12)

fig.savefig(f'fine_tuning_results/cluster-{CLUSTER_NUM}/go/pretrain_loss_{CLUSTER_NUM}.png', dpi=1200, bbox_inches="tight")