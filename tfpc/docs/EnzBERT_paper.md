# Predicting enzymatics function of proteinsequences with attention
EnzBert is a tools that predict Enzyme Commission (EC) number from sequence only and that can pin point which residues is more important for the prediction. For more detail please read our paper [Predicting enzymatics function of protein sequences with attention](http://en.wikipedia.org/)
## First start
- Clone this repository or download a freeze version on [Zenodo](http://code_for_paper_replication)
- Download all the necessary data on [Zenodo](http://data_with_code) and place this on a data directory at the root
- Install python3.6 or more recent version if you don't have it
- Install all needed dependencies with (pytorch/matplotlib/) -> Créer fichier requierment

## Using EnzBert to predict Enzyme Commission number and important residues for your sequence
- Navigate to the jupyter notebook folder of the project
- Oppen a terminal/console
- Type :
```
jupyter notebook
```
- Open the "Use finetune model simple" notebook 
- Replace the sequence with your sequence in the second cells and change the protein name if you want
- Execute the cells with Ctrl+Enter

## Replication of experience of the paper
###  Experience on EC40
#### Finetunning
```
python3 training.py data/models/fine_tune_models/ProtBert_EC40_layer_norm_proba_r1/config.json
```

#### Evaluation metric
```
python3 explore_result.py eval_on_EC40_dataset
```
OR
```
python3 explore_result.py one
INFO:root:I will search xp in the folder data/models/fine_tune_models/
1 : ProtBert_EC40
2 : ProtBert_ECPred
which experience do you want to analyse ?
1
Do you want to load the train dataset ?(0/1)0
0 : classif_EC_pred_lvl_2
Choose an indice of a task(q to quit) : 0
Do you want to plot the graph ?(Y/N)N
Do you want to evaluate the task on test set ?(Y/N)Y
Do you want to override the calc metric ?(Y/N)N
```

### Experience on EC_pred train
#### Finetunning
```
python3 training.py data/models/fine_tune_models/ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r1/config.json
```

#### Evaluation metric
```
python3 explore_result.py eval_on_ECPred_dataset
```
OR
```
python3 explore_result.py one
INFO:root:I will search xp in the folder data/models/fine_tune_models/
1 : ProtBert_EC40_layer_norm_proba_r1
2 : ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r1
which experience do you want to analyse ?
2 # Choose the number that correspond
Do you want to load the train dataset ?(0/1)0
0 : classif_EC_pred_lvl_2
Choose an indice of a task(q to quit) : 0
Do you want to plot the graph ?(Y/N)N
Do you want to evaluate the task on test set ?(Y/N)Y
Do you want to override the calc metric ?(Y/N)N
```

### Experience on interpretability
#### Finetune a network on all latest data
```
python3 training.py data/models/fine_tune_models/ProtBert_model_SwissProt_2021_04/config.json
```
#### Generate the scores on M-CSA sequences
```
python3 explore_result.py get_residues_of_interesets_automatic
NFO:root:I will search xp in the folder data/models/fine_tune_models/
1 : ProtBert_EC40_layer_norm_proba_r1
2 : ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r1
3 : ProtBert_model_SwissProt_2021_04
which experience do you want to analyse ?
3
```
That will create all scores for all interpretability methods on all the sequences of M-CSA. You can find result in the data/residues_of_interest folder.
#### Evaluation metric and Graphic
To graph token-F1 in respect to number of token, graph the precision recall curve and AP, PR_AUC, max-F1 value in the console
```
python3 explore_result.py compare_intepretability_methods
Choose your datasets (selected_res or catalytic_site)catalytic_site
0 - max_follow_by_mean_order3
1 - LIME_with_5000_corupt_seq_with_m_replacment_character
2 - max_follow_by_mean_order2
3 - Rollout
4 - GradCam
5 - max_follow_by_max_order3
6 - max_follow_by_mean_order1
7 - mean_follow_by_max_order3
8 - mean_follow_by_max_order2
9 - max_follow_by_max_order2
10 - mean_follow_by_mean_order1
11 - mean_follow_by_mean_order3
12 - mean_follow_by_mean_order1_Transformer_untrainned
13 - integrated_grad
14 - Attn_last_layer
15 - LRP_rollout_cls
16 - mean_follow_by_mean_order1_Transformer_untrainned_old
17 - LRP_rollout_sum_col
18 - mean_follow_by_mean_order2
19 - Gradients
20 - InputXGrad
21 - old_interpretability_to_lvl_2
22 - max_follow_by_max_order1
23 - mean_follow_by_max_order1
Which method do you want ? (indice seperate by comma) : 5,6
Do you want to simplified the label ?(Y/N) :N
```