# Code for the paper Predicting enzymatic function of protein sequences with attention

## How to use it

1- First clone this repository:

```
git clone git@gitlab.inria.fr:nbuton/tfpc.git
```

2 (optional)- Install python3.7 if you don't have this python version (Maybe other version are possible but not tested):

```
sudo apt install python3.7
```

3 (optional)- Create and activate a virtual environment:

```
virtualenv tfpc_env
source tfpc_env/bin/activate
```

4- Installed all the python libraries:

```
cd tfpc
python3.7 -m pip install -r requirements.txt
```

5- Download all trained models and datasets (https://doi.org/10.5281/zenodo.7253910):

```
mkdir data
wget https://zenodo.org/records/7253910/files/data.zip
unzip data.zip
rm data.zip
```

### How to run a prediction on your sequence

- Launch the prediction script:

```
python3 predictions.py --chosen_model EnzBert_SwissProt_2021_04 --fasta_path example.fasta --enzyme_a_priori --output_folder_path data --max_seq_lenght 2048 --verbose --output_attentions_scores
```

## Reproduce the table and figure from the paper

Launch the "generate_table_and_fig_for_paper.py" script to regenerate the figure from the paper :

```
python3.7 generate_table_and_fig_for_paper.py table1
python3.7 generate_table_and_fig_for_paper.py table3
python3.7 generate_table_and_fig_for_paper.py table4
python3.7 generate_table_and_fig_for_paper.py table5_and_figure4
```

- Figure 5 can be generated with a Jupyter notebook: "jupyter_notebook/1D_and_3D_example_for_paper.ipynb"

## How to fine-tune a model:

- In the data/models/fine_tune_models directory create a folder with the name of your choice (YOUR_FOLDER_NAME)
- In this folder create a config.json file like this: docs/config_EnzBert_EC40.md (More info in docs/possible_values_config.md)
- Launch the fine-tuning script:

```
python3.7 training.py data/fine_tune_models/YOUR_FOLDER_NAME/config.json
```