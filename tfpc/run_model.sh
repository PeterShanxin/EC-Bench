#!/bin/bash

virtualenv tfpc_env
source tfpc_env/bin/activate

python3.7 -m pip install -r tfpc/requirements.txt

# Install DIAMOND if not installed
if ! command -v diamond &> /dev/null; then
    echo "DIAMOND not found. Installing DIAMOND..."
    wget http://github.com/bbuchfink/diamond/releases/download/v2.1.8/diamond-linux64.tar.gz
    tar xzf diamond-linux64.tar.gz
    mv diamond /usr/local/bin/diamond
    chmod +x /usr/local/bin/diamond
    rm diamond-linux64.tar.gz
    echo "DIAMOND installed."
else
    echo "DIAMOND is already installed."
fi

wget https://zenodo.org/records/7253910/files/data.zip
# unzip data.zip into data folder
unzip data.zip -d tfpc/data
rm data.zip

python3 tfpc/convertor.py
cd tfpc
python3 blastp.py
python3 training.py data/models/fine_tune_models/mine_100/config.json &
python3 training.py data/models/fine_tune_models/mine_30/config.json &
cd ../

python3 tfpc/predictions.py --cluster 30 --dataset price-149 --chosen_model mine_30 --fasta_path data/price-149.fasta --enzyme_a_priori --output_folder_path results/tfpc --max_seq_lenght 2048 --verbose
python3 tfpc/predictions.py --cluster 100 --dataset price-149 --chosen_model mine_100 --fasta_path data/price-149.fasta --enzyme_a_priori --output_folder_path results/tfpc --max_seq_lenght 2048 --verbose
python3 tfpc/predictions.py --cluster 30 --dataset test --chosen_model mine_30 --fasta_path data/test_ec.fasta --enzyme_a_priori --output_folder_path results/tfpc --max_seq_lenght 4096 --verbose
python3 tfpc/predictions.py --cluster 100 --dataset test --chosen_model mine_100 --fasta_path data/test_ec.fasta --enzyme_a_priori --output_folder_path results/tfpc --max_seq_lenght 4096 --verbose
python3 tfpc/predictions.py --cluster 30 --dataset ens --chosen_model mine_30 --fasta_path data/ens-30.fasta --enzyme_a_priori --output_folder_path results/tfpc --max_seq_lenght 4096 --verbose
python3 tfpc/predictions.py --cluster 100 --dataset ens --chosen_model mine_100 --fasta_path data/ens-100.fasta --enzyme_a_priori --output_folder_path results/tfpc --max_seq_lenght 4096 --verbose
deactivate
