#!/bin/bash

virtualenv tfpc_env
source tfpc_env/bin/activate

python3.7 -m pip install -r requirements.txt

wget https://zenodo.org/records/7253910/files/data.zip
# unzip data.zip into data folder
unzip data.zip -d tfpc/data
rm data.zip

python3 tfpc/convertor.py 
cd tfpc
python3 training.py data/models/fine_tune_models/mine_100_task3/config.json
python3 blastp.py
cd ../
deactivate

