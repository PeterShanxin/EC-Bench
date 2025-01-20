#!/bin/bash
#SBATCH --job-name=MyModel_EC40_classic_r5
#SBATCH --chdir=/home/genouest/dyliss/nbuton/
#SBATCH --output=tfpc/data/outputs_jobs/output_MyModel_EC40_classic_r5.txt
source python_env_these/bin/activate
cd tfpc/
python3 training.py data/models/fine_tune_models/MyModel_EC40_classic_r5/config.json
