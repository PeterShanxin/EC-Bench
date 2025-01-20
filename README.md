# EC-Bench : A Benchmark for Enzyme Commission (EC) Number Prediction
## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Create a conda environment with python 3.7
```
conda create -n glm4ec python=3.7
```
3. Activate the environment
```
conda activate glm4ec
```
4. Install requirements
```
pip install -r requirements.txt
```
5. Install [mmseqs2](https://github.com/soedinglab/MMseqs2)

## Usage
1. Clone the repository  
```
git clone https://github.com/dsaeedeh/EC-Bench.git
```
2. Run download_data.sh to download the data
3. Run extract_coordinates.sh to extract the coordinates from pdb files 
4. run data_preprocessing.sh 
5. Run run_mmseqs2.sh to concat fasta files (pretrain.fasta, train.fasta and test.fasta). Be sure no other .fasta files are existed in the data directory
We need all.fasta file to run mmseqs2 on it!
6. Run create_data.sh to create the final data for pretraining, training, and testing

## Models
List of models:
1. ProteinBERT
2. DeepEC
3. ECPred
4. DeepECtransformer
5. EnzBERT
6. ECRECer
7. CLEAN
8. BLASTp
9. CatFam
10. PRIAMv2

## Contributing

## License



