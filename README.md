# EC-Bench : A Benchmark for Enzyme Commission (EC) Number Prediction
## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

## Usage
1. Clone the repository  
```
git clone https://github.com/dsaeedeh/EC-Bench.git
```
### Data Preparation
2. Run download_data.sh to download the data
3. Run extract_coordinates.sh to extract the coordinates from pdb files 
4. run data_preprocessing.sh 
5. Run run_mmseqs2.sh to concat fasta files (pretrain.fasta, train.fasta and test.fasta). Be sure no other .fasta files are existed in the data directory
We need all.fasta file to run mmseqs2 on it!
6. Run create_data.sh to create the final data for pretraining, training, and testing

## Models
## Models

| Model Name        | Link                        |
|-------------------|------------------------------------|
| ProteinBERT       | [GitHub](https://github.com/nadavbra/protein_bert) |
| DeepEC            | [GitHub](https://bitbucket.org/kaistsystemsbiology/deepec/src/master/) |
| ECPred            | [GitHub](https://github.com/cansyl/ECPred) |
| DeepECtransformer | [GitHub](https://github.com/kaistsystemsbiology/DeepProZyme) |
| EnzBERT           | [GitHub](https://gitlab.inria.fr/nbuton/tfpc) |
| ECRECer           | [GitHub](https://github.com/kingstdio/ECRECer) |
| CLEAN             | [GitHub](https://github.com/tttianhao/CLEAN) |
| BLASTp            | [GitHub](https://github.com/bbuchfink/diamond/wiki) |
| CatFam            | [GitHub]() |
| PRIAMv2           | [GitHub]() |


## Contributing

## License



