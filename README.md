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
1. Run download_data.sh to download the data: 
```
sbatch download_data.sh
```
or 
```
./download_data.sh
```
2. Run extract_coordinates.sh to extract the coordinates from the downloaded data:
```
sbatch extract_coordinates.sh
```
3. Run data_preprocessing.sh; Removes duplicates and invalid sequences and non-enzyme sequences from the data
```
sbatch data_preprocessing.sh
```
4. Run run_mmseqs2.sh to concat fasta files (pretrain.fasta, train.fasta, test.fasta, price.fasta, and ensemble.fasta (if existed)). Be sure no other .fasta files are existed in the data directory.
We need "all.fasta" file to run mmseqs2 on it!
```
sbatch run_mmseqs2.sh
```
5. Run create_data.sh to create the final data for training, testing, and ensemble; Output files are: train_ec.csv, test_ec.csv, ensemble_ec.csv for each similarity threshold.
```
sbatch create_data.sh
```
6. Run go_creator.sh to create the GO terms for pretraining data; Output files are: pretrain_go_final.csv
```
sbatch go_creator.sh
```


## Models

|       Model Name        | Link                                                        | Year |
|-------------------|------------------------------------------------------------|------|
|       ProteinBERT       | [LINK](https://github.com/nadavbra/protein_bert)         |  2022    |
|       DeepEC            | [LINK](https://bitbucket.org/kaistsystemsbiology/deepec/src/master/) |   2018   |
|       ECPred            | [LINK](https://github.com/cansyl/ECPred)                 |  2018    |
|       DeepECtransformer | [LINK](https://github.com/kaistsystemsbiology/DeepProZyme) |   2023   |
|       EnzBERT           | [LINK](https://gitlab.inria.fr/nbuton/tfpc)              |   2023   |
|       ECRECer           | [LINK](https://github.com/kingstdio/ECRECer)             |  2022    |
|       CLEAN             | [LINK](https://github.com/tttianhao/CLEAN)               |  2023    |
|       BLASTp            | [LINK](https://github.com/bbuchfink/diamond/wiki)        |  2008    |
|       CatFam            | [LINK](http://www.bhsai.org/downloads/catfam.tar.gz)     |  2008    |
|       PRIAM           | [LINK](http://priam.prabi.fr/REL_JAN18/Distribution.zip)   |  2013    |


## Contributing

## License



