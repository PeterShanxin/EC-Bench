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

| Model Name        | Link                                                        | Year |
|-------------------|------------------------------------------------------------|------|
| ProteinBERT       | [LINK](https://github.com/nadavbra/protein_bert)         |  2022    |
| DeepEC            | [LINK](https://bitbucket.org/kaistsystemsbiology/deepec/src/master/) |   2018   |
| ECPred            | [LINK](https://github.com/cansyl/ECPred)                 |  2018    |
| DeepECtransformer | [LINK](https://github.com/kaistsystemsbiology/DeepProZyme) |   2023   |
| EnzBERT           | [LINK](https://gitlab.inria.fr/nbuton/tfpc)              |   2023   |
| ECRECer           | [LINK](https://github.com/kingstdio/ECRECer)             |  2022    |
| CLEAN             | [LINK](https://github.com/tttianhao/CLEAN)               |  2023    |
| BLASTp            | [LINK](https://github.com/bbuchfink/diamond/wiki)        |  2008    |
| CatFam            | [LINK](http://www.bhsai.org/downloads/catfam.tar.gz)     |  2008    |
| PRIAM           | [LINK](http://priam.prabi.fr/REL_JAN18/Distribution.zip)   |  2013    |


## Contributing

## License



