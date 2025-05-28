# EC-Bench: A Benchmark for Enzyme Commission Number Prediction

**EC-Bench** is a benchmark framework for evaluating enzyme annotation models that predict Enzyme Commission (EC) numbers from protein sequences. EC numbers describe the biochemical reactions enzymes catalyze, and predicting them accurately is essential for understanding protein function.

While many EC prediction methods already exist — including homology-based tools, deep learning models, contrastive learning techniques, and protein language models — there's been no consistent way to evaluate and compare their performance. **EC-Bench** fills this gap by offering a unified, open-source platform.


![EC-Bench Workflow](./figures/Figure2.svg)


## What EC-Bench provides?
- ✅ A curated dataset and preprocessing pipeline.
- 🔬 A selection of 10 diverse EC prediction models.
- 📊 Standardized evaluation metrics for:
  - Accuracy: F1 score, precision, recall.
  - Efficiency: memory usage, runtime, storage usage.
- 🧪 Support for multiple prediction tasks:
  - **Exact EC Number Prediction**: Predict complete EC numbers at all hierarchy levels.
  - **EC Number Completion**: Fill in missing parts of partial EC numbers.
  - **EC Number Recommendation**: Suggest possible EC numbers for new or partially known enzymes.

## Why use EC-Bench?
- 🔁 Fair and reproducible model comparison.
- 📈 Insight into strengths and limitations of different methods.
- 🧬 Evaluation across both curated and challenging datasets (e.g., Price-149).

Whether you're building a new EC prediction model or exploring the performance of existing ones, **EC-Bench** offers a robust and flexible environment to support your research in enzyme function prediction.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Models](#models)
- [Usage](#usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation
1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Clone the repository  
```
git clone https://github.com/dsaeedeh/EC-Bench.git
```
3. Install each model by following the instructions in their respective README files.

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
We welcome contributions from the community! You can:

- 🧩 Add new models to the benchmark.
- 📊 Suggest or implement new evaluation metrics.
- 🛠 Improve preprocessing pipelines or dataset support.
- 📝 Report issues or suggest enhancements.

To contribute:
1. Fork the repository.
2. Open a pull request with your proposed changes.
3. If you are adding a model, include clear instructions and dependencies.

Feel free to open an issue for discussion before submitting major changes.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{EC-Bench,
  title={EC-Bench: A Benchmark for Enzyme Commission Number Prediction},
  author={Saeedeh Davoudi, Christopher S. Henry, Christopher S. Miller, Farnoush Banaei-Kashani},
  journal={Journal Name},
  year={2025}
}
```

## License
EC-Bench is distributed under the [MIT License](https://github.com/dsaeedeh/EC-Bench/blob/main/LICENSE).




