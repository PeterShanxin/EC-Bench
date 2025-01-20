#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
python3 bin/pretrain_proteinbert.py --dataset-file=data/cluster-100/pretrain_go_final.h5 --autosave-dir=proteinbert_models/cluster-100/go >& data/cluster-100/pretrain-100.log