#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
python3 bin/pretrain_proteinbert.py --dataset-file=data/cluster-30/pretrain_go_final.h5 --autosave-dir=proteinbert_models/cluster-30-new/go >& data/cluster-30/pretrain-30.log