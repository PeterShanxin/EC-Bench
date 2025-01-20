#!/bin/bash
# run tensorflow in singularity container
export HDF5_USE_FILE_LOCKING=FALSE
python3 multiclass_classifiction.py >& multiclass_finetuning_go.log