#!/bin/bash
python3 ./bin/set_h5_testset.py --uniprot-ids-file=data/cluster-30/ids_to_remove.txt --h5-dataset-file=data/cluster-30/pretrain_final.h5 >& h5-test-30.log