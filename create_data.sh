#!/bin/bash
python3 code/create_data.py --ensemble_file_name ensemble.csv --threshold 30 &> create_data_30.log
python3 code/create_data.py --ensemble_file_name ensemble.csv --threshold 100 &> create_data_100.log