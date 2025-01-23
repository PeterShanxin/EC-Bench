import json
import numpy as np
import pandas as pd
import sys,os
sys.path.append(os.getcwd())
import argparse
from ECRECer.tools import exact_ec_from_uniprot as exactec
from ECRECer.tools import funclib
from ECRECer.tools import minitools as mtool
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from Bio import SeqIO

# We use ECRECer to preprocess the data: https://github.com/kingstdio/ECRECer
def create_tsv_from_data(data_path):
    for file in os.listdir(data_path):
        if file.endswith('.data.gz'):
            print(file)
            exactec.run_exact_task(infile=os.path.join(data_path, file), outfile=os.path.join(data_path, file.replace('.data.gz', '.tsv')))

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def preprocessing(data_path, pretrain_path, train_path, test_path, price_path, ensemble_path):
    # Read the price file
    price = pd.read_csv(price_path)
    
    pretrain_data = pd.read_csv(pretrain_path, sep='\t', header=0)
    pretrain_data = mtool.convert_DF_dateTime(inputdf=pretrain_data)
    pretrain_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    pretrain_data.reset_index(drop=True, inplace=True)

    pretrain_data['ec_number'] = pretrain_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    pretrain_data['ec_number'] = pretrain_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    pretrain_data['functionCounts'] = pretrain_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('pretrain_data finished')
    pretrain = pretrain_data.iloc[:,np.r_[0,2:8,10:12]]

    del pretrain_data

    train_data = pd.read_csv(train_path, sep='\t', header=0)
    train_data = mtool.convert_DF_dateTime(inputdf=train_data)
    train_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    train_data['ec_number'] = train_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    train_data['ec_number'] = train_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    train_data['functionCounts'] = train_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('train_data finished')
    train = train_data.iloc[:,np.r_[0,2:8,10:12]]

    del train_data

    if ensemble_path:
        ensemble_data = pd.read_csv(ensemble_path, sep='\t', header=0)
        ensemble_data = mtool.convert_DF_dateTime(inputdf=ensemble_data)
        ensemble_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
        ensemble_data.reset_index(drop=True, inplace=True)

        ensemble_data['ec_number'] = ensemble_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
        ensemble_data['ec_number'] = ensemble_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
        ensemble_data['functionCounts'] = ensemble_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
        print('ensemble_data finished')
        ensemble = ensemble_data.iloc[:,np.r_[0,2:8,10:12]]

        del ensemble_data

    test_data = pd.read_csv(test_path, sep='\t', header=0)
    test_data = mtool.convert_DF_dateTime(inputdf=test_data)
    test_data.drop_duplicates(subset=['id', 'seq'], keep='first', inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    test_data['ec_number'] = test_data.ec_number.parallel_apply(lambda x: mtool.format_ec(x))
    test_data['ec_number'] = test_data.ec_number.parallel_apply(lambda x: mtool.specific_ecs(x))
    test_data['functionCounts'] = test_data.ec_number.parallel_apply(lambda x: 0 if x=='-' else len(x.split(',')))
    print('test_data finished')
    test = test_data.iloc[:,np.r_[0,2:8,10:12]]

    del test_data

    # remove sequences with no EC number
    pretrain = pretrain[pretrain.ec_number != '-']
    train = train[train.ec_number != '-']
    test = test[test.ec_number != '-']
    if ensemble_path:
        ensemble = ensemble[ensemble.ec_number != '-']

    test =test[~test.seq.isin(train.seq)]
    test =test[~test.seq.isin(pretrain.seq)]
    test.reset_index(drop=True, inplace=True) 

    # Remove changed seqences from test set
    test = test[~test.id.isin(test.merge(pretrain, on='id', how='inner').id.values)]
    test = test[~test.id.isin(test.merge(train, on='id', how='inner').id.values)]
    test.reset_index(drop=True, inplace=True)

    # Remove changed seqences from pretrain set, train set, and ensemble set
    pretrain = pretrain[~pretrain.id.isin(pretrain.merge(price, on='id', how='inner').id.values)]
    train = train[~train.id.isin(train.merge(price, on='id', how='inner').id.values)]
    train.reset_index(drop=True, inplace=True)
    pretrain.reset_index(drop=True, inplace=True)

    if ensemble_path:
        ensemble = ensemble[~ensemble.seq.isin(train.seq)]
        ensemble = ensemble[~ensemble.seq.isin(test.seq)]
        ensemble = ensemble[~ensemble.seq.isin(price.seq)]
        ensemble.reset_index(drop=True, inplace=True)
        
        ensemble = ensemble[~ensemble.id.isin(train.id.values)]
        ensemble = ensemble[~ensemble.id.isin(test.id.values)]
        ensemble = ensemble[~ensemble.id.isin(price.id.values)]
        ensemble.reset_index(drop=True, inplace=True)

    # Trim sequences
    with pd.option_context('mode.chained_assignment', None):
        pretrain.ec_number = pretrain.ec_number.parallel_apply(lambda x : str(x).strip())
        pretrain.seq = pretrain.seq.parallel_apply(lambda x : str(x).strip())

        train.ec_number = train.ec_number.parallel_apply(lambda x : str(x).strip())
        train.seq = train.seq.parallel_apply(lambda x : str(x).strip())

        test.ec_number = test.ec_number.parallel_apply(lambda x : str(x).strip())
        test.seq = test.seq.parallel_apply(lambda x : str(x).strip())

        price.ec_number = price.ec_number.parallel_apply(lambda x : str(x).strip())
        price.seq = price.seq.parallel_apply(lambda x : str(x).strip())

        if ensemble_path:
            ensemble.ec_number = ensemble.ec_number.parallel_apply(lambda x : str(x).strip())
            ensemble.seq = ensemble.seq.parallel_apply(lambda x : str(x).strip())
   
    # Create EC number prediction data
    pretrain = pretrain.iloc[:,np.r_[0,7,4]]
    train = train.iloc[:,np.r_[0,7,4]]
    test = test.iloc[:,np.r_[0,7,4]]
    if ensemble_path:
        ensemble = ensemble.iloc[:,np.r_[0,7,4]]

    funclib.table2fasta(table=pretrain[['id', 'seq']], file_out=os.path.join(data_path, 'pretrain.fasta'))
    funclib.table2fasta(table=train[['id', 'seq']], file_out= os.path.join(data_path, 'train.fasta'))
    funclib.table2fasta(table=test[['id', 'seq']], file_out= os.path.join(data_path, 'test.fasta'))
    funclib.table2fasta(table=price[['id', 'seq']], file_out= os.path.join(data_path, 'price-149.fasta'))

    # save the data to csv files
    pretrain.to_csv(os.path.join(data_path, 'pretrain.csv'), index=False)
    train.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    if ensemble_path:
        funclib.table2fasta(table=ensemble[['id', 'seq']], file_out= os.path.join(data_path, 'ensemble.fasta'))
        ensemble.to_csv(os.path.join(data_path, 'ensemble.csv'), index=False)

    # print the number of sequences in each dataset
    print('Number of sequences in pretrain data:', pretrain.shape[0])
    print('Number of sequences in train data:', train.shape[0])
    print('Number of sequences in test data:', test.shape[0])
    if ensemble_path:
        print('Number of sequences in ensemble data:', ensemble.shape[0])
    print('Number of sequences in price data:', price.shape[0])
    print('Data preprocessing finished')

# Check if we have 3d information for all train and test data
def check_3d_information(data_path, train_path, test_path, info_file_path):
    # get the protein ids from fasta files using biopython
    train_ids = [record.id for record in SeqIO.parse(train_path, 'fasta')]
    test_ids = [record.id for record in SeqIO.parse(test_path, 'fasta')]

    # get the ids from json info_file
    with open(info_file_path, 'r') as f:
        info = json.load(f)
        info_ids = list(info.keys())
    
    # check if all ids are in the info file
    train_ids = list(set(train_ids).intersection(set(info_ids)))
    test_ids = list(set(test_ids).intersection(set(info_ids)))
                     
    print(f'Number of train ids in info file: {len(train_ids)}')
    print(f'Number of test ids in info file: {len(test_ids)}')

    # exclude ids not in info file from train and test fasta data and save them to new files 
    SeqIO.write((record for record in SeqIO.parse(train_path, 'fasta') if record.id in train_ids), os.path.join(data_path, 'train_having_3d.fasta'), 'fasta')
    SeqIO.write((record for record in SeqIO.parse(test_path, 'fasta') if record.id in test_ids), os.path.join(data_path, 'test_having_3d.fasta'), 'fasta')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data merging script')
    parser.add_argument('--data_path', type=str, help='Path to the folder containing the data files')
    parser.add_argument('--pretrain_name', type=str, help='name of pretrain data')
    parser.add_argument('--train_name', type=str, help='name of the train data')
    parser.add_argument('--test_name', type=str, help='name of the test data')
    parser.add_argument('--price_name', type=str, help='name of the price data')
    parser.add_argument('--ensemble_name', type=str, help='name of the ensemble data', default=None)
    args = parser.parse_args()

    # run following functions in order; comment out the functions that have been run
    create_tsv_from_data(data_path=args.data_path)
    preprocessing(data_path=args.data_path, pretrain_path=os.path.join(args.data_path, args.pretrain_name), train_path=os.path.join(args.data_path, args.train_name), test_path=os.path.join(args.data_path, args.test_name), price_path=os.path.join(args.data_path, args.price_name), ensemble_path=os.path.join(args.data_path, args.ensemble_name))
    check_3d_information(data_path=args.data_path, train_path=os.path.join(args.data_path, 'train.fasta'), test_path=os.path.join(args.data_path, 'test.fasta'), info_file_path=os.path.join(args.data_path, 'swissprot_coordinates.json'))