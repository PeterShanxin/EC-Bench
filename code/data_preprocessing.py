import numpy as np
import pandas as pd
import sys,os
sys.path.append(os.getcwd())
import argparse
from ECRECer.tools import exact_ec_from_uniprot as exactec
from ECRECer.tools import funclib
from ECRECer.tools import minitools as mtool
from pandarallel import pandarallel
from Bio import SeqIO
pandarallel.initialize(progress_bar=True)

# We use ECRECer to preprocess the data: https://github.com/kingstdio/ECRECer
def create_tsv_from_data(data_path):
    for file in os.listdir(data_path):
        if file.endswith('.data.gz'):
            print(file)
            exactec.run_exact_task(infile=os.path.join(data_path, file), outfile=os.path.join(data_path, file.replace('.data.gz', '.tsv')))

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def preprocessing(pretrain_path, train_path, test_path, price_path, ensemble_path):
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
        ensemble = ensemble[~ensemble.seq.isin(pretrain.seq)]
        ensemble = ensemble[~ensemble.seq.isin(test.seq)]
        ensemble = ensemble[~ensemble.seq.isin(price.seq)]
        ensemble.reset_index(drop=True, inplace=True)

        ensemble = ensemble[~ensemble.id.isin(test.id.values)]
        ensemble = ensemble[~ensemble.id.isin(train.id.values)]
        ensemble = ensemble[~ensemble.id.isin(pretrain.id.values)]
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

    funclib.table2fasta(table=pretrain[['id', 'seq']], file_out='data/pretrain.fasta')
    funclib.table2fasta(table=train[['id', 'seq']], file_out='data/train.fasta')
    funclib.table2fasta(table=test[['id', 'seq']], file_out='data/test.fasta')
    funclib.table2fasta(table=price[['id', 'seq']], file_out='data/price.fasta')
    if ensemble_path:
        funclib.table2fasta(table=ensemble[['id', 'seq']], file_out='data/ensemble.fasta')
 
'''
pretrain: 108,857,557
train: 556,822
test: 2601
price: 184
all: 109417164
'''
def count_protein_number(fasta_file):
    count = 0
    for record in SeqIO.parse(fasta_file, 'fasta'):
        count += 1
    return count


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
    preprocessing(pretrain_path=args.data_path + args.pretrain_path, train_path=args.data_path + args.train_path, test_path=args.data_path + args.test_path, price_path=args.data_path + args.price_path, ensemble_path=args.data_path + args.ensemble_path)