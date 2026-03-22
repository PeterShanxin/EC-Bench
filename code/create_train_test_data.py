import pandas as pd
import json
from Bio import SeqIO
import argparse
import ahocorasick
import psutil
import os


def _get_sequence_column(df, dataset_name):
    if 'sequence' in df.columns:
        return 'sequence'
    if 'seq' in df.columns:
        return 'seq'
    raise KeyError(f"{dataset_name} is missing a sequence column; expected one of: sequence, seq")


def monitor_usage():
    # Monitor RAM usage
    used_memory = psutil.virtual_memory()
    print('memory info: ', used_memory)

    disk_usage = psutil.disk_usage('/')
    print(f"Disk info: {disk_usage}")

def create_clusters(data_path, cluster_path_100, cluster_path_90, cluster_path_70, cluster_path_50, cluster_path_30):
    # read cluster files
    cluster_100 = pd.read_csv(cluster_path_100, sep='\t', header=None)  
    cluster_100.columns = ['representative', 'member']
    cluster_100 = cluster_100.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_100.to_csv(os.path.join(data_path, 'cluster-100/clusterEns_cluster_final.tsv'), sep='\t', index=False, header=False)

    cluster_90 = pd.read_csv(cluster_path_90, sep='\t', header=None)
    cluster_90.columns = ['representative', 'member']
    cluster_90 = cluster_90.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_90.to_csv(os.path.join(data_path, 'cluster-90/clusterEns_cluster_final.tsv'), sep='\t', index=False, header=False)

    cluster_70 = pd.read_csv(cluster_path_70, sep='\t', header=None)
    cluster_70.columns = ['representative', 'member']
    cluster_70 = cluster_70.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_70.to_csv(os.path.join(data_path, 'cluster-70/clusterEns_cluster_final.tsv'), sep='\t', index=False, header=False)

    cluster_50 = pd.read_csv(cluster_path_50, sep='\t', header=None)
    cluster_50.columns = ['representative', 'member']
    cluster_50 = cluster_50.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_50.to_csv(os.path.join(data_path, 'cluster-50/clusterEns_cluster_final.tsv'), sep='\t', index=False, header=False)

    cluster_30 = pd.read_csv(cluster_path_30, sep='\t', header=None)
    cluster_30.columns = ['representative', 'member']
    cluster_30 = cluster_30.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_30.to_csv(os.path.join(data_path, 'cluster-30/clusterEns_cluster_final.tsv'), sep='\t', index=False, header=False)

             
# Create fine-tuning data
'''
1. remove similar sequences from train data based on the clustering result
2. Add 3d information for train and test data
'''
# Create pretrain data
'''
remove EC numbers from pretrain data for the sequences that are similar to the sequences in test data based on the clustering result
'''

def create_data(data_path, train_ec_path, test_ec_path, train_3d_path, test_3d_path, info_file_path, price_file_path, ensemble_file_path, t):
    train = pd.read_csv(train_ec_path)
    test = pd.read_csv(test_ec_path)
    price = pd.read_csv(price_file_path)
    train_seq_col = _get_sequence_column(train, 'train')
    test_seq_col = _get_sequence_column(test, 'test')
    price_seq_col = _get_sequence_column(price, 'price')
    if ensemble_file_path:
        ensemble = pd.read_csv(ensemble_file_path)
        ensemble_seq_col = _get_sequence_column(ensemble, 'ensemble')
    test_ids = list(test['id'])
    price_ids = list(price['id'])
    test_ids.extend(price_ids)
    train_3d_id = [record.id for record in SeqIO.parse(train_3d_path, 'fasta')]
    test_3d_id = [record.id for record in SeqIO.parse(test_3d_path, 'fasta')]
    with open(info_file_path, 'r') as f:
            info = json.load(f)

    if t == 100:
        cluster_paths = [os.path.join(data_path, 'cluster-100')]
        t_list = [1.0]
    elif t == 90:
        cluster_paths = [os.path.join(data_path, 'cluster-90'), os.path.join(data_path, 'cluster-100')]
        t_list = [0.9, 1.0]
    elif t == 70:
        cluster_paths = [os.path.join(data_path, 'cluster-70'), os.path.join(data_path, 'cluster-90'), os.path.join(data_path, 'cluster-100')]
        t_list = [0.7, 0.9, 1.0]
    elif t == 50:
        cluster_paths = [os.path.join(data_path, 'cluster-50'), os.path.join(data_path, 'cluster-70'), os.path.join(data_path, 'cluster-90'), os.path.join(data_path, 'cluster-100')]
        t_list = [0.5, 0.7, 0.9, 1.0]
    elif t == 30:
        cluster_paths = [os.path.join(data_path, 'cluster-30'), os.path.join(data_path, 'cluster-50'), os.path.join(data_path, 'cluster-70'), os.path.join(data_path, 'cluster-90'), os.path.join(data_path, 'cluster-100')]
        t_list = [0.3, 0.5, 0.7, 0.9, 1.0]
    
    # A dictionary to store threshold and its corresponding path
    threshold_paths = dict(zip(t_list, cluster_paths))
    
    # make ids_to_remove global list
    ids_to_remove = []
    
    auto = ahocorasick.Automaton()
    for substr in test_ids:
        auto.add_word(substr, substr)

    # Process clusters for each threshold
    for threshold in t_list:
        path = threshold_paths[threshold]
        clustering_path = f'{path}/clusterEns_cluster_final.tsv'
        clusters = pd.read_csv(clustering_path, sep='\t', header=None)
        auto.make_automaton() 
        for i in range(clusters.shape[0]):
            cluster_list = clusters.iloc[i, 1].split(',')
            for astr in cluster_list:
                if auto.exists(astr):
                    ids_to_remove.extend(cluster_list)
                    for substr in cluster_list:
                        auto.add_word(substr, substr)
                    break
        ids_to_remove = list(set(ids_to_remove))
        preview = ids_to_remove[:10]
        print(f'ids to remove after threshold {threshold}: {len(ids_to_remove)} (preview: {preview})')
        del clusters
    
    # save the ids_to_remove in a tsv file
    ids_to_remove_path = os.path.join(data_path, f'cluster-{t}/ids_to_remove.txt')
    with open(ids_to_remove_path, 'w') as f:
        for id in ids_to_remove:
            f.write(f'{id}\n')

    path = os.path.join(data_path, f'cluster-{t}')

    # keep only the train sequences that have 3d information and remove the sequences that are similar to the test sequences
    train = train[~train['id'].isin(ids_to_remove)]
    train.reset_index(drop=True, inplace=True)
    train = train[train['id'].isin(train_3d_id)]
    train.reset_index(drop=True, inplace=True)
    train_path = path + '/train_ec.csv'
    train.to_csv(train_path, index=False)
    print('train size: ', train.shape)

    # keep only the test sequences that have 3d information
    test= test[test['id'].isin(test_3d_id)]
    test.reset_index(drop=True, inplace=True)
    test_path = path + '/test_ec.csv'
    test.to_csv(test_path, index=False)
    print('test size: ', test.shape)
    
    # make fasta file from test in data_path named test_ec.fasta
    test_fasta_path = os.path.join(data_path, f'test_ec.fasta')
    with open(test_fasta_path, 'w') as f:
        for index, row in test.iterrows():
            f.write(f'>{row["id"]}\n{row[test_seq_col]}\n')
    
    # make fasta file from price in data_path named price-149.fasta
    price_fasta_path = os.path.join(data_path, f'price-149.fasta')
    with open(price_fasta_path, 'w') as f:
        for index, row in price.iterrows():
            f.write(f'>{row["id"]}\n{row[price_seq_col]}\n')

    if ensemble_file_path:
        # remove the sequences that are similar to the test sequences
        ensemble = ensemble[~ensemble['id'].isin(ids_to_remove)]
        ensemble.reset_index(drop=True, inplace=True)
        ensemble_path = path + '/ensemble_ec.csv'
        ensemble.to_csv(ensemble_path, index=False)
        print('ensemble size: ', ensemble.shape)
        # make fasta file from ensemble in data_path named ens_{t}.fasta
        ensemble_fasta_path = os.path.join(data_path, f'ens-{t}.fasta')
        with open(ensemble_fasta_path, 'w') as f:
            for index, row in ensemble.iterrows():
                f.write(f'>{row["id"]}\n{row[ensemble_seq_col]}\n')

    # Step 2 - save train and test data with 3d information
    train['3d_info'] = train['id'].map(info)
    test['3d_info'] = test['id'].map(info)
    
    train_path = path + '/train_ec_3d.csv'
    train.to_csv(train_path, index=False)
    print('train size: ', train.shape)
    
    test_path = path + '/test_ec_3d.csv'
    test.to_csv(test_path, index=False)
    print('test size: ', test.shape)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data creation')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data')
    parser.add_argument('--train_name', type=str, default='train.csv', help='Name of train data')
    parser.add_argument('--test_name', type=str, default='test.csv', help='Name of test data')
    parser.add_argument('--price_file_name', type=str, default='price-149.csv', help='Name of price file')
    parser.add_argument('--ensemble_file_name', type=str, default=None, help='Name of ensemble file')
    parser.add_argument('--train_3d_name', type=str, default='train_having_3d.fasta', help='Name of train 3d data')
    parser.add_argument('--test_3d_name', type=str, default='test_having_3d.fasta', help='Name of test 3d data')
    parser.add_argument('--info_file_name', type=str, default='swissprot_coordinates.json', help='Name of all 3d coordinates file')
    parser.add_argument('--threshod', dest='threshold', type=int, default=30)
    parser.add_argument('--threshold', dest='threshold', type=int, help='Alias for --threshod')
    args = parser.parse_args()

    create_clusters(cluster_path_100=os.path.join(args.data_path, 'cluster-100/clusterEns_cluster.tsv'), cluster_path_90=os.path.join(args.data_path, 'cluster-90/clusterEns_cluster.tsv'), cluster_path_70=os.path.join(args.data_path, 'cluster-70/clusterEns_cluster.tsv'), cluster_path_50=os.path.join(args.data_path, 'cluster-50/clusterEns_cluster.tsv'), cluster_path_30=os.path.join(args.data_path, 'cluster-30/clusterEns_cluster.tsv'), data_path=args.data_path)
    create_data(data_path=args.data_path, train_ec_path=os.path.join(args.data_path, args.train_name), test_ec_path=os.path.join(args.data_path, args.test_name), train_3d_path=os.path.join(args.data_path, args.train_3d_name), test_3d_path=os.path.join(args.data_path, args.test_3d_name), info_file_path=os.path.join(args.data_path, args.info_file_name), price_file_path=os.path.join(args.data_path, args.price_file_name), ensemble_file_path=os.path.join(args.data_path, args.ensemble_file_name), t=args.threshold)
