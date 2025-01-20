import pandas as pd
import json
from Bio import SeqIO
import argparse
import ahocorasick
import psutil

def monitor_usage():
    # Monitor RAM usage
    used_memory = psutil.virtual_memory()
    print('memory info: ', used_memory)

    disk_usage = psutil.disk_usage('/')
    print(f"Disk info: {disk_usage}")

def create_clusters(cluster_path_100, cluster_path_90, cluster_path_70, cluster_path_50, cluster_path_30):
    # read cluster files
    cluster_100 = pd.read_csv(cluster_path_100, sep='\t', header=None)  
    cluster_100.columns = ['representative', 'member']
    cluster_100 = cluster_100.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_100.to_csv('data/cluster-100/clusterEns_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_90 = pd.read_csv(cluster_path_90, sep='\t', header=None)
    cluster_90.columns = ['representative', 'member']
    cluster_90 = cluster_90.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_90.to_csv('data/cluster-90/clusterEns_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_70 = pd.read_csv(cluster_path_70, sep='\t', header=None)
    cluster_70.columns = ['representative', 'member']
    cluster_70 = cluster_70.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_70.to_csv('data/cluster-70/clusterEns_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_50 = pd.read_csv(cluster_path_50, sep='\t', header=None)
    cluster_50.columns = ['representative', 'member']
    cluster_50 = cluster_50.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_50.to_csv('data/cluster-50/clusterEns_cluster_final.tsv', sep='\t', index=False, header=False)

    cluster_30 = pd.read_csv(cluster_path_30, sep='\t', header=None)
    cluster_30.columns = ['representative', 'member']
    cluster_30 = cluster_30.groupby('representative')['member'].apply(lambda x: ','.join(x)).reset_index()
    cluster_30.to_csv('data/cluster-30/clusterEns_cluster_final.tsv', sep='\t', index=False, header=False)

             
# Create fine-tuning data
'''
1. remove similar sequences from train data based on the clustering result
2. Add 3d information for train and test data
'''
# Create pretrain data
'''
remove EC numbers from pretrain data for the sequences that are similar to the sequences in test data based on the clustering result
'''

def create_data(pretrain_ec_path, train_ec_path, test_ec_path, train_3d_path, test_3d_path, info_file_path, price_file_path, t):
    train = pd.read_feather(train_ec_path)
    test = pd.read_feather(test_ec_path)
    price = pd.read_csv(price_file_path)
    test_ids = list(test['id'])
    price_ids = list(price['id'])
    test_ids.extend(price_ids)
    pretrain = pd.read_feather(pretrain_ec_path)
    train_3d_id = [record.id for record in SeqIO.parse(train_3d_path, 'fasta')]
    test_3d_id = [record.id for record in SeqIO.parse(test_3d_path, 'fasta')]
    with open(info_file_path, 'r') as f:
            info = json.load(f)

    if t == 100:
        cluster_paths = ['data/cluster-100']
        t_list = [1.0]
    elif t == 90:
        cluster_paths = ['data/cluster-90', 'data/cluster-100']
        t_list = [0.9, 1.0]
    elif t == 70:
        cluster_paths = ['data/cluster-70', 'data/cluster-90', 'data/cluster-100']
        t_list = [0.7, 0.9, 1.0]
    elif t == 50:
        cluster_paths = ['data/cluster-50', 'data/cluster-70', 'data/cluster-90', 'data/cluster-100']
        t_list = [0.5, 0.7, 0.9, 1.0]
    elif t == 30:
        cluster_paths = ['data/cluster-30', 'data/cluster-50', 'data/cluster-70', 'data/cluster-90', 'data/cluster-100']
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
        print('ids to remove: ', ids_to_remove)
        del clusters
    path = 'data/cluster-' + str(t)
    train = train[~train['id'].isin(ids_to_remove)]
    train.reset_index(drop=True, inplace=True)
    train = train[train['id'].isin(train_3d_id)]
    train.reset_index(drop=True, inplace=True)
    test= test[test['id'].isin(test_3d_id)]
    test.reset_index(drop=True, inplace=True)
    
    # Step 2
    train_info_list = []
    test_info_list = []
    
    for i in range(train.shape[0]):
        if train.iloc[i, 0] in info.keys():
            train_info_list.append(info[train.iloc[i, 0]])
    for i in range(test.shape[0]):
        if test.iloc[i, 0] in info.keys():
            test_info_list.append(info[test.iloc[i, 0]])

    train['3d_info'] = train_info_list
    test['3d_info'] = test_info_list
    train_path = path + '/train_ec_3d.csv'
    train.to_csv(train_path, index=False)
    print('train size: ', train.shape)
    test_path = path + '/test_ec_3d.csv'
    test.to_csv(test_path, index=False)
    print('test size: ', test.shape)
    
    # remove 3d_info column from train and test data and save them in csv format
    train_no_3d = train.drop(columns=['3d_info'])
    train_no_3d_path = path + '/train_ec.csv'
    train_no_3d.to_csv(train_no_3d_path, index=False)
    test_no_3d = test.drop(columns=['3d_info'])
    test_no_3d_path = path + '/test_ec.csv'
    test_no_3d.to_csv(test_no_3d_path, index=False)

    # Save the unique EC numbers with their numerical representation in a csv file
    all_ecs = []
    for i in range(train.shape[0]):
        all_ecs.extend(train['ec_number'][i].split(','))
    print('ec number of train: ', len(list(set(all_ecs))))
    unique_ecs = list(set(all_ecs))
    unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
    unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
    unique_ecs_path = path + '/unique_ecs_train.csv'
    unique_ecs.to_csv(unique_ecs_path, index=False)

    all_ecs = []
    for i in range(test.shape[0]):
        all_ecs.extend(test['ec_number'][i].split(','))
    print('ec number of test: ', len(list(set(all_ecs))))
    unique_ecs = list(set(all_ecs))
    unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
    unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
    unique_ecs_path = path + '/unique_ecs_test.csv'
    unique_ecs.to_csv(unique_ecs_path, index=False)

    all_ecs = []
    for i in range(price.shape[0]):
        all_ecs.extend(price['ec_number'][i].split(','))
    print('ec number of price: ', len(list(set(all_ecs))))
    unique_ecs = list(set(all_ecs))
    unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
    unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
    unique_ecs_path = path + '/unique_ecs_price.csv'
    unique_ecs.to_csv(unique_ecs_path, index=False)

    del train_info_list, test_info_list

    # Use boolean indexing to update the 'ec_number' column where the condition is met
    pretrain = pretrain[~pretrain['id'].isin(ids_to_remove)]
    pretrain.reset_index(drop=True, inplace=True)
    pretrain_path = path + '/pretrain_ec.csv'
    pretrain.to_csv(pretrain_path, index=False)
    # print number of unique EC numbers in pretrain data
    all_ecs = []
    for i in range(pretrain.shape[0]):
        all_ecs.extend(pretrain['ec_number'][i].split(','))
    print('ec number of pretrain: ', len(list(set(all_ecs))))
    # Save the unique EC numbers with their numerical representation in a csv file
    unique_ecs = list(set(all_ecs))
    unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
    unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
    unique_ecs_path = path + '/unique_ecs_pretrain.csv'
    unique_ecs.to_csv(unique_ecs_path, index=False)
  
def create_ens_data(t):
    if t == 100:
        cluster_paths = ['data/cluster-100']
        t_list = [1.0]
        path = 'data/cluster-100'  
    elif t == 90:
        cluster_paths = ['data/cluster-90', 'data/cluster-100']
        t_list = [0.9, 1.0]
        path = 'data/cluster-90'  
    elif t == 70:
        cluster_paths = ['data/cluster-70', 'data/cluster-90', 'data/cluster-100']
        t_list = [0.7, 0.9, 1.0]
        path = 'data/cluster-70'  
    elif t == 50:
        cluster_paths = ['data/cluster-50', 'data/cluster-70', 'data/cluster-90', 'data/cluster-100']
        t_list = [0.5, 0.7, 0.9, 1.0]
        path = 'data/cluster-50'  
    elif t == 30:
        cluster_paths = ['data/cluster-30', 'data/cluster-50', 'data/cluster-70', 'data/cluster-90', 'data/cluster-100']
        t_list = [0.3, 0.5, 0.7, 0.9, 1.0]
        path = 'data/cluster-30'

    train = pd.read_csv(path + '/train_task3.csv')
    valid = pd.read_csv(path +'/valid_task3.csv')
    test = pd.read_csv('data/test_task3.csv')
    price = pd.read_csv('data/price.csv')
    #  https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2024_03/knowledgebase/uniprot_sprot-only2024_03.tar.gz
    # gunzip uniprot_sprot-only2024_03.tar.gz
    # tar -xvf uniprot_sprot-only2024_03.tar

    ens = pd.read_csv('data/uniprot_sprot_2024_03.csv') # cols : id,seq,ec_number
    test_ids = list(test['id'])
    price_ids = list(price['id'])
    test_ids.extend(price_ids)  

    # A dictionary to store threshold and its corresponding path
    threshold_paths = dict(zip(t_list, cluster_paths))
    # make ids_to_remove global list
    ids_to_remove = []
    auto = ahocorasick.Automaton()
    for substr in test_ids:
        auto.add_word(substr, substr)

    # Process clusters for each threshold
    for threshold in t_list:
        pathh = threshold_paths[threshold]
        clustering_path = f'{pathh}/clusterEns_cluster_final.tsv'
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
        print('ids to remove: ', ids_to_remove)
        del clusters
    
    ens = ens[~ens['id'].isin(test_ids)]
    ens = ens[~ens['id'].isin(train['id'].tolist())]
    ens = ens[~ens['id'].isin(valid['id'].tolist())]
    ens = ens[~ens['id'].isin(ids_to_remove)]
    # Shuffle the DataFrame
    ens = ens.sample(frac=1)
    # Reset the index
    ens.reset_index(drop=True, inplace=True)
    pretrain_path = path + '/ens.csv'
    ens.to_csv(pretrain_path, index=False)
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning data creation')
    parser.add_argument('--pretrain_ec_path', type=str, default='data/pretrain_ec.feather', help='Path to pretrain ec data')
    parser.add_argument('--train_ec_path', type=str, default='data/train_ec.feather', help='Path to train ec data')
    parser.add_argument('--test_ec_path', type=str, default='data/test_ec.feather', help='Path to test ec data')
    parser.add_argument('--train_3d_path', type=str, default='data/train_having_3d.fasta', help='Path to train 3d data')
    parser.add_argument('--test_3d_path', type=str, default='data/test_having_3d.fasta', help='Path to test 3d data')
    parser.add_argument('--info_file_path', type=str, default='data/swissprot_coordinates.json', help='Path to all 3d coordinates file')
    parser.add_argument('--price_file_path', type=str, default='data/price-149.csv', help='Path to price file')
    parser.add_argument('--ens_threshod', type=int, default=30)
    args = parser.parse_args()

    create_clusters(cluster_path_100='data/cluster-100/clusterEns_cluster.tsv', cluster_path_90='data/cluster-90/clusterEns_cluster.tsv', cluster_path_70='data/cluster-70/clusterEns_cluster.tsv', cluster_path_50='data/cluster-50/clusterEns_cluster.tsv', cluster_path_30='data/cluster-30/clusterEns_cluster.tsv')
    create_data(pretrain_ec_path=args.pretrain_ec_path, train_ec_path=args.train_ec_path, test_ec_path=args.test_ec_path, train_3d_path=args.train_3d_path, test_3d_path=args.test_3d_path, info_file_path=args.info_file_path, price_file_path=args.price_file_path, t=argparse.ens_threshod)
    create_ens_data(args.ens_threshod)