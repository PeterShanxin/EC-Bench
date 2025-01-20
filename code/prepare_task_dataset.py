import pandas as pd
import sys,os
sys.path.append(os.getcwd())
from ECRECer.tools import filetool as ftool
from ECRECer.tools import funclib
from pandarallel import pandarallel
from ECRECer.tools import embedding_esm as esmebd


pandarallel.initialize(progress_bar=True)

# read train and test datasets for each cluster
def read_data(cluster_path, price_path):
    train = pd.read_csv(cluster_path + '/train_ec.csv', header=0)
    test = pd.read_csv(cluster_path + '/test_ec.csv', header=0)
    price = pd.read_csv(price_path, header=0)
    return train, test, price

# prepare data for EC number prediction task; no changes needed just save the data in save_path
def prepare_task3_data(cluster_path, price_path, save_path):    
    train, test, price = read_data(cluster_path, price_path)
    train.to_csv(save_path + '/train_task3.csv', index=False)
    test.to_csv(save_path + '/test_task3.csv', index=False)
    price.to_csv(save_path + '/price_task3.csv', index=False)
    train.to_feather(save_path + '/train_task3.feather')
    test.to_feather(save_path + '/test_task3.feather')
    price.to_feather(save_path + '/price_task3.feather')

    funclib.table2fasta(table=train[['id', 'seq']], file_out=save_path + '/train_task3.fasta')
    funclib.table2fasta(table=test[['id', 'seq']], file_out=save_path + '/test_task3.fasta')
    funclib.table2fasta(table=price[['id', 'seq']], file_out=save_path + '/price_task3.fasta')

# ESM embedding for each sequence
def esm_embedding():
    # create featureBank folder if not exists
    ftool.create_folder('ECRECer/data/featureBank')
    # load snp data
    sn_train = pd.read_feather('data/snp_train_data.feather')
    sn_test = pd.read_feather('data/snp_test_data.feather')
    sn_price = pd.read_feather('data/snp_price.feather')

    full_snp_data = pd.concat([sn_train, sn_test, sn_price], axis=0)
    full_snap_data = full_snp_data.sort_values(by=['id', 'date_annotation_update'], ascending=False)
    full_snap_data = full_snap_data[['id', 'seq']].drop_duplicates(subset='id', keep='first')
    full_snap_data.reset_index(drop=True, inplace=True)
    
    needesm = full_snap_data

    if len(needesm)>0 and not os.path.exists('ECRECer/data/featureBank/embd_esm0.feather'):
        tr_rep0, tr_rep32, tr_rep33 = esmebd.get_rep_multi_sequence(sequences=needesm, model='esm1b_t33_650M_UR50S',seqthres=1022)    
        tr_rep0.to_feather('ECRECer/data/featureBank/embd_esm0.feather')
        tr_rep32.to_feather('ECRECer/data/featureBank/embd_esm32.feather')
        tr_rep33.to_feather('ECRECer/data/featureBank/embd_esm33.feather')
    

#main function
def main():
    # read cluster paths
    cluster_paths = ['data/cluster-30', 'data/cluster-50', 'data/cluster-70', 'data/cluster-90', 'data/cluster-100']
    # read price path
    price_path = 'data/price-149.csv'
    # create save paths
    save_paths = ['ECRECer/data/datasets/task3']
    # prepare data for each cluster
    for save_path in save_paths:
        for cluster_path in cluster_paths: 
            cluster_name = cluster_path.split('/')[-1]
            print('Preparing data for cluster ', cluster_name)
            # create save path if not exists
            ftool.create_folder(save_path+'/'+cluster_name)
            prepare_task3_data(cluster_path, price_path, save_path+'/'+cluster_name)

if __name__ == "__main__":
    esm_embedding()
    main()



 