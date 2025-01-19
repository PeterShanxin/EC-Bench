import time
import numpy as np
import pandas as pd
import sys
import os

import psutil
sys.path.append(os.getcwd())
from ECRECer.tools import funclib
import ECRECer.benchmark_common as bcommon
import ECRECer.benchmark_train as btrain
import ECRECer.benchmark_test as btest
import ECRECer.config as cfg
from memory_profiler import memory_usage
import shutil

from pandarallel import pandarallel #  import pandaralle
pandarallel.initialize() # init

if not os.path.exists(cfg.RESULTSDIR+cfg.DIR_CLUSTER):
    os.makedirs(cfg.RESULTSDIR+cfg.DIR_CLUSTER)

if not os.path.exists(cfg.MODELDIR):
    os.makedirs(cfg.MODELDIR)
    
#read train test data for each cluster
def read_data(cluster_path):
    train = pd.read_csv(cluster_path + '/train_task3.csv')
    test = pd.read_csv(cluster_path + '/test_task3.csv')
    price = pd.read_csv(cluster_path + '/price_task3.csv')
    price_149 = pd.read_csv(cluster_path + '/price-149.csv')
    val_ensemble = pd.read_csv(cluster_path + '/ens.csv')
    val_ensemble['ec_number'] = val_ensemble['ec_number'].astype(str)
    return train, test, price, price_149, val_ensemble

#res32
esm_32 = pd.read_feather(cfg.FILE_FEATURE_ESM32)

cluster_path =  cfg.DIR_DATASETS + 'task3/' + cfg.DIR_CLUSTER
train, test, price, price_149, val_ensemble = read_data(cluster_path)

# montior storage, run time and memory usage
start_time = time.time()
total, used, free = shutil.disk_usage("/")
print(f"Total before Blastp: {total // (2**30)} GB")
print(f"Train Blastp Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")
funclib.getblast(train)
end_time = time.time()
runtime = end_time - start_time
print(f"Train Blastp Runtime: {runtime} seconds")
total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (2**30)} GB")
print(f"Train Blastp Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")
# Get the memory usage in MB
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
print(f"Train Blastp Memory usage: {memory_usage} MB")

# res_data_train = funclib.getblast_usedb(cfg.FILE_BLAST_TRAIN_DB, train)
# res_data_train = res_data_train[['id', 'sseqid']].merge(train, left_on='sseqid',right_on='id', how='left')[['id_x','sseqid','ec_number']]
# res_data_train = res_data_train.rename(columns={'id_x':'id','sseqid':'id_ref', 'ec_number':'ec_number_pred'})
# res_data_train = res_data_train.merge(train, on='id', how='left')[['id','ec_number','ec_number_pred']]
# res_data_train = res_data_train.drop_duplicates(subset='id')

start_time = time.time()
res_data = funclib.getblast_usedb(cfg.FILE_BLAST_TRAIN_DB, test)
res_data = res_data[['id', 'sseqid']].merge(train, left_on='sseqid',right_on='id', how='left')[['id_x','sseqid','ec_number']]
res_data = res_data.rename(columns={'id_x':'id','sseqid':'id_ref', 'ec_number':'ec_number_pred'})
res_data = res_data.merge(test, on='id', how='left')[['id','ec_number','ec_number_pred']]
end_time = time.time()
runtime = end_time - start_time

print(f"Test Blastp Runtime: {runtime} seconds")

total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (2**30)} GB")
print(f"Test Blastp Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")
# Get the memory usage in MB
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB

print(f"Test Blastp Memory usage: {memory_usage} MB")

# res_data_price = funclib.getblast_usedb(cfg.FILE_BLAST_TRAIN_DB, price)
# res_data_price = res_data_price[['id', 'sseqid']].merge(train, left_on='sseqid',right_on='id', how='left')[['id_x','sseqid','ec_number']]
# res_data_price = res_data_price.rename(columns={'id_x':'id','sseqid':'id_ref', 'ec_number':'ec_number_pred'})
# res_data_price = res_data_price.merge(price, on='id', how='left')[['id','ec_number','ec_number_pred']]

# res_data_price_149 = funclib.getblast_usedb(cfg.FILE_BLAST_TRAIN_DB, price_149)
# res_data_price_149 = res_data_price_149[['id', 'sseqid']].merge(train, left_on='sseqid',right_on='id', how='left')[['id_x','sseqid','ec_number']]
# res_data_price_149 = res_data_price_149.rename(columns={'id_x':'id','sseqid':'id_ref', 'ec_number':'ec_number_pred'})
# res_data_price_149 = res_data_price_149.merge(price_149, on='id', how='left')[['id','ec_number','ec_number_pred']]

# res_data_ens = funclib.getblast_usedb(cfg.FILE_BLAST_TRAIN_DB, val_ensemble)
# res_data_ens = res_data_ens[['id', 'sseqid']].merge(train, left_on='sseqid',right_on='id', how='left')[['id_x','sseqid','ec_number']]
# res_data_ens = res_data_ens.rename(columns={'id_x':'id','sseqid':'id_ref', 'ec_number':'ec_number_pred'})
# res_data_ens = res_data_ens.merge(val_ensemble, on='id', how='left')[['id','ec_number','ec_number_pred']]

# res_data_train = res_data_train.drop_duplicates(subset='id')
# res_data = res_data.drop_duplicates(subset='id')
# res_data_price = res_data_price.drop_duplicates(subset='id')
# res_data_price_149 = res_data_price_149.drop_duplicates(subset='id')
# res_data_ens = res_data_ens.drop_duplicates(subset='id')

# # BLASTP
# train_p = train.merge(res_data_train[['id','ec_number_pred']], on='id', how='left')
# test_p = test.merge(res_data[['id','ec_number_pred']], on='id', how='left')
# price_p = price.merge(res_data_price[['id','ec_number_pred']], on='id', how='left')
# price_149_p = price_149.merge(res_data_price_149[['id','ec_number_pred']], on='id', how='left')
# val_ensemble_p = val_ensemble.merge(res_data_ens[['id','ec_number_pred']], on='id', how='left')

# train_p.to_csv(cfg.RESULTSDIR+cfg.DIR_CLUSTER+'/train_blastp.csv', index=None)
# test_p.to_csv(cfg.RESULTSDIR+cfg.DIR_CLUSTER+'/test_blastp.csv', index=None)
# price_p.to_csv(cfg.RESULTSDIR+cfg.DIR_CLUSTER+'/price_blastp.csv', index=None)
# price_149_p.to_csv(cfg.RESULTSDIR+cfg.DIR_CLUSTER+'/price_149_blastp.csv', index=None)
# val_ensemble_p.to_csv(cfg.RESULTSDIR+cfg.DIR_CLUSTER+'/price_149_blastp.csv', index=None)

train_set= funclib.split_ecdf_to_single_lines(train)
test_set = funclib.split_ecdf_to_single_lines(test)
price_set = funclib.split_ecdf_to_single_lines(price)
price_149_set = funclib.split_ecdf_to_single_lines(price_149)
#val_ensemble_set = funclib.split_ecdf_to_single_lines(val_ensemble)

print('loading ec to label dict')
dict_ec_label = btrain.make_ec_label(train_label=train_set['ec_number'], 
                                            test_label=test_set['ec_number'], 
                                            price_label=price_set['ec_number'],
                                            price_149_label=price_149_set['ec_number'],
                                            #ens_label=val_ensemble_set['ec_number'],
                                            file_save= cfg.FILE_EC_LABEL_DICT, 
                                            force_model_update=cfg.UPDATE_MODEL)

train_set['ec_label'] = train_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))
test_set['ec_label'] = test_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))
price_set['ec_label'] = price_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))
price_149_set['ec_label'] = price_149_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))
#val_ensemble_set['ec_label'] = val_ensemble_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))

train_set.ec_label.astype('int')
test_set.ec_label.astype('int')
price_set.ec_label.astype('int')
price_149_set.ec_label.astype('int')
#val_ensemble_set.ec_label.astype('int')

train_set2 = train_set.copy()
test_set2 = test_set.copy()
price_set2 = price_set.copy()
price_149_set2 = price_149_set.copy()
#val_ensemble_set2 = val_ensemble_set.copy()

train_set2 = train_set2.merge(esm_32, on='id', how='left')
test_set2 = test_set2.merge(esm_32, on='id', how='left')
price_set2 = price_set2.merge(esm_32, on='id', how='left')
price_149_set2 = price_149_set2.merge(esm_32, on='id', how='left')
#val_ensemble_set2 = val_ensemble_set2.merge(esm_32, on='id', how='left')

train_set = train_set.merge(esm_32, on='id', how='left')
test_set = test_set.merge(esm_32, on='id', how='left')
price_set = price_set.merge(esm_32, on='id', how='left')
price_149_set = price_149_set.merge(esm_32, on='id', how='left')
#val_ensemble_set = val_ensemble_set.merge(esm_32, on='id', how='left')

train_X = train_set2.iloc[:, 4:]
train_Y = pd.DataFrame(train_set2['ec_label'])

test_X = test_set2.iloc[:, 4:]
test_Y = pd.DataFrame(test_set2['ec_label'])

price_X = price_set2.iloc[:, 4:]
price_Y = pd.DataFrame(price_set2['ec_label'])

price_149_X = price_149_set2.iloc[:, 4:]
price_149_Y = pd.DataFrame(price_149_set2['ec_label'])

# val_ensemble_X = val_ensemble_set2.iloc[:, 4:]
# val_ensemble_Y = pd.DataFrame(val_ensemble_set2['ec_label'])

#train
bcommon.prepare_slice_file(x_data=train_X, y_data=train_Y, x_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_train_x_esm32.txt'), y_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_train_y_esm32.txt'), ec_label_dict=dict_ec_label)
#test
bcommon.prepare_slice_file(x_data=test_X, y_data=test_Y, x_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_test_x_esm32.txt'), y_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_test_y_esm32.txt'), ec_label_dict=dict_ec_label)
#price
bcommon.prepare_slice_file(x_data=price_X, y_data=price_Y, x_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_price_x_esm32.txt'), y_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_price_y_esm32.txt'), ec_label_dict=dict_ec_label)
#price-149
bcommon.prepare_slice_file(x_data=price_149_X, y_data=price_149_Y, x_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_price_149_x_esm32.txt'), y_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_price_149_y_esm32.txt'), ec_label_dict=dict_ec_label)
#val_ensemble
# bcommon.prepare_slice_file(x_data=val_ensemble_X, y_data=val_ensemble_Y, x_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_val_ensemble_x_esm32.txt'), y_file=os.path.join('ECRECer/data/datasets/task3', cfg.DIR_CLUSTER, 'slice_val_ensemble_y_esm32.txt'), ec_label_dict=dict_ec_label)

print('step 6 trainning slice model')
if not os.path.exists(cfg.MODELDIR+'slice_esm32'):
    os.makedirs(cfg.MODELDIR+'slice_esm32')

# montior storage, run time and memory usage
start_time = time.time()
btrain.train_ec_slice(trainX=cfg.ESM_DIR+'slice_train_x_esm32.txt', trainY=cfg.ESM_DIR+'slice_train_y_esm32.txt', 
                        modelPath=cfg.MODELDIR+'slice_esm32')
end_time = time.time()
runtime = end_time - start_time

print(f"Train Runtime: {runtime} seconds")

total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (2**30)} GB")
print(f"Train Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")
# Get the memory usage in MB
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
print(f"Train Memory usage: {memory_usage} MB")

start_time = time.time()
slice_pred_test, test_probs = btest.get_slice_res(slice_query_file=cfg.ESM_DIR+'slice_test_x_esm32.txt', 
                                    model_path=cfg.MODELDIR+'slice_esm32', 
                                    dict_ec_label=dict_ec_label, 
                                    test_set=test_set, 
                                    res_file=os.path.join(cfg.ROOTDIR, 'tmp/test.txt'))
end_time = time.time()
runtime = end_time - start_time

print(f"Test Runtime: {runtime} seconds")

total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (2**30)} GB")
print(f"Test Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")
# Get the memory usage in MB
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
print(f"Test Memory usage: {memory_usage} MB")

slice_pred_train, train_probs = btest.get_slice_res(slice_query_file=cfg.ESM_DIR+'slice_train_x_esm32.txt', 
                                    model_path=cfg.MODELDIR+'slice_esm32', 
                                    dict_ec_label=dict_ec_label, 
                                    test_set=train_set, 
                                    res_file=os.path.join(cfg.ROOTDIR, 'tmp/train.txt'))
slice_pred_price, price_probs = btest.get_slice_res(slice_query_file=cfg.ESM_DIR+'slice_price_x_esm32.txt',
                                    model_path=cfg.MODELDIR+'slice_esm32',
                                    dict_ec_label=dict_ec_label,
                                    test_set=price_set,
                                    res_file=os.path.join(cfg.ROOTDIR, 'tmp/price.txt'))
slice_pred_price_149, price_149_probs = btest.get_slice_res(slice_query_file=cfg.ESM_DIR+'slice_price_149_x_esm32.txt',
                                    model_path=cfg.MODELDIR+'slice_esm32',
                                    dict_ec_label=dict_ec_label,
                                    test_set=train_set,
                                    res_file=os.path.join(cfg.ROOTDIR, 'tmp/price_149.txt'))
# slice_pred_ens, ens_probs = btest.get_slice_res(slice_query_file=cfg.ESM_DIR+'slice_val_ensemble_x_esm32.txt',
#                                     model_path=cfg.MODELDIR+'slice_esm32',
#                                     dict_ec_label=dict_ec_label,
#                                     test_set=val_ensemble_set,
#                                     res_file=os.path.join(cfg.ROOTDIR, 'tmp/ens.txt'))

# keep first 11 columns of test_probs and price_probs
train_probs = train_set.iloc[:, np.r_[0:11]]
test_probs = test_probs.iloc[:, np.r_[0:11]]
price_probs = price_probs.iloc[:, np.r_[0:11]]
price_149_probs = price_149_probs.iloc[:, np.r_[0:11]]
# ens_probs = ens_probs.iloc[:, np.r_[0:11]]

s1res_train = train_set2.iloc[:,np.r_[0:5]].merge(slice_pred_train, on='id', how='left')
s1res_test = test_set2.iloc[:,np.r_[0:5]].merge(slice_pred_test, on='id', how='left')
s1res_price = price_set2.iloc[:,np.r_[0:5]].merge(slice_pred_price, on='id', how='left')
s1res_price_149 = price_149_set2.iloc[:,np.r_[0:5]].merge(slice_pred_price_149, on='id', how='left')
# s1res_ens = val_ensemble_set2.iloc[:,np.r_[0:5]].merge(slice_pred_ens, on='id', how='left')

s1res_train_prob = s1res_train.merge(train_probs, on='id', how='left')
s1res_test_prob = s1res_test.merge(test_probs, on='id', how='left')
s1res_price_prob = s1res_price.merge(price_probs, on='id', how='left')
s1res_price_149_prob = s1res_price_149.merge(price_149_probs, on='id', how='left')
# s1res_ens_prob = s1res_ens.merge(ens_probs, on='id', how='left')

# remove seq column from s1res_test_prob and s1res_price_prob
s1res_train_prob = s1res_train_prob.drop(columns=['seq'])
s1res_test_prob = s1res_test_prob.drop(columns=['seq'])
s1res_price_prob = s1res_price_prob.drop(columns=['seq'])
s1res_price_149_prob = s1res_price_149_prob.drop(columns=['seq'])
# s1res_ens_prob = s1res_ens_prob.drop(columns=['seq'])

if not os.path.exists(os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER)):
    os.makedirs(os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER))

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_train_probs.tsv')
s1res_train_prob.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_probs.tsv')
s1res_test_prob.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_price_probs.tsv')
s1res_price_prob.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_price_149_probs.tsv')
s1res_price_149_prob.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_ens_probs.tsv')
# s1res_ens_prob.to_csv(slres_path, sep='\t', index=None)

# keep columns id, seq, ec_number and top0 (rename top0 to predicted_ec_number) and save it to a csv file 
s1res_train = s1res_train[['id', 'ec_number', 'top0']].rename(columns={'top0':'predicted_ec_number'})
s1res_test = s1res_test[['id', 'ec_number', 'top0']].rename(columns={'top0':'predicted_ec_number'})
s1res_price = s1res_price[['id', 'ec_number', 'top0']].rename(columns={'top0':'predicted_ec_number'})
s1res_price_149 = s1res_price_149[['id', 'ec_number', 'top0']].rename(columns={'top0':'predicted_ec_number'})
# s1res_ens = s1res_ens[['id', 'ec_number', 'top0']].rename(columns={'top0':'predicted_ec_number'})

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_train.tsv')
s1res_train.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results.tsv')
s1res_test.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_price.tsv')
s1res_price.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_price_149.tsv')
s1res_price_149.to_csv(slres_path, sep='\t', index=None)

slres_path = os.path.join(cfg.RESULTSDIR, cfg.DIR_CLUSTER, 'ec_results_ens.tsv')
# s1res_ens.to_csv(slres_path, sep='\t', index=None)





