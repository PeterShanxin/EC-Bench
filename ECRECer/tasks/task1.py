import sys,os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import time
import datetime
from tqdm import tqdm
from functools import reduce
import joblib

from ECRECer.tools import funclib
import ECRECer.benchmark_train as btrain
import ECRECer.benchmark_test as btest

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import ECRECer.config as cfg

from pandarallel import pandarallel #  import pandaralle
pandarallel.initialize() # init

if not os.path.exists(cfg.RESULTSDIR+cfg.DIR_CLUSTER):
    os.makedirs(cfg.RESULTSDIR+cfg.DIR_CLUSTER)
    
if not os.path.exists(cfg.MODELDIR):
    os.makedirs(cfg.MODELDIR)

#read train test data for each cluster
def read_data(cluster_path):
    train = pd.read_csv(cluster_path + '/train_task1.csv')
    test = pd.read_csv(cluster_path + '/test_task1.csv')
    price = pd.read_csv(cluster_path + '/price_task1.csv')
    return train, test, price

#res32
esm_32 = pd.read_feather('ECRECer/data/featureBank/embd_esm32.feather')

cluster_path =  cluster_path =  cfg.DIR_DATASETS + 'task1/' + cfg.DIR_CLUSTER
train, test, price = read_data(cluster_path)
print(cfg.DIR_CLUSTER, '\t', 'train size: {0}\ttest size: {1}\tprice size: {2}'.format(len(train), len(test), len(price)))

# blast
start = time.time()
res_data=funclib.getblast(train,test)
print(' aligment finished \n query samples:{0}\n results samples: {1}'.format(len(test), len(res_data)))

res_data = res_data[['id', 'sseqid']].merge(train, left_on='sseqid', right_on='id', how='left')[['id_x', 'isenzyme']]
res_data =res_data.rename(columns={'id_x':'id','isenzyme':'isenzyme_blast'})
res_data = test[['id','isenzyme']].merge(res_data, on='id', how='left')

end = time.time()
mem, disk = funclib.monitor_usage()
time_cost = end - start

# save calculateMetrix results
baselineName, acc, precision, npv, recall, f1, tp, fp, fn, tn, up, un = funclib.caculateMetrix(groundtruth=res_data.isenzyme, predict=res_data.isenzyme_blast, baselineName='Blast', type='unfinded')
res_path = os.path.join(cfg.RESULTSDIR+cfg.DIR_CLUSTER, 'isEnzyme_matrix_blast.csv')
res = pd.DataFrame({'baselineName':baselineName, 'acc':acc, 'precision':precision, 'npv':npv, 'recall':recall, 'f1':f1, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn, 'up':up, 'un':un, 'mem':mem, 'disk':disk, 'time':time_cost})
res.to_csv(res_path, index=None)

start = time.time()
res_data_price = funclib.getblast_usedb(cfg.FILE_BLAST_TRAIN_DB,price)
print(' aligment finished \n query samples:{0}\n results samples: {1}'.format(len(price), len(res_data_price)))
res_data_price = res_data_price[['id', 'sseqid']].merge(train, left_on='sseqid', right_on='id', how='left')[['id_x', 'isenzyme']]
res_data_price =res_data_price.rename(columns={'id_x':'id','isenzyme':'isenzyme_blast'})
res_data_price = price[['id','isenzyme']].merge(res_data_price, on='id', how='left')
baselineName, acc, precision, npv, recall, f1, tp, fp, fn, tn, up, un = funclib.caculateMetrix(groundtruth=res_data_price.isenzyme, predict=res_data_price.isenzyme_blast, baselineName='Blast', type='unfinded')

end = time.time()
time_cost = end - start
mem, disk = funclib.monitor_usage()

# save calculateMetrix results
res_path = os.path.join(cfg.RESULTSDIR+cfg.DIR_CLUSTER, 'isEnzyme_matrix_blast_price.csv')
res = pd.DataFrame({'baselineName':baselineName, 'acc':acc, 'precision':precision, 'npv':npv, 'recall':recall, 'f1':f1, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn, 'up':up, 'un':un, 'mem':mem, 'disk':disk, 'time':time_cost})
res.to_csv(res_path, index=None)

trainset = train.copy()
testset = test.copy()
priceset = price.copy()
MAX_SEQ_LENGTH = 1500
trainset.seq = trainset.seq.map(lambda x : x[0:MAX_SEQ_LENGTH].ljust(MAX_SEQ_LENGTH, 'X'))
testset.seq = testset.seq.map(lambda x : x[0:MAX_SEQ_LENGTH].ljust(MAX_SEQ_LENGTH, 'X'))
priceset.seq = priceset.seq.map(lambda x : x[0:MAX_SEQ_LENGTH].ljust(MAX_SEQ_LENGTH, 'X'))

# get blast results
blastres_test=pd.DataFrame()
blastres_price = pd.DataFrame()

blastres_test['id']=res_data.id
blastres_test['isenzyme_groundtruth']=res_data.isenzyme
blastres_test['isenzyme_pred_blast']=res_data.isenzyme_blast

blastres_price['id']=res_data_price.id
blastres_price['isenzyme_groundtruth']=res_data_price.isenzyme
blastres_price['isenzyme_pred_blast']=res_data_price.isenzyme_blast

train_esm = trainset.merge(esm_32, on='id', how='left')
test_esm = testset.merge(esm_32, on='id', how='left')
price_esm = priceset.merge(esm_32, on='id', how='left')

X_train = np.array(train_esm.iloc[:,3:])
X_test = np.array(test_esm.iloc[:,3:])
X_price = np.array(price_esm.iloc[:,3:])

Y_train = np.array(train_esm.isenzyme.astype('int')).flatten()
Y_test = np.array(test_esm.isenzyme.astype('int')).flatten()
Y_price = np.array(price_esm.isenzyme.astype('int')).flatten()

# Calculate storage, memory and time cost of the following code
start = time.time()

groundtruth, predict, predictprob, model = funclib.knnmain(X_train, Y_train, X_test, Y_test, type='binary')
blastres_test['isenzyme_pred_xg'] = predict
blastres_test.isenzyme_pred_xg =blastres_test.isenzyme_pred_xg.astype('bool')
blastres_test['isenzyme_pred_slice']=blastres_test.apply(lambda x: x.isenzyme_pred_xg if str(x.isenzyme_pred_blast)=='nan' else x.isenzyme_pred_blast, axis=1)
baselineName, acc, precision, npv, recall, f1, tp, fp, fn, tn, up, un = funclib.caculateMetrix( groundtruth=blastres_test.isenzyme_groundtruth,  predict=blastres_test.isenzyme_pred_slice, baselineName='ECRECer', type='unfinded')

end = time.time()
mem, disk = funclib.monitor_usage()
time_cost = end - start

model_path = os.path.join(cfg.MODELDIR, 'isenzyme.h5')
joblib.dump(model, model_path)

blast_path = os.path.join(cfg.RESULTSDIR+cfg.DIR_CLUSTER, 'isEnzyme_results.tsv')
blastres_test.to_csv(blast_path, sep='\t', index=None)
# save calculateMetrix results
res_path = os.path.join(cfg.RESULTSDIR+cfg.DIR_CLUSTER, 'isEnzyme_matrix.csv')
res = pd.DataFrame({'baselineName':baselineName, 'acc':acc, 'precision':precision, 'npv':npv, 'recall':recall, 'f1':f1, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn, 'up':up, 'un':un, 'mem':mem, 'disk':disk, 'time':time_cost})
res.to_csv(res_path, index=None)

start = time.time()

groundtruth, predict, predictprob, model = funclib.knnmain(X_train, Y_train, X_price, Y_price, type='binary')
blastres_price['isenzyme_pred_xg'] = predict
blastres_price.isenzyme_pred_xg =blastres_price.isenzyme_pred_xg.astype('bool')
blastres_price['isenzyme_pred_slice']=blastres_price.apply(lambda x: x.isenzyme_pred_xg if str(x.isenzyme_pred_blast)=='nan' else x.isenzyme_pred_blast, axis=1)
baselineName, acc, precision, npv, recall, f1, tp, fp, fn, tn, up, un = funclib.caculateMetrix( groundtruth=blastres_price.isenzyme_groundtruth,  predict=blastres_price.isenzyme_pred_slice, baselineName='ECRECer', type='unfinded')

end = time.time()
time_cost = end - start
mem, disk = funclib.monitor_usage()

model_path = os.path.join(cfg.MODELDIR, 'isenzyme_price.h5')
joblib.dump(model, model_path)
blast_path = os.path.join(cfg.RESULTSDIR+cfg.DIR_CLUSTER, 'isEnzyme_results_price.tsv')
blastres_price.to_csv(blast_path, sep='\t', index=None)
# save calculateMetrix results
res_path = os.path.join(cfg.RESULTSDIR+cfg.DIR_CLUSTER, 'isEnzyme_matrix_price.csv')
res = pd.DataFrame({'baselineName':baselineName, 'acc':acc, 'precision':precision, 'npv':npv, 'recall':recall, 'f1':f1, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn, 'up':up, 'un':un, 'mem':mem, 'disk':disk, 'time':time_cost})
res.to_csv(res_path, index=None)




