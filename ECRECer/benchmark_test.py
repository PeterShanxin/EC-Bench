import re
import pandas as pd
import numpy as np
import joblib
import os, sys
sys.path.append(os.getcwd())

import ECRECer.benchmark_common as bcommon
import ECRECer.config as cfg
from Bio import SeqIO

# region 获取「酶｜非酶」预测结果
def get_isEnzymeRes(querydata, model_file):
    """
    Args:
        querydata ([DataFrame])
        model_file ([string])
    Returns:
        [DataFrame]
    """
    model = joblib.load(model_file)
    predict = model.predict(querydata)
    predictprob = model.predict_proba(querydata)
    return predict, predictprob[:, 1]
# endregion

# region
def get_howmany_Enzyme(querydata, model_file):
    """
    Args:
        querydata ([DataFrame])
        model_file ([string])
    Returns:
        [DataFrame]
    """
    model = joblib.load(model_file)
    predict = model.predict(querydata)
    predictprob = model.predict_proba(querydata)
    return predict+1, predictprob 
# endregion

# region
def get_slice_res(slice_query_file, model_path, dict_ec_label,test_set, res_file):
    """
    Args:
        slice_query_file ([string])
        model_path ([string])
        res_file ([string]])
    Returns:
        [DataFrame]
    """

    cmd = '''./ECRECer/slice_predict {0} {1} {2} -o 32 -b 0 -t 32 -q 0'''.format(slice_query_file, model_path, res_file)
    print(cmd)
    os.system(cmd)
    result_slice = pd.read_csv(res_file, header=None, skiprows=1, sep=' ')
    # 5.3 
    slice_pred_rank, slice_pred_prob = sort_results(result_slice)
    # 5.4 
    slice_pred_ec = translate_slice_pred(slice_pred=slice_pred_rank, ec2label_dict = dict_ec_label, test_set=test_set)
    slice_pred_prob = translate_slice_pred_prob(slice_pred=slice_pred_prob, test_set=test_set)

    return slice_pred_ec, slice_pred_prob
# endregion

#region
def sort_results(result_slice):
    """
    @pred_top
    @pred_pb_top
    """
    pred_top = []
    pred_pb_top = []
    aac = []
    for index, row in result_slice.iterrows():
        row_trans = [*row.apply(lambda x: x.split(':')).values]
        row_trans = pd.DataFrame(row_trans).sort_values(by=[1], ascending=False)
        pred_top += [list(np.array(row_trans[0]).astype('int'))]
        pred_pb_top += [list(np.array(row_trans[1]).astype('float'))]
    pred_top = pd.DataFrame(pred_top)
    pred_pb_top = pd.DataFrame(pred_pb_top)
    return pred_top, pred_pb_top
#endregion

#region
def get_test_set(data):
    """
    Args:
        data ([DataFrame])

    Returns:
        [DataFrame]
    """
    testX = data.iloc[:,7:]
    testY = data.iloc[:,:6]
    return testX, testY
#endregion

#region 
def translate_slice_pred(slice_pred, ec2label_dict, test_set):
    """
    Args:
        slice_pred ([DataFrame])
        ec2label_dict ([dict]])
        test_set ([DataFrame])

    Returns:
        [type]: [description]
    """
    label_ec_dict = {value:key for key, value in ec2label_dict.items()}
    res_df = pd.DataFrame()
    res_df['id'] = test_set.id
    colNames = slice_pred.columns.values
    for colName in colNames:
        res_df['top'+str(colName)] = slice_pred[colName].apply(lambda x: label_ec_dict.get(x))
    return res_df
#endregion

#region 
def translate_slice_pred_prob(slice_pred, test_set):
    """
    Args:
        slice_pred ([DataFrame])
        ec2label_dict ([dict]])
        test_set ([DataFrame])

    Returns:
        [type]: [description]
    """
    res_df = pd.DataFrame()
    res_df['id'] = test_set.id
    colNames = slice_pred.columns.values
    for colName in colNames:
        res_df['top_prob'+str(colName)] = slice_pred[colName]
    return res_df
#endregion

#region 
def run_integrage(slice_pred, dict_ec_transfer):
    """
    Args:
        slice_pred ([DataFrame])
        dict_ec_transfer ([dict])
    Returns:
        [DataFrame]
    """
    # top10
    slice_pred = slice_pred.iloc[:,np.r_[0:11, 21:28]]

    with pd.option_context('mode.chained_assignment', None):
        slice_pred['is_enzyme_i'] = slice_pred.apply(lambda x: int(x.isemzyme_blast) if str(x.isemzyme_blast)!='nan' else x.isEnzyme_pred_xg, axis=1)

    # EC
    for i in range(9):
        with pd.option_context('mode.chained_assignment', None):
            slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if x.is_enzyme_i==0 else x['top'+str(i)], axis=1)
            slice_pred['top0'] = slice_pred.apply(lambda x: x.ec_number_blast if str(x.ec_number_blast)!='nan' else x.top0, axis=1)

    for i in range(1,10):
        with pd.option_context('mode.chained_assignment', None):
            slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if str(x.ec_number_blast)!='nan' else x['top'+str(i)], axis=1) #有比对结果的
            slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if int(x.functionCounts_pred_xg) < int(i+1) else x['top'+str(i)], axis=1) #无比对结果的
    with pd.option_context('mode.chained_assignment', None):
        slice_pred['top0']=slice_pred['top0'].apply(lambda x: '' if x=='-' else x)
    
    # EC
    for index, row in slice_pred.iterrows():
        ecitems=row['top0'].split(',')
        if len(ecitems)>1:
            for i in range(len(ecitems)):
                slice_pred.loc[index,'top'+str(i)] =  ecitems[i].strip()

    slice_pred.reset_index(drop=True, inplace=True)

    with pd.option_context('mode.chained_assignment', None):
        slice_pred['pred_functionCounts'] = slice_pred.apply(lambda x: int(x['functionCounts_blast']) if str(x['functionCounts_blast'])!='nan' else x.functionCounts_pred_xg ,axis=1)
    colnames=[  'id', 
                'pred_ec1', 
                'pred_ec2', 
                'pred_ec3', 
                'pred_ec4', 
                'pred_ec5' , 
                'pred_ec6', 
                'pred_ec7', 
                'pred_ec8', 
                'pred_ec9', 
                'pred_ec10',
                'pred_isEnzyme', 
                'pred_functionCounts'
            ]
    slice_pred=slice_pred.iloc[:, np.r_[0:11, 18,19]]
    slice_pred.columns = colnames

    # EC
    for i in range(1,11):
        slice_pred['pred_ec'+str(i)] = slice_pred['pred_ec'+str(i)].apply(lambda x: dict_ec_transfer.get(x) if x in dict_ec_transfer.keys() else x)
    
    with pd.option_context('mode.chained_assignment', None):
        slice_pred.pred_functionCounts[slice_pred.pred_ec1.isnull()] = 0

    return slice_pred
#endregion

if __name__ == '__main__':

    EMBEDDING_METHOD = 'esm32'

    print('step 1: loading data')
    train = pd.read_feather(cfg.TRAIN_FEATURE)
    test = pd.read_feather(cfg.TEST_FEATURE)
    feature_df = bcommon.load_data_embedding(embedding_type=EMBEDDING_METHOD)
    train = train.merge(feature_df, on='id', how='left')
    test = test.merge(feature_df, on='id', how='left')

    dict_ec_label = np.load(cfg.FILE_EC_LABEL_DICT, allow_pickle=True).item()
    dict_ec_transfer = np.load(cfg.FILE_TRANSFER_DICT, allow_pickle=True).item() 

    print('step 2 get blast results')
    blast_res = bcommon.get_blast_prediction(  reference_db=cfg.FILE_BLAST_TRAIN_DB, 
                                                train_frame=train, 
                                                test_frame=test,
                                                results_file=cfg.FILE_BLAST_RESULTS,
                                                identity_thres=cfg.TRAIN_BLAST_IDENTITY_THRES
                                            )

    print('step 3. get isEnzyme results')
    testX, testY = get_test_set(data=test)
    isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=testX, model_file=cfg.ISENZYME_MODEL)

    print('step 4. get howmany functions ')
    howmany_Enzyme_pred, howmany_Enzyme_pred_prob = get_howmany_Enzyme(querydata=testX, model_file=cfg.HOWMANY_MODEL)

    print('step 5. get EC prediction results')

    bcommon.prepare_slice_file( x_data=testX, 
                                y_data=testY['ec_number'], 
                                x_file=cfg.FILE_SLICE_TESTX,
                                y_file=cfg.FILE_SLICE_TESTY,
                                ec_label_dict=dict_ec_label
                            )

    # slice_pred_ec = get_slice_res(slice_query_file=cfg.FILE_SLICE_TESTX, model_path= cfg.MODELDIR, dict_ec_label=dict_ec_label, test_set=test,  res_file=cfg.FILE_SLICE_RESULTS)
    slice_pred_ec = get_slice_res(slice_query_file=cfg.DATADIR+'slice_test_x_esm33.txt', model_path= cfg.MODELDIR+'/slice_esm33', dict_ec_label=dict_ec_label, test_set=test,  res_file=cfg.FILE_SLICE_RESULTS)
    
    slice_pred_ec['isEnzyme_pred_xg'] = isEnzyme_pred
    slice_pred_ec['functionCounts_pred_xg'] = howmany_Enzyme_pred
    slice_pred_ec = slice_pred_ec.merge(blast_res, on='id', how='left')

    # slice_pred_ec.to_csv(cfg.RESULTSDIR + 'singele_slice.tsv', sep='\t', index=None)
    # 5.5 blast EC

    # 6.(slice_pred=slice_pred_ec, dict_ec_transfer=dict_ec_transfer)
    slice_pred_ec = run_integrage(slice_pred=slice_pred_ec, dict_ec_transfer = dict_ec_transfer)    
    slice_pred_ec.to_csv(cfg.FILE_INTE_RESULTS, sep='\t', index=None)

    print('predict finished')