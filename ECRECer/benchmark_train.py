import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from sklearn.model_selection import train_test_split
import ECRECer.benchmark_common as bcommon
import ECRECer.config as cfg
from keras.callbacks import TensorBoard,ModelCheckpoint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def make_ec_label(train_label, test_label, price_label, price_149_label, file_save, force_model_update=False):
    ecset = sorted( set(list(train_label) + list(test_label) + list(price_label) + list(price_149_label)))
    ec_label_dict = {k: v for k, v in zip(ecset, range(len(ecset)))}
    np.save(file_save, ec_label_dict)
    return  ec_label_dict

def train_ec_slice(trainX, trainY, modelPath, force_model_update=False):
    """[训练slice模型]
    Args:
        trainX ([DataFarame]): [X特征]
        trainY ([DataFrame]): [ Y标签]]
        modelPath ([string]): [存放模型的目录]
        force_model_update (bool, optional): [是否强制更新模型]. Defaults to False.
    """
    print('start train slice model')
    cmd = ''' ECRECer/slice_train {0} {1} {2} -m 100 -c 300 -s 300 -k 700 -o 32 -t 32 -C 1 -f 0.000001 -siter 20 -stype 0 -q 0 '''.format(trainX, trainY, modelPath)
    print(cmd)
    os.system(cmd)
    print('train finished')

def get_train_X_Y(traindata, feature_bankfile, task=1):
    """
    Args:
        traindata ([DataFrame]): [description]

    Returns:
        [DataFrame]: [trianX, trainY]
    """
    traindata = traindata.merge(feature_bankfile, on='id', how='left')
    train_X = np.array(traindata.iloc[:,3:])
    if task == 1:
        train_Y = bcommon.make_onehot_label(label_list=traindata['isenzyme'].to_list(), save=True, file_encoder=cfg.DICT_LABEL_T1,type='single')
    if task == 2:
        train_Y = bcommon.make_onehot_label(label_list=traindata['functionCounts'].to_list(), save=True, file_encoder=cfg.DICT_LABEL_T2, type='single')
    if task == 3 :
        train_Y = bcommon.make_onehot_label(label_list=traindata['ec_number'], save=True, file_encoder=cfg.DICT_LABEL_T3, type='multi')
    return train_X, train_Y

def train_isenzyme(X,Y, model_file, vali_ratio=0.2, force_model_update=False, epochs=1):
    x_train, x_vali, y_train, y_vali = train_test_split(np.array(X), np.array(Y), test_size=vali_ratio, shuffle=False, random_state=42)
    x_train = x_train.reshape(x_train.shape[0],1,-1)
    x_vali  = x_vali.reshape(x_vali.shape[0],1,-1)
    tbCallBack = TensorBoard(log_dir=f'{cfg.TEMPDIR}model/task1', histogram_freq=1,write_grads=True)
    checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy', mode='auto', save_best_only='True')
    instant_model = bcommon.mgru_attion_model(input_dimensions=X.shape[1], gru_h_size=512, dropout=0.2, lossfunction='binary_crossentropy', 
                    evaluation_metrics='accuracy', activation_method = 'sigmoid', output_dimensions=Y.shape[1] )
        
    instant_model.fit(x_train, y_train, validation_data=(x_vali, y_vali), batch_size=512, epochs= epochs, callbacks=[tbCallBack, checkpoint])
    print(f'train_isenzyme finished, best model saved to: {model_file}')
        

def train_howmany_enzyme(X, Y, model_file, force_model_update=False, epochs=1):
    x_train, x_vali, y_train, y_vali = train_test_split(np.array(X), np.array(Y), test_size=0.3, shuffle=False, random_state=42)
    x_train = x_train.reshape(x_train.shape[0],1,-1)
    x_vali  = x_vali.reshape(x_vali.shape[0],1,-1)
    tbCallBack = TensorBoard(log_dir=f'{cfg.TEMPDIR}model/task2', histogram_freq=1,write_grads=True)
    checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy', mode='auto', save_best_only='True')
    instant_model = bcommon.mgru_attion_model(input_dimensions=X.shape[1], gru_h_size=128, dropout=0.2, lossfunction='categorical_crossentropy', 
                    evaluation_metrics='accuracy', activation_method = 'softmax', output_dimensions=Y.shape[1])
    
    instant_model.fit(x_train, y_train, validation_data=(x_vali, y_vali), batch_size=128, epochs= epochs, callbacks=[tbCallBack, checkpoint])
    
    print(f'train how many enzyme finished, best model saved to: {model_file}')

def train_ec(X, Y, model_file,  force_model_update=False, epochs=1):
    x_train, x_vali, y_train, y_vali = train_test_split(np.array(X), np.array(Y), test_size=0.3, shuffle=False, random_state=42)
    x_train = x_train.reshape(x_train.shape[0],1,-1)
    x_vali  = x_vali.reshape(x_vali.shape[0],1,-1)
    instant_model = bcommon.mgru_attion_model(input_dimensions=X.shape[1], gru_h_size=512, dropout=0.2, lossfunction='categorical_crossentropy', 
                    evaluation_metrics='accuracy', activation_method = 'softmax', output_dimensions=Y.shape[1] )
    tbCallBack = TensorBoard(log_dir=f'{cfg.TEMPDIR}model/task3', histogram_freq=1,write_grads=True)
    checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy', mode='auto', save_best_only='True')
    instant_model.fit(x_train, y_train, validation_data=(x_vali, y_vali), batch_size=16, epochs= epochs, callbacks=[tbCallBack, checkpoint])
    
    print(f'train EC model finished, best model saved to: {model_file}')


if __name__ =="__main__":

    EMBEDDING_METHOD = 'esm32'

    # 1. read tranning data
    print('step 1 loading task data')

    # please specific the tranning data for each tasks
    data_task1_train = pd.read_feather(cfg.FILE_TASK1_TRAIN)
    data_task2_train = pd.read_feather(cfg.FILE_TASK2_TRAIN)
    data_task3_train = pd.read_feather(cfg.FILE_TASK3_TRAIN)
    
    # 2. read tranning feature
    print(f'step 2: Loading features, embdding method={EMBEDDING_METHOD}')
    feature_df = bcommon.load_data_embedding(embedding_type=EMBEDDING_METHOD)
    
    #3. train task1 model
    print('step 3: train isEnzyme model')
    task1_X, task1_Y = get_train_X_Y(traindata=data_task1_train, feature_bankfile=feature_df, task=1)
    train_isenzyme(X=task1_X, Y=task1_Y, model_file= cfg.ISENZYME_MODEL, force_model_update=cfg.UPDATE_MODEL, epochs=400)
    '''
    #4. train task2 model
    print('step 4: train how many enzymes model')
    task2_X, task2_Y = get_train_X_Y(traindata=data_task2_train, feature_bankfile=feature_df, task=2)
    train_howmany_enzyme(X=task2_X, Y=task2_Y, model_file=cfg.HOWMANY_MODEL, force_model_update=cfg.UPDATE_MODEL, epochs=400)
    '''
    #5. train task3 model
    print('step 5 train EC model')
    task3_X, task3_Y = get_train_X_Y(traindata=data_task3_train, feature_bankfile=feature_df, task=3)
    train_ec(X=task3_X, Y=task3_Y, model_file=cfg.EC_MODEL, force_model_update=cfg.UPDATE_MODEL, epochs=400)
    print('train finished')
