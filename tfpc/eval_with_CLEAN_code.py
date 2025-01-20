from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("models_architectures/")

#from models_architectures.ensembling_model import EnsemblingModel


def get_eval_metrics(pred_label, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
       
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
   
    return pre, rec, f1

@torch.no_grad()
def get_proba_for_df(model,vocab,df,max_seq_len = 2048):
    print(df.columns)
    dico_proba = {}
    dico_true_target = {}
    for _, row in tqdm(df.iterrows(),total=len(df)):
        sequence = row["Sequence"]
        list_true_ec = row["EC number"]
        
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        dico_proba[sequence] = list(proba_pred.cpu().numpy())
        dico_true_target[sequence] = list_true_ec
    return dico_proba,dico_true_target
    
def get_pred_and_true(dico_proba,dico_true_target,inverse_class_vocab,order_seq):
    pred_label = []
    true_label = []
    for sequence in order_seq:
        proba = dico_proba[sequence]
        true_label_this_seq = dico_true_target[sequence].split(";")
        ind_max = np.argmax(proba)
        pred_ec = inverse_class_vocab[ind_max].split(";")
        pred_label.append(pred_ec)
        true_label.append(true_label_this_seq)

    return pred_label,true_label


# def get_best_threshold(model,vocab,class_vocab):
#     df_train = pd.read_json("data/datasets/CLEAN_dataset/CLEAN_dataset_train.json")
#     df_subset = df_train.sample(n=10)
#     dico_proba,dico_true_target = get_proba_for_df(model,vocab,df_subset)
#     all_thesholds = np.linspace(0,1,num=100)
#     for threshold in all_thesholds:
#         get_pred_from_proba(dico_proba,threshold,class_vocab)
    

def get_prediction(model,vocab,inverse_class_vocab,dataset_name):
    if dataset_name=="NEW-392":
        df_test = pd.read_csv("data/datasets/CLEAN_dataset/new.csv",sep="\t")
    elif dataset_name=="Price-149":
        df_test = pd.read_csv("data/datasets/CLEAN_dataset/price.csv",sep="\t")
    dico_proba, dico_true_target = get_proba_for_df(model,vocab,df_test)
    order_seq = list(dico_proba.keys())
    pred_label,true_label = get_pred_and_true(dico_proba,dico_true_target,inverse_class_vocab,order_seq)
    all_label = set()
    for one_true_list in true_label:
        for ec in one_true_list:
            all_label.add(ec)
    for list_pred in pred_label:
        for ec in list_pred:
            all_label.add(ec)
    return pred_label, true_label, all_label


if __name__=="__main__":
    if torch.cuda.is_available():
        device="cuda"
    else:
        device = "cpu"
    vocab_path="pre_trained_models/30_layer_uniparc+BFD/vocab.pkl"
    model_name = "EnzBert_CLEAN_train" #"EnzBert_SwissProt_2021_04" # "EnzBert_CLEAN_train"
    class_vocab = torch.load("data/models/fine_tune_models/"+model_name+"/classif_EC_pred_lvl_2_vocab.pth",map_location=device)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}
    vocab = torch.load("data/models/" + vocab_path,map_location=device)
    model = torch.load("data/models/fine_tune_models/"+model_name+"/classif_EC_pred_lvl_2.pth",map_location=device)
    model = model.eval()

    all_test_dataset = ["NEW-392","Price-149"]
    for dataset_name in all_test_dataset:
        pred_label, true_label, all_label = get_prediction(model,vocab,inverse_class_vocab,dataset_name)
        pre, rec, f1 = get_eval_metrics(pred_label, true_label, all_label)

        print('-' * 75)
        print("Dataset:",dataset_name)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3}')
        print('-' * 75)  
