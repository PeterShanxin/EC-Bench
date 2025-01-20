import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import pandas as pd
from utils_ensemble import list_of_result_to_dico, acc_of_ensemble
import torch
from scipy.signal import savgol_filter
from sklearn.model_selection import ShuffleSplit
import json
import argparse
from utils.json_loader import load_json_into_pandas_dataframe


def create_calibration_graph(
    path_res, nb_categorie, normalization_type, temperature_softmax=1
):
    extract_name = path_res.split("/")[1]
    extract_name = extract_name[:-61]
    # Which model I want to put on the ensemble and on which dataset(train/test/valid)
    if type_dataset == "valid":
        # We load the coresponding dataset
        df = load_json_into_pandas_dataframe(
            "data/datasets/EC_prediction/EC_prediction_valid.json"
        )
    else:
        exit("Only valid supported for now")

    # On convertie les label vers leurs valeurs dans le vocabulaire du/des models
    dict_label = torch.load(
        "data/models/fine_tune_models/ProtBert_EC40_classic_r1/classif_EC_pred_lvl_2_vocab.pth"
    )

    # On convertie les labels du dataset
    for ind, row in df.iterrows():
        label = row["label"]
        df.at[ind, "label"] = dict_label[label]

    # On tronque le dataset à 1024 AA en longueur max
    for ind, row in df.iterrows():
        sequence = row["primary"]
        if len(sequence) > 1024:
            df.at[ind, "primary"] = sequence[:1024]

    res_i = torch.load(prefix + path_res)
    res_i = [[r[0], r[1][:nb_class], r[2]] for r in res_i]
    nb_exemples_total = len(res_i)
    dico_pred = list_of_result_to_dico([res_i])

    taille_categorie = 1 / nb_categorie
    all_categorie = [[] for _ in range(nb_categorie)]
    for res in res_i:
        if normalization_type == "softmax":
            proba = np.max(softmax(np.array(res[1]) / temperature_softmax))
        elif normalization_type == "classic_sum":
            mini = np.min(res[1])
            proba = np.max((res[1] - mini) / np.sum((res[1] - mini)))
        else:
            raise RuntimeError("Normlization unknown")
        correct = res[2].item()
        quotient = int(proba / taille_categorie)
        all_categorie[quotient].append([correct, proba])

    # ECE : Expected Calibration Error
    score_ECE = 0

    for cat in all_categorie:
        len_BM = len(cat)
        acc_BM = np.mean([c[0] for c in cat])
        conf_BM = np.mean([c[1] for c in cat])
        if len_BM > 0:
            score_ECE += (len_BM / nb_exemples_total) * np.abs(acc_BM - conf_BM)

    score_ECE = score_ECE * 100
    # print(extract_name)
    # print("ECE score en pourcentage:", score_ECE)

    new_tab = []
    nb = 0
    for cat in all_categorie:
        cat = [c[0] for c in cat]
        new_tab.append(np.mean(cat))
        nb += 1
        if nb == 1:
            print(cat)
    offset = taille_categorie / 2
    x = [taille_categorie * k + offset for k in range(nb_categorie)]

    # TEST two group
    print(x)
    # Fin Test two group

    plt.plot(x, new_tab, label=extract_name)
    return score_ECE


def Calc_best_ECE_change_temp(path_res, nb_categorie=20):
    """
    A CONTINUER A CODER
    Validation croisé avec les données de "validation" :
    On prend 20% des données on estime la meilleurs temperature et on prend le score ECE sur les 90% restant et on repete 5 fois
    pour avoir le ECE best moyen et la temperature moyenne optimale.
    """
    extract_name = path_res.split("/")[1]
    extract_name = extract_name[:-61]
    # Which model I want to put on the ensemble and on which dataset(train/test/valid)
    if type_dataset == "valid":
        # We load the coresponding dataset
        df = load_json_into_pandas_dataframe(
            "data/datasets/EC_prediction/EC_prediction_valid.json"
        )
    else:
        exit("Only valid supported for now")

    # On convertie les label vers leurs valeurs dans le vocabulaire du/des models
    dict_label = torch.load(
        "data/models/fine_tune_models/fine_tune_ProtBert_BFD_for_EC_prediction_r2/classif_EC_pred_lvl_2_vocab.pth"
    )

    # On convertie les labels du dataset
    for ind, row in df.iterrows():
        label = row["label"]
        df.at[ind, "label"] = dict_label[label]

    # On tronque le dataset à 1024 AA en longueur max
    for ind, row in df.iterrows():
        sequence = row["primary"]
        if len(sequence) > 1024:
            df.at[ind, "primary"] = sequence[:1024]

    res_i = torch.load(prefix + path_res)
    res_i = [[r[0], r[1][:nb_class], r[2]] for r in res_i]
    nb_exemples_total = len(res_i)
    dico_pred = list_of_result_to_dico([res_i])

    print(dico_pred)

    num_part = 0
    nb_slice_train = int(len(dico_pred) * 0.2)
    train_keys = list(dico_pred.keys())[
        num_part * nb_slice_train : (num_part + 1) * nb_slice_train
    ]
    test_keys = (
        list(dico_pred.keys())[: num_part * nb_slice_train]
        + list(dico_pred.keys())[(num_part + 1) * nb_slice_train :]
    )
    train_data = [dico_pred[key][0] for key in train_keys]
    test_data = [dico_pred[key][0] for key in test_keys]

    # CALC TEMP optimale on train
    taille_categorie = 1 / nb_categorie
    all_categorie = [[] for _ in range(nb_categorie)]
    for res in train_data:
        proba = np.max(softmax(np.array(res[1]) / temperature_softmax))
        correct = res[2].item()
        quotient = int(proba / taille_categorie)
        all_categorie[quotient].append([correct, proba])

    # ECE : Expected Calibration Error
    score_ECE = 0

    for cat in all_categorie:
        len_BM = len(cat)
        acc_BM = np.mean([c[0] for c in cat])
        conf_BM = np.mean([c[1] for c in cat])
        if len_BM > 0:
            score_ECE += (len_BM / nb_exemples_total) * np.abs(acc_BM - conf_BM)

    # CALC SCORE ON TEST

    # RETURN MEAN all values of score and temperature opti
    score_ECE = score_ECE * 100
    print(extract_name)
    print("ECE score en pourcentage:", score_ECE)


def plot_theorical_calibration(nb_categorie):
    taille_categorie = 1 / nb_categorie
    # create theorical perfect calibration and plot the graph
    offset = taille_categorie / 2
    x = [taille_categorie * k + offset for k in range(nb_categorie)]
    true_value = [
        (1 / nb_categorie) * k + (1 / nb_categorie) / 2 for k in range(nb_categorie)
    ]

    plt.plot(x, true_value, label="theorical")


# Specification of the argument to specify when you launch the script
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="indicate the config file path")
args = parser.parse_args()

# Load other attribute from config file
fichier = open(args.config_path)
dict_config = json.load(fichier)
print(dict_config)
version = "classic"  # "classic"  or "multiple_temp"

type_dataset = dict_config["type_dataset"]
prefix = dict_config["prefix"]
nb_class = dict_config["nb_class"]
list_nb_cat = dict_config["list_nb_cat"]
list_norm_type = dict_config["list_norm_type"]

list_path_res = dict_config["list_path_res"]
# "label_smoothing_EC_pred/fine_tune_for_EC_prediction_with_label_smoothing_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
# "My_ensemble_EC_pred/fine_tune_for_EC_prediction_with_lr_scheduler_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl"
if version == "classic":
    all_score_ECE = []
    for nb_categorie in list_nb_cat:
        for path_res, norm_type in zip(list_path_res, list_norm_type):
            score_ECE = create_calibration_graph(
                path_res, nb_categorie, norm_type, temperature_softmax=1
            )
            all_score_ECE.append(score_ECE)
            # Calc_best_ECE_change_temp(path_res)
        plot_theorical_calibration(nb_categorie)
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed accuracy")
        plt.legend()
        plt.show()

    print("Mean :", np.mean(all_score_ECE))
    print("Std :", np.std(all_score_ECE))
elif version == "multiple_temp":
    all_temps = [1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8]
    all_mean = []
    all_std = []
    for temp in all_temps:
        all_score_ECE = []
        for nb_categorie in list_nb_cat:
            for path_res, norm_type in zip(list_path_res, list_norm_type):
                score_ECE = create_calibration_graph(
                    path_res, nb_categorie, norm_type, temperature_softmax=temp
                )
                all_score_ECE.append(score_ECE)
                # Calc_best_ECE_change_temp(path_res)
        print("Temp :", temp)
        print("Mean :", np.mean(all_score_ECE))
        print("Std :", np.std(all_score_ECE))
        all_mean.append(np.mean(all_score_ECE))
        all_std.append(np.std(all_score_ECE))

    for m in all_mean:
        print(m)
    print("##" * 30)
    for s in all_std:
        print(s)
