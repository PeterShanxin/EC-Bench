import matplotlib.pyplot as plt
from itertools import combinations
import torch
import pandas as pd
import numpy as np
from utils_ensemble import (
    list_of_result_to_dico,
    acc_of_ensemble,
    acc_of_ensemble_with_cross_val,
)
import matplotlib.patches as mpatches
import argparse
import scipy
import copy
import json
from utils.json_loader import load_json_into_pandas_dataframe

compteur_color = 0
all_legends = []


def graph_ensemble_strategies(
    LIST_PATH_RESULT,
    type_dataset,
    decalage,
    method_aggreg,
    nb_class,
    prefix,
    cross_val,
    temperature_softmax=1,
):
    global compteur_color
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

    nb_model = len(LIST_PATH_RESULT)
    all_result = []
    for path_res in LIST_PATH_RESULT:
        res_i = torch.load(prefix + path_res)
        res_i = [[r[0], r[1][:nb_class], r[2]] for r in res_i]
        all_result.append(res_i)

    list_all_norm = []
    for indice, res_mod_i in enumerate(all_result):
        list_all_norm.append(np.linalg.norm([r[1] for r in res_mod_i]))

    array = np.array(list(range(nb_model)))
    dico_nb_model_acc = {str(k): None for k in range(1, nb_model + 1)}
    nb_combi_effec = 0
    nb_total_combi = 0
    for k in range(1, nb_model + 1):
        nb_total_combi += scipy.special.binom(nb_model, k)
    for k in range(1, nb_model + 1):
        all_combi = list(combinations(array, k))
        all_acc_mean = []
        # Pour l'instant acc std n'est pas utiliser mais ça pourras changé pour l'integrer dans le graph
        all_acc_std = []
        for combi in all_combi:
            print((nb_combi_effec / nb_total_combi) * 100, "% effectué")
            nb_combi_effec += 1
            result_selec = []
            for ind in combi:
                result_selec.append(all_result[ind])
            dico_pred = list_of_result_to_dico(result_selec)
            if cross_val:
                acc_mean, acc_std = acc_of_ensemble_with_cross_val(
                    df,
                    dico_pred,
                    method=method_aggreg,
                    list_all_norm=list_all_norm,
                    temp=temperature_softmax,
                )
                all_acc_std.append(acc_std)
            else:
                acc_mean = acc_of_ensemble(
                    df,
                    dico_pred,
                    method=method_aggreg,
                    list_all_norm=list_all_norm,
                    temp=temperature_softmax,
                )
            all_acc_mean.append(acc_mean)
        dico_nb_model_acc[str(k)] = all_acc_mean

    print(dico_nb_model_acc)

    all_values = list(dico_nb_model_acc.values())
    for k in range(len(all_values)):
        pos = np.zeros(len(all_values))
        plt.boxplot(
            all_values[k],
            positions=[k + 1 + decalage],
            notch=True,
            patch_artist=True,
            boxprops=dict(facecolor="C" + str(compteur_color)),
        )
    plt.title("Accuracy depending on the number of models")

    compteur_color += 1


# Specification of the argument to specify when you launch the script
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="indicate the config file path")
args = parser.parse_args()

# Load other attribute from config file
fichier = open(args.config_path)
dict_config = json.load(fichier)
print(dict_config)

type_dataset = dict_config["type_dataset"]
prefix = dict_config["prefix"]
nb_class = dict_config["nb_class"]
list_of_list_ens_compare = dict_config["list_of_list_ens_compare"]
if "cross_val" in dict_config.keys():
    cross_val = dict_config["cross_val"]
else:
    cross_val = False

if "strat_post_calib" in dict_config.keys():
    strat_post_calib = dict_config["strat_post_calib"]
else:
    strat_post_calib = False

# La strategie post calib fonctionne uniquement avec une cross validation
if strat_post_calib and not cross_val:
    raise RuntimeError(
        "Impossible de faire strategie post calibration sans cross validation"
    )

decalage = 0

# Boucle for temp a enlever

for LIST_PATH_RESULT in list_of_list_ens_compare:
    name = LIST_PATH_RESULT[0].split("/")[1]
    name = "_".join(name.split("_")[:-9])
    if strat_post_calib:
        print("PROBA calibré")
        # Box plot for proba
        graph_ensemble_strategies(
            LIST_PATH_RESULT,
            type_dataset,
            decalage=decalage,
            method_aggreg="weight_median",
            nb_class=nb_class,
            prefix=prefix,
            cross_val=cross_val,
        )

        legend_i = mpatches.Patch(
            facecolor="C" + str(compteur_color - 1),
            label=name + "_weight_median",
        )
        all_legends.append(legend_i)
        decalage += 0.17

    # Box plot for weight
    print("POIDS")
    graph_ensemble_strategies(
        LIST_PATH_RESULT,
        type_dataset,
        decalage=decalage,
        method_aggreg="weight",
        nb_class=nb_class,
        prefix=prefix,
        cross_val=cross_val,
    )
    legend_i = mpatches.Patch(
        facecolor="C" + str(compteur_color - 1), label=name + "_weight"
    )
    all_legends.append(legend_i)

    decalage += 0.17
    print("PROBA")
    # Box plot for proba
    graph_ensemble_strategies(
        LIST_PATH_RESULT,
        type_dataset,
        decalage=decalage,
        method_aggreg="proba",  #'proba'
        nb_class=nb_class,
        prefix=prefix,
        cross_val=cross_val,
    )

    legend_i = mpatches.Patch(
        facecolor="C" + str(compteur_color - 1), label=name + "_proba"
    )
    all_legends.append(legend_i)
    decalage += 0.17


plt.xlabel("Number of model")
plt.ylabel("Accuracy")
plt.legend(handles=all_legends)
plt.show()
