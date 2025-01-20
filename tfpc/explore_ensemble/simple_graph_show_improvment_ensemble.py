import torch
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from utils.json_loader import load_json_into_pandas_dataframe


def get_perf_choosen_model(choosen_model, list_dico, df, model_weight):
    somme = 0
    for ind, row in df.iterrows():
        sequence = row["primary"]
        label = row["label"]
        all_preds = []
        for ind_model in choosen_model:
            pred_model_i = model_weight[ind_model] * np.array(
                list_dico[ind_model][sequence]
            )
            all_preds.append(pred_model_i)
        all_preds = np.array(all_preds)
        final_prediction = all_preds.mean(axis=0)
        somme += np.argmax(final_prediction) == label
    return somme / len(df)


def calc_accord(res_mod_i, res_mod_j):
    accord = 0
    for key, value in res_mod_i.items():
        pred_i = np.argmax(res_mod_i[key])
        pred_j = np.argmax(res_mod_j[key])
        if pred_i == pred_j:
            accord += 1
    return accord / len(res_mod_i)


# Parametre
folder_save = "data/tests_data/for_the_paper/"
nb_replique = 5
prefix = "ProtBert_EC40_layer_norm_proba_r"
suffix = "_complete_output_on_valid_limit_output_False_cls_outFalse.pkl"

# Chargement du jeux de données pour vérifier les labels
df = load_json_into_pandas_dataframe(
    "data/datasets/EC_prediction/EC_prediction_valid.json"
)

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

# On prépare les dictionaire pour acceder rapidement au prédiction de chauque model
list_dico = []
for num_replique in range(nb_replique):
    data = torch.load(folder_save + prefix + str(num_replique + 1) + suffix)
    sequences = [d[0][1:] for d in data]
    predictions = [d[1] for d in data]
    dico_pred = dict()
    for seq, pred in zip(sequences, predictions):
        dico_pred[seq] = pred
    list_dico.append(dico_pred)


# On affiche la matrice d'accord
matrice_accord = np.zeros((nb_replique, nb_replique))
for i in range(nb_replique):
    for j in range(nb_replique):
        res_mod_i = list_dico[i]
        res_mod_j = list_dico[j]
        accord = calc_accord(res_mod_i, res_mod_j)
        matrice_accord[i][j] = accord

plt.matshow(matrice_accord, cmap="bwr")
for (i, j), z in np.ndenumerate(matrice_accord):
    print(z)
    plt.text(
        j,
        i,
        "{:0.01f}".format(z * 100),
        ha="center",
        va="center",
        color="white",
        fontsize="x-large",
    )
plt.xlabel("Numero of model")
plt.ylabel("Numero of model")
plt.title("Agreement of models predictions")
plt.colorbar()
plt.show()


"""
acc_one = []
for k in range(0, nb_replique):
    acc = get_perf_choosen_model([k], list_dico, df, [1] * nb_replique)
    acc_one.append(acc)

min_acc = np.min(acc_one)
"""
model_weight = [
    0.04090423,
    0.16061692,
    0.001,
    0.12071269,
    0.12071269,
]  # np.array([a - min_acc for a in acc_one]) * 100 + 0.001
print("model_weight :", model_weight)


# On parcours les séquences et on regarde les performances
perf_each_ens = []
all_res = [[] for _ in range(nb_replique)]
array = np.array(list(range(nb_replique)))
for k in range(1, nb_replique + 1):
    all_combi = list(combinations(array, k))
    all_acc = []
    for choosen_model in all_combi:
        acc = get_perf_choosen_model(choosen_model, list_dico, df, model_weight)
        all_acc.append(acc)

        all_res[k - 1].append(acc)
        plt.scatter([k], [acc])
    perf = [np.mean(all_acc), np.std(all_acc)]

    perf_each_ens.append(perf)
print(perf_each_ens)
plt.show()

plt.errorbar(
    array,
    [p[0] for p in perf_each_ens],
    yerr=2 * np.array([p[1] for p in perf_each_ens]),
)
plt.show()

print(all_res)
prec_med = None
for ind, res in enumerate(all_res):
    for r in res:
        # plt.scatter([ind], [r])
        pass
    med = np.mean(res)
    if ind != 0:
        plt.plot([ind, ind + 1], [prec_med, med], color="blue")
    prec_med = med
    plt.boxplot(res, positions=[ind + 1], showfliers=False)
plt.xlabel("Number of models")
plt.ylabel("Accuracy")
plt.title("Accuracy on EC prediction at lvl 2")
plt.show()
