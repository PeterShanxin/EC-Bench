from interpretability.evaluate_interpretability_method import get_scores
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def nb_res_ninety_percent(one_map):
    sorted_one_map = np.flip(np.sort(one_map))
    somme_attn = one_map.sum() * 0.9
    somme_en_cours = 0
    compteur = 0
    while somme_en_cours < somme_attn:
        somme_en_cours += sorted_one_map[compteur]
        compteur += 1
    return compteur


def graph_number_of_attention():
    base_path = "data/residues_of_interest/"
    method_name = "Raw_attention_cls"
    scores = get_scores(base_path, method_name)

    somme_nb_interet = np.zeros(480)
    somme_prop_interest = np.zeros(480)
    list_all_dist = [[] for _ in range(480)]
    somme_prot = 0
    for sequence, scores_one_seq in tqdm(scores.items()):
        for indice, one_map in enumerate(scores_one_seq):
            nb_res = nb_res_ninety_percent(one_map)
            somme_nb_interet[indice] += nb_res
            somme_prop_interest[indice] += nb_res / one_map.shape[0]
            list_all_dist[indice].append(nb_res)
        somme_prot += 1
    mean_nb_interet_per_head = somme_nb_interet / somme_prot
    mean_prop_interet_per_head = somme_prop_interest / somme_prot

    plt.hist(mean_prop_interet_per_head)
    plt.show()

    plt.hist(mean_nb_interet_per_head)
    plt.show()

    for k in range(480):
        plt.hist(list_all_dist[k], alpha=0.2)
    plt.show()

    for k in range(480):
        if np.mean(list_all_dist[k]) < 50:
            plt.hist(list_all_dist[k], alpha=0.2)
    plt.show()

    standar_dev = np.std(list_all_dist, axis=1)
    print(standar_dev.shape)
    plt.hist(standar_dev)
    plt.show()
