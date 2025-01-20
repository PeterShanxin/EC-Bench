# import graph_tool as gt
# from graph_tool.centrality import pagerank
from interpretability.utils import (
    get_attentions_map_simple,
    get_weight_head_combi,
    set_hook_to_get_attention_map,
    load_sequences,
    load_model_and_vocab,
)
from tqdm import tqdm
import numpy as np
from analyse_how_to_agregate_attention_heads.repartition_agregated_attentions import (
    nb_res_ninety_percent,
)
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import networkx as nx

import torch


def calc_prop_head(combi_head, nb_head):
    weight_matrice_each_head = []
    for w_i in combi_head:
        longueur_entree = w_i.shape[0]
        taille_head = int(longueur_entree / nb_head)
        split_each_head = torch.split(w_i.T, taille_head, dim=0)
        list_sum = []
        for sp in split_each_head:
            list_sum.append(torch.norm(sp).item())
        list_sum = np.array(list_sum)
        list_sum = list_sum / list_sum.sum()
        weight_matrice_each_head.append(list_sum)
    weight_matrice_each_head = np.array(weight_matrice_each_head)

    """
    plt.matshow(weight_matrice_each_head)
    plt.title(
        "Vecteur normaliser par layer de la norme de frobenius des matrices de poids appliqué à chauqe tete d'attention"
    )
    plt.colorbar()
    plt.show()
    """

    return weight_matrice_each_head


@torch.no_grad()
def generate_attention_head_analysis(path_model, path_vocab, which_dataset, nb_seq):
    max_seq_len = 1024

    extract_name = "_".join(path_model.split("/")[2:4])

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences, list_labels = load_sequences(
        which_dataset, nb_seq, with_label=True, with_roles=False
    )
    model, vocab, pad_indice = load_model_and_vocab(path_model, path_vocab)

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################

    (
        activation,
        nb_layer,
        nb_head,
        list_hook,
        ancienne_fonciton_lib,
    ) = set_hook_to_get_attention_map(model)

    combi_head = get_weight_head_combi(model)
    prop_each_head_each_layer = calc_prop_head(combi_head, nb_head)
    print(prop_each_head_each_layer)
    print(prop_each_head_each_layer.shape)

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################

    list_all_dist = [[] for _ in range(480)]
    somme_prot = 0
    # We calc the metric on all the dev set
    for sequence, labels in tqdm(
        zip(list_sequences, list_labels), total=len(list_sequences)
    ):
        labels = np.array(labels[:max_seq_len])
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        model(input_batch)

        attentions_map = get_attentions_map_simple(
            activation,
            nb_layer,
            input_batch,
            pad_indice,
        )

        for num_attn_map, one_map in enumerate(attentions_map):
            # pagerank_test(attentions_map)
            for attn_row in one_map:
                nb_res = nb_res_ninety_percent(attn_row)
                list_all_dist[num_attn_map].append(nb_res)

    ##########################################################################
    # STEP 5 : We delete the hook and get the model back to the original state#
    ##########################################################################
    # We delete the hook from the model
    for hook in list_hook:
        hook.remove()
    # On rétablie la fonction correct dans la librairie pytorch
    torch.functional.multi_head_attention_forward = ancienne_fonciton_lib
    model = torch.nn.DataParallel(model)
    model = model.train()

    # We save the results

    mean_nb_res_each_head = []
    for num_head in range(480):
        mean_nb_res_each_head.append(np.mean(list_all_dist[num_head]))

    plt.hist(mean_nb_res_each_head)
    plt.show()

    for k in range(480):
        plt.hist(list_all_dist[k], alpha=0.2)
    plt.show()
