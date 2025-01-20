from os import sep
import networkx as nx
import torch
from tqdm import tqdm
import csv
import pickle as pkl
import os
import numpy as np
from sklearn.metrics import average_precision_score
from analyse_how_to_agregate_attention_heads.analyse_attention_heads import (
    get_weight_head_combi,
    calc_prop_head,
)

from interpretability.utils import (
    get_attentions_map_simple,
    set_hook_to_get_attention_map,
    load_sequences,
    load_model_and_vocab,
)

# from multiprocessing import Pool
from multiprocessing.dummy import Pool


def construct_attn_graph(
    attentions_maps, prop_each_head_each_layer, nb_layer, nb_head, for_max_flow
):
    # print("prop_each_head_each_layer :", prop_each_head_each_layer.shape)
    len_seq = attentions_maps.shape[1]
    # print("Len seq :", len_seq)
    weigthed_head_per_layer = []
    for l in range(nb_layer):
        tmp = None
        for h in range(nb_head):
            w_vec = attentions_maps[l * nb_head + h] * prop_each_head_each_layer[l][h]
            if tmp is None:
                tmp = w_vec
            else:
                tmp += w_vec
        weigthed_head_per_layer.append(tmp)
    weigthed_head_per_layer = np.array(weigthed_head_per_layer)

    ident_matrix = np.identity(len_seq)
    zeros_matrix = np.zeros((len_seq, len_seq))

    list_2d_block = []
    for l in range(nb_layer):
        tmp = [zeros_matrix for _ in range(nb_layer)]
        tmp[l] = weigthed_head_per_layer[l]
        if l != nb_layer - 1:
            tmp[l + 1] = ident_matrix
        list_2d_block.append(tmp)

    adjacency_matrix = np.block(list_2d_block)
    if for_max_flow:
        adjacency_matrix = add_node_zero(adjacency_matrix, len_seq)

    # With networkX
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph())

    return G


def add_node_zero(adjacency_matrix, len_seq):
    nb_row, nb_col = adjacency_matrix.shape
    one_line = np.zeros((nb_col))
    one_line[:len_seq] = 1
    adjacency_matrix = np.vstack((one_line, adjacency_matrix))
    nb_row, nb_col = adjacency_matrix.shape
    one_col = np.zeros((nb_row, 1))
    adjacency_matrix = np.hstack((one_col, adjacency_matrix))
    return adjacency_matrix


def calc_page_rank_from_g(G, len_seq, nb_layer):
    personalization_pagerank = {k: 1 for k in range(len_seq)}

    dumping_factor = 0.85
    nb_noeud = G.number_of_nodes()  # nb_layer * len_seq
    noeud_de_fin = [k for k in range(nb_noeud - len_seq, nb_noeud)]

    pr = nx.pagerank(
        G,
        dumping_factor,
        max_iter=100,  # 600
        personalization=personalization_pagerank,
        tol=1e-6,  # 1e-10
    )

    weight_res = [pr[q] for q in noeud_de_fin]

    return weight_res


def calc_maxflow_from_g(G, len_seq, nb_layer):
    nb_noeud = G.number_of_nodes()
    noeud_de_fin = [k for k in range(nb_noeud - len_seq, nb_noeud)]

    all_flow = []
    for output_node in noeud_de_fin:
        flow_value, _ = nx.maximum_flow(G, 0, output_node, capacity="weight")
        all_flow.append(flow_value)

    return all_flow


def create_result_dict(attentions_map, nb_layer, nb_head, method):
    list_head_res = []
    for num_layer in range(nb_layer):
        for num_head in range(nb_head):
            attn_map_head_i = attentions_map[num_layer * nb_head + num_head]
            score_residue = result_one_head(attn_map_head_i, method)
            list_head_res.append(score_residue)
    return np.array(list_head_res)


def create_score_vector(
    attentions_map,
    nb_layer,
    nb_head,
    prop_each_head_each_layer,
    method,
    filter_threashold=1e-3,
):
    basic_method = [
        "max_follow_by_max_order1",
        "max_follow_by_mean_order1",
        "mean_follow_by_max_order1",
        "mean_follow_by_mean_order1",
        "max_follow_by_max_order2",
        "max_follow_by_mean_order2",
        "mean_follow_by_max_order2",
        "mean_follow_by_mean_order2",
        "max_follow_by_max_order3",
        "max_follow_by_mean_order3",
        "mean_follow_by_max_order3",
        "mean_follow_by_mean_order3",
        "max_follow_by_max_order4",
        "max_follow_by_mean_order4",
        "mean_follow_by_max_order4",
        "mean_follow_by_mean_order4",
    ]
    len_seq = attentions_map.shape[1]
    if method == "weighted_pageRank_total":
        attn_graph = construct_attn_graph(
            attentions_map,
            prop_each_head_each_layer,
            nb_layer,
            nb_head,
            for_max_flow=False,
        )

        scores = calc_page_rank_from_g(attn_graph, len_seq, nb_layer)
    elif method == "flowmax_total":
        attn_graph = construct_attn_graph(
            attentions_map,
            prop_each_head_each_layer,
            nb_layer,
            nb_head,
            for_max_flow=True,
        )

        scores = calc_maxflow_from_g(attn_graph, len_seq, nb_layer)
    elif "weighted_pageRank_each_head_follow_by_" in method:
        type_agreg = method.split("_follow_by_")[-1]
        all_scores = []
        for head in attentions_map:
            # head = add_node_zero(head, len_seq)
            G = nx.from_numpy_matrix(head, create_using=nx.DiGraph())
            scores_one_head = calc_page_rank_from_g(G, len_seq, nb_layer)
            all_scores.append(scores_one_head)
        all_scores = np.array(all_scores)

        if type_agreg == "mean":
            all_scores = all_scores.mean(axis=0)
        elif type_agreg == "max":
            all_scores = all_scores.max(axis=0)
        else:
            raise RuntimeError("Agregation type not define")
        scores = all_scores

    elif "flowmax_each_head_follow_by_" in method:
        type_agreg = method.split("_follow_by_")[-1]
        all_scores = []
        for head in attentions_map:
            head = add_node_zero(head, len_seq)
            # Filter head
            head[head < filter_threashold] = 0
            G = nx.from_numpy_matrix(head, create_using=nx.DiGraph())
            scores_one_head = calc_maxflow_from_g(G, len_seq, nb_layer)
            all_scores.append(scores_one_head)
        all_scores = np.array(all_scores)
        if type_agreg == "mean":
            all_scores = all_scores.mean(axis=0)
        elif type_agreg == "max":
            all_scores = all_scores.max(axis=0)
        else:
            raise RuntimeError("Agregation type not define")
        scores = all_scores
    elif method in basic_method:
        # "max_follow_by_max_order1"
        # tab=["max","max_order1"]
        first_agreg = method.split("_follow_by_")[0]
        second_agreg = method.split("_follow_by_")[1].split("_")[0]
        order = method.split("_follow_by_")[1].split("_")[1]

        if order == "order1":
            if first_agreg == "mean":
                scores = attentions_map.mean(axis=1)
            elif first_agreg == "max":
                scores = attentions_map.max(axis=1)
            else:
                raise RuntimeError("Agregation type not define")
        elif order == "order2" or order == "order3":
            if first_agreg == "mean":
                scores = attentions_map.mean(axis=0)
            elif first_agreg == "max":
                scores = attentions_map.max(axis=0)
            else:
                raise RuntimeError("Agregation type not define")
        elif order == "order4":
            if first_agreg == "mean":
                scores = attentions_map.mean(axis=2)
            elif first_agreg == "max":
                scores = attentions_map.max(axis=2)
            else:
                raise RuntimeError("Agregation type not define")
        else:
            raise RuntimeError("Order unknown")

        # assert scores.shape == (nb_head * nb_layer, len_seq)

        if order == "order1" or order == "order4":
            if second_agreg == "mean":
                scores = scores.mean(axis=0)
            elif second_agreg == "max":
                scores = scores.max(axis=0)
            else:
                raise RuntimeError("Agregation type not define")
        elif order == "order2":
            if second_agreg == "mean":
                scores = scores.mean(axis=0)
            elif second_agreg == "max":
                scores = scores.max(axis=0)
            else:
                raise RuntimeError("Agregation type not define")
        elif order == "order3":
            if second_agreg == "mean":
                scores = scores.mean(axis=1)
            elif second_agreg == "max":
                scores = scores.max(axis=1)
            else:
                raise RuntimeError("Agregation type not define")

        else:
            raise RuntimeError("Order unknown")

    else:
        raise RuntimeError("Not implemented yet")

    # We delete attention on cls token here
    scores = scores[1:]
    # assert len(scores) == attentions_map.shape[1] - 1

    return np.array(scores)


"""
Old function
def calc_list_res(attentions_map, nb_layer, nb_head, list_method):
    list_res = []
    for method in list_method:
        dico_res_method_i = create_result_dict(
            attentions_map, nb_layer, nb_head, method=method
        )
        list_res.append(dico_res_method_i)
    return list_res
"""


def score_on_residue_with_attn_map_multiprocess(
    path_model,
    path_vocab,
    which_dataset,
    nb_seq,
    method,
    one_output_vec=False,
    n_job=1,  # 10
):
    print("We start the method ", method)

    if method == "somme_col":
        folder_save = "data/residues_of_interest/Raw_attention_sum_col/"
    elif method == "attn_on_cls":
        folder_save = "data/residues_of_interest/Raw_attention_cls/"
    elif "_follow_by_" in method or "flowmax" in method or "pageRank" in method:
        folder_save = "data/residues_of_interest/" + method + "/"
    else:
        raise RuntimeError(
            "Don't have folder for this for now, you can create it and add an elif"
        )
    # print("folder_save :", folder_save)
    extract_name = "_".join(path_model.split("/")[2:4])
    name_output_file = "score_with_" + extract_name + "_on_dataset_" + which_dataset

    # We load the sequences
    list_sequences, list_labels = load_sequences(
        which_dataset, nb_seq, with_label=True, with_roles=False
    )
    generic_pre = [path_model, path_vocab, which_dataset, nb_seq]
    generic_post = [method, one_output_vec]
    if n_job > 1:
        # We cute sequences in n_job lists
        size_part = int(len(list_sequences) / n_job) + 1
        all_list_sequences = []
        all_list_labels = []
        for num_job in range(n_job):
            seq_select = list_sequences[num_job * size_part : (num_job + 1) * size_part]
            label_select = list_labels[num_job * size_part : (num_job + 1) * size_part]
            all_list_sequences.append(seq_select)
            all_list_labels.append(label_select)

        data_prep = [
            generic_pre + [l_seq, l_lab] + generic_post
            for l_seq, l_lab in zip(all_list_sequences, all_list_labels)
        ]
        print(len(data_prep))
        p = Pool(processes=n_job)
        output_all_procs = p.map(score_on_residue_with_attn_map, data_prep)
        print("FINIT")
        p.close()
    else:
        output_all_procs = [
            score_on_residue_with_attn_map(
                generic_pre + [list_sequences, list_labels] + generic_post
            )
        ]

    total_dict_scores = dict()
    total_dict_labels = dict()
    for res_one_process in output_all_procs:
        total_dict_scores = {**total_dict_scores, **res_one_process[0]}
        total_dict_labels = {**total_dict_labels, **res_one_process[1]}

    results = [total_dict_scores, total_dict_labels]
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)
    torch.save(results, open(folder_save + name_output_file + ".pkl", "wb"))


@torch.no_grad()
def score_on_residue_with_attn_map(all_args):
    """
    Input :
    which_dataset = "binding_site" or "catalytic_site"
    possible  method with 480 scores per seq :  "simetrize_sum_col",
                        "somme_col_with_seuil_0.3",
                        "somme_col",
                        "mean_col",
                        "maximum_col",
                        "somme_row",
                        "attn_on_cls",
                        "somme_normalize_by_seq_size",
                        "nb_attn_with_seuil_0.001"

            method with one scores per seq :
                    "max_follow_by_max",
                    "max_follow_by_mean",
                    "mean_follow_by_max",
                    "mean_follow_by_mean",
                    "weighted_pageRank_total",
                    "weighted_pageRank_each_head_follow_by_mean",
                    "weighted_pageRank_each_head_follow_by_mean",
                    "flowmax_total",
                    "flowmax_each_head_follow_by_max",
                    "flowmax_each_head_follow_by_mean",

    """
    (
        path_model,
        path_vocab,
        which_dataset,
        nb_seq,
        list_sequences,
        list_labels,
        method,
        one_output_vec,
    ) = all_args
    max_seq_len = 1024
    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################

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

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################
    compteur = 0

    dico_scores = dict()
    dico_labels = dict()

    # We calc the metric on all the dev set
    for sequence, labels in tqdm(
        zip(list_sequences, list_labels), total=len(list_sequences)
    ):
        labels = np.array(labels[:max_seq_len])
        sequence = sequence[:max_seq_len]
        # print("len seq :", len(sequence))

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

        if one_output_vec:
            list_res = create_score_vector(
                attentions_map,
                nb_layer,
                nb_head,
                prop_each_head_each_layer,
                method=method,
            )
        else:
            list_res = create_result_dict(
                attentions_map, nb_layer, nb_head, method=method
            )
        # print(list_res.shape)
        dico_scores[sequence] = list_res
        dico_labels[sequence] = labels

        compteur += 1

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

    results = [dico_scores, dico_labels]

    return results


def get_list_dist(
    attentions_map, nb_layer, nb_head, with_cls, seperate_head, ninety_pourcent=False
):
    attentions_map_here = attentions_map.copy()
    if seperate_head:
        # We suppose 480 head, dimension of ProtBert_BFD, we can get size from attention_map
        all_dists = np.zeros((480, 1025))
        prop_nb_attn_90_pourcent = np.zeros((480))
    else:
        all_dists = np.zeros((1025))
    for num_layer in range(nb_layer):
        for num_head in range(nb_head):
            attn_map_head_i = attentions_map_here[num_layer * nb_head + num_head]

            if with_cls is False:
                # We forget the CLS token
                attn_map_head_i = attn_map_head_i[1:]
            if ninety_pourcent:
                tmp_prop = []
                # We take 90 pourcent of the weights
                for j in range(len(attn_map_head_i)):
                    ind_sort = np.flip(np.argsort(attn_map_head_i[j]))
                    somme = 0
                    k = 0
                    last_attn_value = None
                    while somme < 0.9:
                        somme += attn_map_head_i[j, ind_sort[k]]
                        k += 1
                        last_attn_value = attn_map_head_i[j, ind_sort[k]]
                    cal_threashold = last_attn_value
                    attn_map_head_i[j, attn_map_head_i[j] < cal_threashold] = 0
                    tmp_prop.append(
                        (attn_map_head_i[j] > cal_threashold).sum()
                        / len(attn_map_head_i[j])
                    )
                prop_nb_attn_90_pourcent[num_layer * nb_head + num_head] = np.mean(
                    tmp_prop
                )
            else:
                # We take only big attention
                attn_map_head_i[attn_map_head_i < 0.1] = 0
            ind_attention = np.nonzero(attn_map_head_i)
            distances = list(np.abs(ind_attention[0] - ind_attention[1]))
            if seperate_head:
                all_dists[num_layer + num_head * nb_head, distances] += 1
            else:
                all_dists[distances] += 1
    if ninety_pourcent:
        return np.array(all_dists, dtype=np.int32), prop_nb_attn_90_pourcent
    else:
        return np.array(all_dists, dtype=np.int32)


@torch.no_grad()
def distance_attention_on_attn_map(path_model, path_vocab, which_dataset, nb_seq):
    """
    Input :
    which_dataset = "binding_site" or "catalytic_site"
    """
    max_seq_len = 1024
    folder_save = "data/distribution_attentions_length/"
    extract_name = "_".join(path_model.split("/")[2:4])
    name_output_file = (
        "all_attentions_distances_"
        + extract_name
        + "_on_dataset_"
        + which_dataset
        + "_on_"
        + str(nb_seq)
        + "_sequence"
    )

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences, list_labels, _ = load_sequences(
        which_dataset, nb_seq, with_label=True, with_roles=True
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

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################
    compteur = 0

    all_distances_with_cls = np.zeros((max_seq_len + 1))
    all_distances_without_cls = np.zeros((max_seq_len + 1))
    all_distance_per_head = np.zeros((480, max_seq_len + 1))
    all_distance_per_head_90_pourcent_of_attention = np.zeros((480, max_seq_len + 1))

    all_nb_prop_attn = np.zeros((480))

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

        merge_distances_cls = get_list_dist(
            attentions_map, nb_layer, nb_head, with_cls=True, seperate_head=False
        )
        all_distances_with_cls += merge_distances_cls

        merge_distances_without_cls = get_list_dist(
            attentions_map, nb_layer, nb_head, with_cls=False, seperate_head=False
        )
        all_distances_without_cls += merge_distances_without_cls

        distances_per_head_without_cls = get_list_dist(
            attentions_map, nb_layer, nb_head, with_cls=False, seperate_head=True
        )
        all_distance_per_head += distances_per_head_without_cls

        distances_per_head_wit_cls_90p, prop_nb_attn_90_pourcent = get_list_dist(
            attentions_map,
            nb_layer,
            nb_head,
            with_cls=True,
            seperate_head=True,
            ninety_pourcent=True,
        )
        all_distance_per_head_90_pourcent_of_attention += distances_per_head_wit_cls_90p
        all_nb_prop_attn += prop_nb_attn_90_pourcent

        compteur += 1

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

    result = [
        all_distances_with_cls,
        all_distances_without_cls,
        all_distance_per_head,
        all_distance_per_head_90_pourcent_of_attention,
        all_nb_prop_attn,
    ]
    torch.save(result, open(folder_save + name_output_file + ".pkl", "wb"))


def construct_attn_graph_old(
    attentions_maps, prop_each_head_each_layer, nb_layer, nb_head
):
    print("prop_each_head_each_layer :", prop_each_head_each_layer.shape)
    len_seq = attentions_maps.shape[1]
    print("Len seq :", len_seq)
    array_PK = []
    for l in range(nb_layer):
        tmp_one_layer = []
        for h in range(nb_head):
            one_matrice_PK = []
            for n in range(len_seq):
                one_line = np.zeros((len_seq))
                one_line[n] = prop_each_head_each_layer[l][h]
                one_matrice_PK.append(one_line)
            tmp_one_layer.append(one_matrice_PK)
        array_PK.append(tmp_one_layer)
    array_PK = np.array(array_PK)

    assert array_PK.shape == (nb_layer, nb_head, len_seq, len_seq)

    all_matrice_P = None
    for l in range(nb_layer):
        one_matrice_p = []
        for h in range(nb_head):
            one_line_P = [array_PK[l][h] for _ in range(nb_head)]
            one_matrice_p.append(one_line_P)

        if all_matrice_P is None:
            all_matrice_P = np.expand_dims(np.block(one_matrice_p), axis=0)
        else:
            all_matrice_P = np.concatenate(
                (all_matrice_P, np.expand_dims(np.block(one_matrice_p), axis=0))
            )

    assert all_matrice_P.shape == (nb_layer, len_seq * nb_head, len_seq * nb_head)

    all_matrice_H = None
    zeros_matrix = np.zeros((len_seq, len_seq))
    for l in range(nb_layer):
        one_matrice_h = []
        for h in range(nb_head):
            one_line_H = []
            for h2 in range(nb_head):
                if h2 != h:
                    one_line_H.append(zeros_matrix)
                else:
                    one_line_H.append(attentions_maps[h + l * nb_head])
            one_matrice_h.append(one_line_H)
        if all_matrice_H is None:
            all_matrice_H = np.expand_dims(np.block(one_matrice_h), axis=0)
        else:
            all_matrice_H = np.concatenate(
                (all_matrice_H, np.expand_dims(np.block(one_matrice_h), axis=0))
            )

    assert all_matrice_H.shape == (nb_layer, len_seq * nb_head, len_seq * nb_head)

    list_block_adj_matrix = []
    big_zeros_matrix = np.zeros((len_seq * nb_head, len_seq * nb_head))
    for l in range(nb_layer):
        one_line = []
        for l2 in range(nb_layer):
            if l2 != l and l2 != (l + 1):
                one_line.append(big_zeros_matrix)
            elif l2 == l:
                one_line.append(all_matrice_H[l])
            else:
                one_line.append(all_matrice_P[l])
        list_block_adj_matrix.append(one_line)

    adjacency_matrix = np.block(list_block_adj_matrix)

    personalization_pagerank = {k: 1 for k in range(len_seq)}

    print(adjacency_matrix.shape)
    dumping_factor = 0.85
    nb_noeud = nb_layer * nb_head * len_seq
    noeud_de_fin = [k for k in range(nb_noeud - len_seq, nb_noeud)]

    """
    # With graphtool
    g = gt.Graph(directed=True)
    g.add_edge_list(np.transpose(adjacency_matrix.nonzero()))
    pr = pagerank(g, damping=dumping_factor)  # pers=noeud_de_fin

    """

    # With networkX
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph())
    print("Graph created")
    # nx.draw(G)  # with_labels=True
    # plt.show()
    pr = nx.pagerank(
        G,
        dumping_factor,
        max_iter=600,
        personalization=personalization_pagerank,
    )
    print(type(pr))

    weight_res = [pr[q] for q in noeud_de_fin]
    print("len_seq :", len_seq)
    print("noeud_de_fin :", len(noeud_de_fin))
    print("len weight res :", len(weight_res))
    # On enleve le poid du [CLS]
    weight_res = weight_res[:-1]
    print(weight_res)

    # maximum_flow(graph, 0, 1).flow_value


def result_one_head(attn_map_head_i, method):
    if method == "somme_col":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        score_residue = attn_map_head_i_without_cls.sum(axis=0)
    elif method == "somme_normalize_by_seq_size":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        seq_size = attn_map_head_i_without_cls.shape[0]
        score_residue = attn_map_head_i_without_cls.sum(axis=0) / seq_size
    elif method == "somme_col_with_seuil_0.3":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        attn_map_head_i_without_cls[attn_map_head_i_without_cls < 0.3] = 0
        score_residue = attn_map_head_i_without_cls.sum(axis=0)
    elif method == "mean_col":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        score_residue = attn_map_head_i_without_cls.mean(axis=0)
    elif method == "simetrize_mean_col":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        attn_map_head_i_without_cls = (
            attn_map_head_i_without_cls + attn_map_head_i_without_cls.T
        ) / 2
        score_residue = attn_map_head_i_without_cls.mean(axis=0)
    elif method == "simetrize_sum_col":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        attn_map_head_i_without_cls = (
            attn_map_head_i_without_cls + attn_map_head_i_without_cls.T
        ) / 2
        score_residue = attn_map_head_i_without_cls.sum(axis=0)
    elif method == "maximum_col":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        score_residue = attn_map_head_i_without_cls.max(axis=0)
    elif method == "attn_on_cls":
        score_residue = attn_map_head_i[0, 1:]
    elif method == "nb_attn_with_seuil_0.001":
        # We don't take the cls token
        attn_map_head_i_without_cls = attn_map_head_i[1:, 1:]
        attn_map_head_i_without_cls[attn_map_head_i_without_cls < 0.001] = 0
        # We count the number of attention that are superior to 0.3
        attn_map_head_i_without_cls[attn_map_head_i_without_cls >= 0.001] = 1
        score_residue = attn_map_head_i_without_cls.sum(axis=0)
    else:
        raise RuntimeError("Method does not exist")

    score_residue = score_residue.astype(np.float16)

    return list(score_residue)