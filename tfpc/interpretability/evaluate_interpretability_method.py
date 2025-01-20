from interpretability.utils import load_sequences
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch
from os import walk
import os
from difflib import SequenceMatcher
import pickle as pkl
from tqdm import tqdm
import random
import pandas as pd
from scipy import stats
import logging
from utils.json_loader import load_json_into_pandas_dataframe
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from utils import prg
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from scipy.interpolate import make_interp_spline, BSpline
import copy


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def combine_dico_seq(dico_seq1, dico_seq2):
    dico_combine = dict()
    nb_not_in_the_two = 0
    for key in dico_seq1.keys():
        if key in dico_seq1.keys() and key in dico_seq2.keys():
            dico_combine[key] = dico_seq1[key] + dico_seq2[key]
        else:
            nb_not_in_the_two += 1
    logging.info(
        "We have %s sequence that are not present at least in one of the two method",
        nb_not_in_the_two,
    )
    return dico_combine


def compare_intepretability_methods(add_combi_with_best=True):
    base_path = "data/residues_of_interest/"
    # List of all method
    all_methods = os.listdir(base_path)
    all_methods = [a for a in all_methods if "." not in a]

    print(all_methods)
    method_that_need_training = []

    # Load catalytic datasets with correct labels
    list_sequences, list_labels = load_sequences(
        "catalytic_site", "all", with_label=True, with_roles=False
    )
    data = list(map(list, zip(list_sequences, list_labels)))

    # We define a sklearn spliter
    kf = KFold(n_splits=5)
    dico_AP_metrics = dict()
    dico_F1_metrics = dict()
    dico_score_correctness = dict()
    num_split = 0
    for ind_train, ind_test in kf.split(data):
        logging.info("We begin the %sst/th split of data", num_split + 1)
        train_data = [data[c] for c in ind_train]
        test_data = [data[c] for c in ind_test]

        # Load scores from different interpretability methods
        logging.info("We load the interpretability score from dump file")
        dico_scores = dict()
        for method in all_methods:
            dico_scores[method] = get_scores(base_path, method)

        # Specific calculation to select good attention from train dataset
        logging.info("We select which attention head to use")
        for method in ["Raw_attention_sum_col", "Raw_attention_cls"]:
            num_best_head = get_best_head(train_data, dico_scores[method])
            new_dict = dict()
            for sequence, scores in dico_scores[method].items():
                new_dict[sequence] = dico_scores[method][sequence][num_best_head]
            dico_scores[method] = new_dict

        # Calc correlation of the different methods
        logging.info("We calc correlation between each pair of methods")
        calc_correlation(dico_scores)

        # Calc dict with normalization
        logging.info("We calc dict with normalization")
        for method in all_methods:
            dico_scores[method + "_min_max_norm"] = normalize_all_dict(
                dico_scores[method], norm_type="min_max"
            )
            dico_scores[method + "_std_mean_norm"] = normalize_all_dict(
                dico_scores[method], norm_type="std_mean"
            )
            dico_scores[method + "_unit_length_norm1"] = normalize_all_dict(
                dico_scores[method], norm_type="unit_length_norm1"
            )
            dico_scores[method + "_unit_length_norm2"] = normalize_all_dict(
                dico_scores[method], norm_type="unit_length_norm2"
            )

        if add_combi_with_best:
            logging.info(
                "We combine the best method with all the other to test the combination predictive power"
            )
            for method in all_methods:
                dico_scores[
                    "combination_"
                    + "Raw_attention_sum_col"
                    + "_and_"
                    + method
                    + "_with_unit_length_norm1"
                ] = combine_dico_seq(
                    dico_scores["Raw_attention_sum_col_unit_length_norm1"],
                    dico_scores[method + "_unit_length_norm1"],
                )

        for method in tqdm(dico_scores.keys()):
            print("Method :", method)
            threshold = get_threshold(train_data, dico_scores[method])
            dico_correctness_test_seq = get_correct_score(
                test_data, dico_scores[method], threshold
            )

            if method not in dico_score_correctness.keys():
                dico_score_correctness[method] = dico_correctness_test_seq
            else:
                dico_score_correctness[method] = {
                    **dico_score_correctness[method],
                    **dico_correctness_test_seq,
                }
            # Get scores on test set
            # AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight
            Avreage_precision = get_AP(test_data, dico_scores[method])
            f1_scores = get_f1(test_data, dico_scores[method], threshold)
            if method not in dico_AP_metrics.keys():
                dico_AP_metrics[method] = []
                dico_F1_metrics[method] = []
            dico_AP_metrics[method].append(Avreage_precision)
            dico_F1_metrics[method].append(f1_scores)
        num_split += 1
        print(num_split, "/5 splits process")

    # Print the results of differents methods
    for method in dico_AP_metrics.keys():
        values_AP = dico_AP_metrics[method]
        value_F1 = dico_F1_metrics[method]
        print("For method : ", method)
        print("AP :", np.mean(values_AP), "+/-", 2 * np.std(values_AP))
        print("F1 :", np.mean(value_F1), "+/-", 2 * np.std(value_F1))
    fichier = open("dico_res_correctness.pkl", "wb")
    pkl.dump(dico_score_correctness, fichier)


def how_many_token_ninety_percent_attn(test_data, raw_attn_scores, numero_head):
    train_data = del_more_than_max_len(test_data)
    all_nb_token = []
    for seq, _ in train_data:
        scores = np.flip(np.sort(np.array(raw_attn_scores[seq][numero_head])))
        sum_scores = np.sum(scores)
        ninety_percent_scores = 0.9 * sum_scores
        somme = 0
        nb_token = 0
        while somme < ninety_percent_scores:
            somme += scores[nb_token]
            nb_token += 1
        all_nb_token.append(nb_token)

    return np.mean(all_nb_token)


def calc_metric_with_train(method, dico_scores_one_method, nb_split, data):
    kf = KFold(n_splits=nb_split)
    indices_split = kf.split(data)
    num_split = 0
    all_ap = []
    list_precision_gain = []
    list_recall_gain = []
    for ind_train, ind_test in indices_split:
        logging.info("We begin the %sst/th split of data", num_split + 1)
        train_data = [data[c] for c in ind_train]
        test_data = [data[c] for c in ind_test]

        # Specific calculation to select good attention from train dataset
        logging.info("We select which attention head to use")
        num_best_head = get_best_head(train_data, dico_scores_one_method)
        dico_scores_one_method_select_best = dict()
        for sequence, _ in dico_scores_one_method.items():
            dico_scores_one_method_select_best[sequence] = dico_scores_one_method[
                sequence
            ][num_best_head]

        # Get scores on test set
        # AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight
        (
            average_precision,
            AUPRGC,
            max_F_gain,
            precision_gain,
            recall_gain,
        ) = get_AP_and_AUPRGC(test_data, dico_scores_one_method_select_best)

        all_ap.append(average_precision)
        list_recall_gain.append(recall_gain)
        list_precision_gain.append(precision_gain)

        num_split += 1
        print(num_split, "/", nb_split, "splits process")
    return all_ap, list_recall_gain, list_precision_gain, AUPRGC, F_gain


def calc_metric_without_train(method, dico_scores_one_method, data):
    # Get scores on test set
    # AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight
    (
        average_precision,
        PRG_AUC,
        max_F_gain,
        precision_gain,
        recall_gain,
        precision,
        recall,
    ) = get_AP_and_AUPRGC(data, dico_scores_one_method)

    list_nb_token, list_token_f1 = get_token_f1(data, dico_scores_one_method)

    return (
        average_precision,
        recall_gain,
        precision_gain,
        list_nb_token,
        list_token_f1,
        PRG_AUC,
        max_F_gain,
        precision,
        recall,
    )


def curve_AP_in_respect_to_size_train():
    all_ap = []
    all_std_ap = []
    all_size_train = []
    all_methods = ["Raw_attention_cls"]
    for nb_split in range(2, 10, 1):
        mean_Ap_score, std_Ap_score, size_train = compare_intepretability_methods_V2(
            all_methods, nb_split
        )
        all_ap.append(mean_Ap_score)
        all_std_ap.append(std_Ap_score)
        all_size_train.append(size_train)
    all_std_ap = np.array(all_std_ap) * 2
    plt.errorbar(all_size_train, all_ap, yerr=all_std_ap)
    plt.show()


def get_best_wost_and_random(method_name, norm_type):

    apply_filter = False  # True  # False
    min_catalytic_site = 4
    max_len = 500

    which_dataset = "catalytic_site"
    base_path = "data/residues_of_interest/"
    # Load catalytic datasets with correct labels
    list_sequences, list_labels = load_sequences(
        which_dataset, "all", with_label=True, with_roles=False
    )

    list_sequences = [l[:1024] for l in list_sequences]
    list_labels = [l[:1024] for l in list_labels]

    scores = get_scores(base_path, method_name, which_dataset="catalytic_site")
    scores = normalize_all_dict(scores, norm_type=norm_type)

    # minimum_num_catalytic_res = 5  # 5

    # all_data = list(map(list, zip(list_sequences, list_labels)))

    best_PR_AUC = 0
    best_sequence = None
    best_label = None
    best_scores = None

    worst_PR_AUC = 100
    worst_sequence = None
    worst_label = None
    worst_scores = None

    list_score_PR_AUC = []
    list_scores = []
    list_catalytic_res = []
    order_list_sequences = []

    nb_bin = 10
    size_bin = 1025 / nb_bin
    performance_per_lenght = [[] for _ in range(nb_bin)]

    when_have_cystein_and_when_not = [[] for _ in range(2)]

    performance_depending_on_number_of_catalytic_site = [[] for _ in range(25)]

    somme_res_catalytique = 0
    somme_all_res = 0
    all_lenght = []

    for sequence, _ in scores.items():
        # Ignore this sequence because error in M-CSA database on it, linked sequence has changed in uniprot db
        if (
            sequence
            == "MAEMATATRLLGWRVASWRLRPPLAGFVSQRAHSLLPVDDAINGLSEEQRQLRQTMAKFLQEHLAPKAQEIDRSNEFKNLREFWKQLGNLGVLGITAPVQYGGSGLGYLEHVLVMEEISRASGAVGLSYGAHSNLCINQLVRNGNEAQKEKYLPKLISGEYIGALAMSEPNAGSDVVSMKLKAEKKGNHYILNGNKFWITNGPDADVLIVYAKTDLAAVPASRGITAFIVEKGMPGFSTSKKLDKLGMRGSNTCELIFEDCKIPAANILGHENKGVYVLMSGLDLERLVLAGGPLGLMQAVLDHTIPYLHVREAFGQKIGHFQLMQGKMADMYTRLMACRQYVYNVAKACDEGHCTAKDCAGVILYSAECATQVALDGIQCFGGNGYINDFPMGRFLRDAKLYEIGAGTSEVRRLVIGRAFNADFH"
        ):
            continue

        indice_of_label = list_sequences.index(sequence)

        this_label = list_labels[indice_of_label]
        if apply_filter and (
            np.sum(this_label) < min_catalytic_site or len(sequence) > max_len
        ):
            continue
        somme_all_res += len(sequence)
        all_lenght.append(len(sequence))
        indice_catalytic = np.where(np.array(this_label) == 1)[0]
        somme_res_catalytique += len(indice_catalytic)
        # print("this_label", indice_catalytic)
        have_cystein = False
        for ind in indice_catalytic:
            # print("ind :", ind)
            if sequence[ind] == "C":
                have_cystein += 1
        # On vérifie qu'il y a au moins un site catalytic sinon ça ne sert a rien
        if np.sum(this_label) == 0:
            continue
        data = list(map(list, zip([sequence], [this_label])))

        (
            _,
            _,
            _,
            _,
            _,
            PRG_AUC,
            max_F_gain,
        ) = calc_metric_without_train(method_name, scores, data)

        print("Number catalytic site :", len(indice_catalytic))
        performance_depending_on_number_of_catalytic_site[len(indice_catalytic)].append(
            PRG_AUC
        )
        which_bin = int(len(sequence) / size_bin)

        performance_per_lenght[which_bin].append(PRG_AUC)
        if have_cystein:
            when_have_cystein_and_when_not[1].append(PRG_AUC)
        else:
            when_have_cystein_and_when_not[0].append(PRG_AUC)

        list_score_PR_AUC.append(PRG_AUC)
        list_scores.append(scores[sequence])
        list_catalytic_res.append(this_label)
        order_list_sequences.append(sequence)
        if (
            PRG_AUC
            > best_PR_AUC
            # and np.sum(this_label) > minimum_num_catalytic_res
        ):
            best_PR_AUC = PRG_AUC
            best_sequence = sequence
            best_label = this_label
            best_scores = scores[sequence]
        if PRG_AUC < worst_PR_AUC:
            worst_PR_AUC = PRG_AUC
            worst_sequence = sequence
            worst_label = this_label
            worst_scores = scores[sequence]

    list_x = []
    list_y = []
    for ind, bin_content in enumerate(performance_per_lenght):
        middle = ((ind + 1) * size_bin + ind * size_bin) / 2
        list_x.append(middle)
        if len(bin_content) == 0:
            list_y.append(0)
        else:
            list_y.append(np.mean(bin_content))
    print("list_x :", list_x)
    print("list_y :", list_y)
    plt.bar(list_x, list_y, width=size_bin)
    plt.xlabel("Length of the enzyme")
    plt.ylabel("Mean PR_AUC of all proteins")
    plt.show()

    list_x = []
    list_y = []
    effectif = []
    for ind, bin_content in enumerate(
        performance_depending_on_number_of_catalytic_site
    ):
        list_x.append(ind)
        effectif.append(len(bin_content))
        if len(bin_content) == 0:
            list_y.append(0)
        else:
            list_y.append(np.mean(bin_content))

    print("list_x :", list_x)
    print("list_y :", list_y)
    print("Effectic", effectif)
    plt.bar(list_x, list_y, width=1)
    plt.xlabel("Number of catalytic site of the enzyme")
    plt.ylabel("Mean PR_AUC of all proteins")
    plt.show()

    plt.bar(
        [0, 1],
        [
            np.mean(when_have_cystein_and_when_not[0]),
            np.mean(when_have_cystein_and_when_not[1]),
        ],
        width=1,
    )
    plt.xlabel("Have at least one cystein (1) or not (0) as a catalytic residue")
    plt.ylabel("Mean PR_AUC of all proteins")
    plt.show()

    print("The best PR_AUC is on :")
    print("PR_AUC :", best_PR_AUC)
    print("sequence='", best_sequence, "'")
    print("cat_seq =", best_label)
    print("len seq :", len(best_sequence))
    print("Indice best label =", np.where(np.array(best_label) == 1))
    print("Nb catalytic site :", np.sum(best_label))
    print(
        "identity of label :",
        [best_sequence[ind] for ind in np.where(np.array(best_label) == 1)[0]],
    )
    print("scores =", list(best_scores))

    print("The worst PR_AUC is on :")
    print("PR_AUC :", worst_PR_AUC)
    print("sequence='", worst_sequence)
    print("cat_seq =", worst_label)
    print("len seq :", len(worst_sequence))
    print("Indice worst label =", np.where(np.array(worst_label) == 1))
    print("Nb catalytic site :", np.sum(worst_label))
    print(
        "identity of label :",
        [worst_sequence[ind] for ind in np.where(np.array(worst_label) == 1)[0]],
    )
    print("scores =", list(worst_scores))

    print(
        "Il y a", somme_res_catalytique / somme_all_res * 100, "% de site catalytique"
    )
    print("Longueur moyenne enz M-CSA :", np.mean(all_lenght))

    # TOP 10
    list_score_pr_auc = np.array(list_score_PR_AUC)
    argsort = np.argsort(list_score_pr_auc)
    argsort = np.flip(argsort)

    print("Top 10 Pr_AUC :")
    for ind in argsort[:10]:
        print("PR_AUC :", list_score_pr_auc[ind])
        print("sequence='", order_list_sequences[ind], "'")
        # print("cat_seq =", list_catalytic_res[ind])
        print("len cata :", len(list_catalytic_res[ind]))
        print("len seq :", len(order_list_sequences[ind]))
        print("Nb catalytic site :", np.sum(list_catalytic_res[ind]))
        print(
            "identity of label :",
            [
                order_list_sequences[ind][ind2]
                for ind2 in np.where(np.array(list_catalytic_res[ind]) == 1)[0]
            ],
        )
        # print("scores =", list(list_scores[ind]))


def compare_intepretability_methods_V2(
    all_methods,
    which_dataset="catalytic_site",
    calc_normalize=True,
    simplify_label="Y",
    supplement_data=False,
):
    if simplify_label == "Y" or simplify_label == "y":
        simplify_label = True
    else:
        simplify_label = False
    all_normalization = [
        "unit_length_norm1",
        "unit_length_norm2",
        "std_mean",
        "min_max",
    ]
    dico_associated_best_norm = {
        "random": "Unknown",
        "random_uniform": "Unknown",
        "Rollout": "unit_length_norm1",
        "max_follow_by_max_order1": "unit_length_norm2",
        "max_follow_by_mean_order1": "unit_length_norm2",
        "mean_follow_by_max_order1": "unit_length_norm2",
        "mean_follow_by_mean_order1": "unit_length_norm2",
        "max_follow_by_max_order2": "unit_length_norm2",
        "max_follow_by_mean_order2": "unit_length_norm2",
        "mean_follow_by_max_order2": "unit_length_norm2",
        "mean_follow_by_mean_order2": "unit_length_norm2",
        "max_follow_by_max_order3": "unit_length_norm2",
        "max_follow_by_mean_order3": "unit_length_norm2",
        "mean_follow_by_max_order3": "unit_length_norm2",
        "mean_follow_by_mean_order3": "unit_length_norm2",
        "max_follow_by_max_order4": "unit_length_norm2",
        "max_follow_by_mean_order4": "unit_length_norm2",
        "mean_follow_by_max_order4": "unit_length_norm2",
        "mean_follow_by_mean_order4": "unit_length_norm2",
        "InputXGrad": "unit_length_norm1",
        "Gradients": "unit_length_norm1",
        "integrated_grad": "unit_length_norm1",
        "LIME_with_5000_corupt_seq_with_m_replacment_character": "unit_length_norm2",
        "Attn_last_layer": "None",
        "GradCam": "None",
        "LRP_rollout_sum_col": "std_mean",
        "LRP_rollout_cls": "unit_length_norm1",
    }
    dico_corresp_human_readable_labels = {
        "random": "random",
        "random_uniform": "random_uniform",
        "Rollout_unit_length_norm1": "Rollout",
        "mean_follow_by_mean_order1_unit_length_norm2": "Attention agregation",
        "InputXGrad_unit_length_norm1": "Input X Gradient",
        "Gradients_unit_length_norm1": "Gradient",
        "integrated_grad_unit_length_norm1": "Integrated Gradient",
        "LIME_with_5000_corupt_seq_with_m_replacment_character_unit_length_norm2": "LIME",
        "Attn_last_layer": "Attention last layer",
        "GradCam": "GradCam",
        "LRP_rollout_cls_unit_length_norm1": "Transformer LRP",
    }

    dico_corresp_human_readable_labels_corr_matrice = {
        "random": "random",
        "random_uniform": "random_uniform",
        "Rollout": "Rollout",
        "mean_follow_by_mean_order1": "Attention agregation",
        "InputXGrad": "Input X Gradient",
        "Gradients": "Gradient",
        "integrated_grad": "Integrated Gradient",
        "LIME_with_5000_corupt_seq_with_m_replacment_character": "LIME",
        "Attn_last_layer": "Attention last layer",
        "GradCam": "GradCam",
        "LRP_rollout_cls": "Transformer LRP",
    }

    #'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    dico_method_linetype = {
        "random": " ",
        "random_uniform": " ",
        "Rollout_unit_length_norm1": "solid",
        "mean_follow_by_mean_order1_unit_length_norm2": "solid",
        "InputXGrad_unit_length_norm1": "dashed",
        "Gradients_unit_length_norm1": "dotted",
        "integrated_grad_unit_length_norm1": "dashdot",
        "LIME_with_5000_corupt_seq_with_m_replacment_character_unit_length_norm2": "solid",
        "Attn_last_layer": "solid",
        "GradCam": "solid",
        "LRP_rollout_cls_unit_length_norm1": "solid",
        "max_follow_by_max_order4_unit_length_norm2": "solid",
        "max_follow_by_mean_order4_unit_length_norm2": "solid",
        "mean_follow_by_mean_order4_unit_length_norm2": "solid",
        "mean_follow_by_max_order4_unit_length_norm2": "solid",
    }
    dico_method_linewidth = {
        "random": "0",
        "random_uniform": "0",
        "Rollout_unit_length_norm1": 2,
        "mean_follow_by_mean_order1_unit_length_norm2": 2,
        "InputXGrad_unit_length_norm1": 2,
        "Gradients_unit_length_norm1": 2,
        "integrated_grad_unit_length_norm1": 2,
        "LIME_with_5000_corupt_seq_with_m_replacment_character_unit_length_norm2": 2,
        "Attn_last_layer": 2,
        "GradCam": 2,
        "LRP_rollout_cls_unit_length_norm1": 2,
        "max_follow_by_max_order4_unit_length_norm2": 2,
        "max_follow_by_mean_order4_unit_length_norm2": 2,
        "mean_follow_by_mean_order4_unit_length_norm2": 2,
        "mean_follow_by_max_order4_unit_length_norm2": 2,
    }

    dico_method_marker = {
        "random": "",
        "random_uniform": "",
        "Rollout_unit_length_norm1": ">",
        "mean_follow_by_mean_order1_unit_length_norm2": "o",
        "InputXGrad_unit_length_norm1": "",
        "Gradients_unit_length_norm1": "",
        "integrated_grad_unit_length_norm1": "",
        "LIME_with_5000_corupt_seq_with_m_replacment_character_unit_length_norm2": "x",
        "Attn_last_layer": "s",
        "GradCam": "",
        "LRP_rollout_cls_unit_length_norm1": "^",
        "max_follow_by_max_order4_unit_length_norm2": "o",
        "max_follow_by_mean_order4_unit_length_norm2": "o",
        "mean_follow_by_mean_order4_unit_length_norm2": "o",
        "mean_follow_by_max_order4_unit_length_norm2": "o",
    }

    possible_agregation = ["mean", "max"]
    list_possible = [
        p1 + "_follow_by_" + p2 + "_order" + str(k)
        for p1 in possible_agregation
        for p2 in possible_agregation
        for k in range(1, 4)
    ]
    list_possible += [
        p + "_" + norm for p in list_possible for norm in all_normalization
    ]

    for pos in list_possible:
        dico_corresp_human_readable_labels_corr_matrice[pos] = "Attention agregation"
        dico_corresp_human_readable_labels[pos] = "Attention agregation"
        dico_method_linetype[pos] = "solid"
        dico_method_linewidth[pos] = 2
        dico_method_marker[pos] = "o"

    base_path = "data/residues_of_interest/"

    print(all_methods)
    method_that_need_training = ["Raw_attention_cls", "Raw_attention_sum_col"]

    # Load catalytic datasets with correct labels
    list_sequences, list_labels = load_sequences(
        which_dataset, "all", with_label=True, with_roles=False
    )
    # Crop seq to 1024 to have uniformity between methods
    list_labels = [l[:1024] for l in list_labels]
    list_sequences = [l[:1024] for l in list_sequences]

    freq_label = sum([sum(l) for l in list_labels]) / sum([len(l) for l in list_labels])
    print("Label frequency is", freq_label * 100)
    data = list(map(list, zip(list_sequences, list_labels)))
    random.shuffle(data)

    # Load scores from different interpretability methods
    logging.info("We load the interpretability score from dump file")
    dico_scores = dict()
    for method in all_methods:
        dico_scores[method] = get_scores(
            base_path, method, which_dataset="catalytic_site"
        )

        for keys in dico_scores[method]:
            dico_scores[method][keys] = list(dico_scores[method][keys])

    # dico_random = dict()
    # for seq, scores in dico_scores["mean_follow_by_mean_order1"].items():
    #     dico_random[seq] = copy.copy(scores)
    #     np.random.shuffle(dico_random[seq])
    # dico_scores["random"] = dico_random

    # dico_random_uniform = dict()
    # for seq, scores in dico_scores["mean_follow_by_mean_order1"].items():
    #     dico_random_uniform[seq] = np.random.random(len(scores))
    # dico_scores["random_uniform"] = dico_random_uniform

    # Crop seq to 1024 to have uniformity between methods
    for method_name, scores in dico_scores.items():
        new_dico = dict()
        for seq, value in dico_scores[method_name].items():
            new_seq = seq[:1024]
            new_value = value[:1024]
            new_dico[new_seq] = new_value
        dico_scores[method_name] = copy.copy(new_dico)

    # Create correlation of scores
    all_seqs = list(dico_scores[list(dico_scores.keys())[0]].keys())
    dico_faltten = dict()
    for method_name, scores in dico_scores.items():
        flatten_scores = np.concatenate([scores[s] for s in all_seqs])
        dico_faltten[method_name] = flatten_scores

    all_labels = list(dico_faltten.keys())
    corr_mat = np.corrcoef(list(dico_faltten.values()))

    corr = (corr_mat + corr_mat.T) / 2  # made symmetric
    np.fill_diagonal(corr, 1)
    dissimilarity = 1 - np.abs(corr)
    hierarchy = linkage(squareform(dissimilarity), method="average")
    labels = fcluster(hierarchy, 0.5, criterion="distance")

    index_order = []
    for k in range(1, np.max(labels) + 1):
        indices_of_this_cluster = list(np.where(labels == k)[0])
        index_order += indices_of_this_cluster

    corr_mat = corr_mat[index_order]
    corr_mat = corr_mat[:, index_order]
    all_labels = [all_labels[i] for i in index_order]
    if simplify_label:
        all_labels = [
            dico_corresp_human_readable_labels_corr_matrice[l] for l in all_labels
        ]
    if supplement_data:
        plt.rcParams.update({"font.size": 10})
        plt.matshow(corr_mat)

        # plt.gca().tick_params(labeltop=False)
        # plt.gca().tick_params(labelbottom=True)

        for (i, j), z in np.ndenumerate(corr_mat):
            if "{:0.2f}".format(z) == "-0.00":
                z = 0
            if z > 0.5:
                plt.text(j, i, "{:0.2f}".format(z), ha="center", va="center")
            else:
                plt.text(
                    j, i, "{:0.2f}".format(z), ha="center", va="center", color="white"
                )
        plt.colorbar()
        # plt.margins(3, 3, tight=None)
        plt.yticks(list(range(len(all_labels))), all_labels)
        plt.xticks(list(range(len(all_labels))), all_labels, rotation="vertical")
        # plt.subplots_adjust(bottom=-100.5, top=600.5)
        # plt.subplots_adjust(top=0.85)
        plt.tight_layout()

        # plt.savefig("fig.svg", bbox_inches="tight", format="svg")
        plt.show()

    # Add method with normalization
    logging.info("We calc dict with normalization")
    for method in all_methods:
        if method in method_that_need_training:
            method_that_need_training.append(method + "_min_max_norm")
            method_that_need_training.append(method + "_std_mean_norm")
            method_that_need_training.append(method + "_unit_length_norm1")
            method_that_need_training.append(method + "_unit_length_norm2")
        best_norm_for_this_method = dico_associated_best_norm[method]
        if (
            best_norm_for_this_method != "None"
            and best_norm_for_this_method != "Unknown"
        ):
            dico_scores[method + "_" + best_norm_for_this_method] = normalize_all_dict(
                dico_scores[method], norm_type=best_norm_for_this_method
            )
            dico_scores.pop(method, None)
        elif best_norm_for_this_method == "Unknown":
            for normalization in all_normalization:
                dico_scores[method + "_" + normalization] = normalize_all_dict(
                    dico_scores[method], norm_type=normalization
                )

    dico_AP_metrics = dict()
    dico_recall_precision_gain = dict()
    dico_recall_precision = dict()
    dico_token_f1 = dict()
    dico_AUPRGC = dict()
    dico_max_F1GAIN = dict()
    for method in dico_scores.keys():
        logging.info("We treat the method %s", method)
        (
            Avreage_precision,
            list_recall_gain,
            list_precision_gain,
            list_nb_token,
            list_token_f1,
            AUPRGC,
            max_F_gain,
            list_precision,
            list_recall,
        ) = calc_metric_without_train(method, dico_scores[method], data)
        print("Avreage_precision :", Avreage_precision)
        dico_AP_metrics[method] = Avreage_precision
        dico_recall_precision_gain[method] = [list_recall_gain, list_precision_gain]
        dico_recall_precision[method] = [list_recall, list_precision]
        dico_token_f1[method] = [list_nb_token, list_token_f1]
        dico_AUPRGC[method] = AUPRGC
        dico_max_F1GAIN[method] = max_F_gain

    plt.rcParams.update({"font.size": 16})

    plt.gca().xaxis.label.set_color("black")  # setting up X-axis label color to yellow
    plt.gca().yaxis.label.set_color("black")  # setting up Y-axis label color to blue

    plt.gca().tick_params(
        axis="x", colors="#696969"
    )  # setting up X-axis tick color to red
    plt.gca().tick_params(
        axis="y", colors="#696969"
    )  # setting up Y-axis tick color to black

    plt.gca().spines["left"].set_color("#696969")  # setting up Y-axis tick color to red
    plt.gca().spines["top"].set_color(
        "#696969"
    )  # setting up above X-axis tick color to red

    for method, (list_nb_token, list_token_f1) in dico_token_f1.items():
        if simplify_label:
            plt.plot(
                list_nb_token,
                list_token_f1,
                label=dico_corresp_human_readable_labels[method],
                linewidth=dico_method_linewidth[method],
                linestyle=dico_method_linetype[method],
                marker=dico_method_marker[method],
            )
        else:
            plt.plot(
                list_nb_token,
                list_token_f1,
                label=method,
                linewidth=dico_method_linewidth[method],
                linestyle=dico_method_linetype[method],
                marker=dico_method_marker[method],
            )

    plt.xlabel("# of Tokens")
    plt.ylabel("Mean token-f1 score")
    # plt.title("Mean f1 score per fixed number of tokens")
    # Only show ticks on the left and bottom spines
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.legend()
    plt.show()

    plt.rcParams.update({"font.size": 16})

    plt.gca().xaxis.label.set_color("black")  # setting up X-axis label color to yellow
    plt.gca().yaxis.label.set_color("black")  # setting up Y-axis label color to blue

    plt.gca().tick_params(
        axis="x", colors="#696969"
    )  # setting up X-axis tick color to red
    plt.gca().tick_params(
        axis="y", colors="#696969"
    )  # setting up Y-axis tick color to black

    plt.gca().spines["left"].set_color("#696969")  # setting up Y-axis tick color to red
    plt.gca().spines["top"].set_color(
        "#696969"
    )  # setting up above X-axis tick color to red

    for method, (
        recall_gain,
        precision_gain,
    ) in (
        dico_recall_precision.items()
    ):  # Can be replaced by dico_recall_precision_gain to have the other graph
        argsort = np.argsort(recall_gain)
        recall_gain = recall_gain[argsort]
        precision_gain = precision_gain[argsort]
        current_point = 0
        curren_list = []
        step = 0.01
        next_step = step
        recall_gain_smooth = []
        precision_gain_smooth = []

        while current_point < len(recall_gain):
            curren_list.append(precision_gain[current_point])
            if recall_gain[current_point] > next_step:
                middle_point = np.mean([next_step - step, next_step])
                mean_precision = np.mean(curren_list)
                recall_gain_smooth.append(middle_point)
                precision_gain_smooth.append(mean_precision)
                curren_list = []
                next_step += step
            current_point += 1

        if simplify_label:
            plt.plot(
                recall_gain_smooth,
                precision_gain_smooth,
                label=dico_corresp_human_readable_labels[method],
                linewidth=dico_method_linewidth[method],
                linestyle=dico_method_linetype[method],
                marker=dico_method_marker[method],
            )
        else:
            plt.plot(
                recall_gain_smooth,
                precision_gain_smooth,
                label=method,
                linewidth=dico_method_linewidth[method],
                linestyle=dico_method_linetype[method],
                marker=dico_method_marker[method],
            )
    # plt.hlines(freq_label, 0, 1, linestyles="dashed", label="Random baseline")
    plt.xlabel("Recall")  # Gain
    plt.ylabel("Precision")  # Gain
    plt.title("Precision recall curve")
    plt.legend()
    plt.show()

    dico_AP_metrics = {
        k: v for k, v in sorted(dico_AP_metrics.items(), key=lambda item: item[1])
    }
    # Print the results of differents methods
    print(
        "Method type        & PRG-AUC (x100) & FGainmax (\%)        & Time (s)      \\ % max F1  can "
    )
    for method in dico_AP_metrics.keys():
        if "follow_by" in method:
            method_name = new_name_method(method)
        else:
            method_name = method
        values_AP = dico_AP_metrics[method]
        print(
            method_name,
            "&",
            str(round(dico_AUPRGC[method] * 100, 2)),
            "&",
            str(round(dico_max_F1GAIN[method] * 100, 2)),
            "& \\\ ",
        )
        print("\midrule")
        # print("AP :", np.mean(values_AP), "+/-", 2 * np.std(values_AP))
        # print("AUPRGC (X100) :", dico_AUPRGC[method] * 100)
        # print("best_f1_gain_score :", dico_max_F1GAIN[method])

    # fichier = open("dico_res_correctness.pkl", "wb")
    # pkl.dump(dico_score_correctness, fichier)


def new_name_method(method):
    # Order 2 : axis 0 -> axis 0 : First dim and first dim : (L*H,N1,N2) -> (N1,N2) -> (N2)
    # Order 3 : axis 0 -> axis 1 : First dim and second dim : (L*H,N1,N2) -> (N1,N2) -> (N1)
    # Order 1 : axis 1 -> axis 0 : Second dim and first dim : (L*H,N1,N2) -> (L*H,N2) -> (N2)
    # Order 4 : axis 2 -> axis 0 : Third dim and second dim : (L*H,N1,N2) -> (L*H,N1) -> (N1) -> Why this doesn't exist ? -> No sense with mean because sum to one but with max ? AttnAgg3M2m ?
    tab = method.split("follow_by")
    first_pool = tab[0]
    first_pool = first_pool[:-1]
    if first_pool == "mean":
        fpool = "m"
    elif first_pool == "max":
        fpool = "M"
    else:
        raise RuntimeError("Unknown pooling type")
    second_pool = tab[1].split("order")[0]
    second_pool = second_pool[1:-1]
    if second_pool == "mean":
        spool = "m"
    elif second_pool == "max":
        spool = "M"
    else:
        raise RuntimeError("Unknown pooling type")
    order = tab[1].split("order")[1]
    order = order[0]
    if order == "1":
        return "AttnAgg2" + fpool + "1" + spool
    elif order == "2":
        return "AttnAgg1" + fpool + "1" + spool
    elif order == "3":
        return "AttnAgg1" + fpool + "2" + spool
    elif order == "4":
        return "AttnAgg3" + fpool + "1" + spool
    else:
        raise RuntimeError("Unknown pooling type")


def calc_proportion_each_property(all_methods, calc_normalize=True):
    base_path_labels = "data/datasets/Swiss-prot/all_seperate_prop/"
    base_path = "data/residues_of_interest/"
    name_property = [
        "active site",
        "lipid moiety-binding region",
        "metal ion-binding site",
        "binding site",
        "calcium-binding region",
        "disulfide bond",
        "glycosylation site",
        "cross-link",
        "modified residue",
        "initiator methionine",
        "zinc finger region",
        "DNA-binding region",
        "nucleotide phosphate-binding region",
        "non-standard amino acid",
        "transmembrane region",
        "intramembrane region",
        "region of interest",
        "site",
        "coiled-coil region",
    ]

    method_that_need_training = ["Raw_attention_cls", "Raw_attention_sum_col"]

    # Load scores from different interpretability methods
    logging.info("We load the interpretability score from dump file")
    dico_scores = dict()
    for method in all_methods:
        dico_scores[method] = get_scores(base_path, method)
    all_methods += ["random"]
    # Add a random baseline
    dico_random = dict()
    for seq, scores in dico_scores["mean_follow_by_mean_order1"].items():
        dico_random[seq] = np.random.random(len(scores))
    dico_scores["random"] = dico_random
    # Add method with normalization
    if calc_normalize:
        logging.info("We calc dict with normalization")
        for method in all_methods:
            if method in method_that_need_training:
                method_that_need_training.append(method + "_min_max_norm")
                method_that_need_training.append(method + "_std_mean_norm")
                method_that_need_training.append(method + "_unit_length_norm1")
                method_that_need_training.append(method + "_unit_length_norm2")
            dico_scores[method + "_min_max_norm"] = normalize_all_dict(
                dico_scores[method], norm_type="min_max"
            )
            dico_scores[method + "_std_mean_norm"] = normalize_all_dict(
                dico_scores[method], norm_type="std_mean"
            )
            dico_scores[method + "_unit_length_norm1"] = normalize_all_dict(
                dico_scores[method], norm_type="unit_length_norm1"
            )
            dico_scores[method + "_unit_length_norm2"] = normalize_all_dict(
                dico_scores[method], norm_type="unit_length_norm2"
            )
            """
            dico_scores[method + "_exacerbate_weight"] = normalize_all_dict(
                dico_scores[method], norm_type="exacerbate_weight"
            )
            """

    # We consider each head as a different strategy for now because we don't want training for these
    key_to_delete = []
    list_all_keys = []
    list_all_values = []
    for method in dico_scores.keys():
        if method in method_that_need_training:
            list_key_to_add, list_dict_to_add = seperate_head_in_different_starts(
                dico_scores[method], method
            )
            key_to_delete.append(method)
            list_all_keys += list_key_to_add
            list_all_values += list_dict_to_add

    for key, value in zip(list_all_keys, list_all_values):
        dico_scores[key] = value

    for key in key_to_delete:
        dico_scores.pop(key, None)

    dico_proportion = dict()
    dict_proportion_normalize = dict()
    for method in dico_scores.keys():
        logging.info("We treat the method %s", method)
        print(method)
        dict_prop_one_method = dict()
        dict_prop_one_method_normalize = dict()
        for one_name_prop in name_property:
            df_label = load_json_into_pandas_dataframe(
                base_path_labels + one_name_prop + ".json"
            )
            proportion, proportion_divide_by_how_much_label = get_proportion(
                dico_scores[method], df_label
            )
            dict_prop_one_method[one_name_prop] = proportion
            dict_prop_one_method_normalize[
                one_name_prop
            ] = proportion_divide_by_how_much_label
        dict_proportion_normalize[method] = dict_prop_one_method_normalize
        dico_proportion[method] = dict_prop_one_method

    """
    # We only print the best scores for each method and each property
    print(
        "ATTENTION possibilité de sommer a plus de 100 car on prend le max pour les attentions"
    )

    # First graph : We get the max for each property with all the variant (for exemple lime we have the variant lime_normalize min max or
    # lime normalize norm1, ect...), the consequence is that for one method of interpretability we can sum at more than one, especialy for
    # the method with raw attention because we take the head with the best proportion and there are 480 head each time -> so be carefull to
    # the statistical variance
    proportion_simplify = only_get_best_proportion(
        dico_proportion, all_methods, name_property
    )
    plot_cumulative_proportion(proportion_simplify, name_property)
    print(proportion_simplify)

    # Second graph : Same as above but we divide by the number of residue that have each property in order to avoid to represent only the property
    # on residue that are very common, because it's not often what we want.
    proportion_normalize_simplify = only_get_best_proportion(
        dict_proportion_normalize, all_methods, name_property
    )
    plot_cumulative_proportion(proportion_normalize_simplify, name_property)
    """
    # Third graph : We take the best sub-strategy but we don't take for each prop independently like in graph one, we use the cummulative sum on
    # each prop to take the best. The advantage is that we can have plot that are under 100% and it's more representative of one sub-strategy BUT
    # we lost the total potential of a strategy (for example for the raw attention we only take the best head that detect all property, but it's
    # maybe differents head for each property)
    proportion_simplify_cummulative = only_get_best_cumulative_proportion(
        dico_proportion, all_methods, name_property
    )
    plot_cumulative_proportion(proportion_simplify_cummulative, name_property)

    # 4th graph : same as graph 3 but with the normalization of graph 2
    proportion_normalize_simplify_cumulative = only_get_best_cumulative_proportion(
        dict_proportion_normalize, all_methods, name_property
    )
    plot_cumulative_proportion(proportion_normalize_simplify_cumulative, name_property)


def plot_cumulative_proportion(new_dict_simplify, all_property_name):
    dict_p1 = {"strategy_name": list(new_dict_simplify.keys())}
    dict_p2 = {
        prop: [v[prop] for v in new_dict_simplify.values()]
        for prop in all_property_name
    }
    dico_describe_df = {**dict_p1, **dict_p2}
    print(dico_describe_df)
    df = pd.DataFrame.from_dict(dico_describe_df)
    df = df.sort_values(by="active site", ascending=False)
    df.plot(
        x="strategy_name",
        kind="barh",
        stacked=True,
        title="Stacked Bar Graph",
        mark_right=True,
    )
    plt.title(
        "ATTENTION possibilité de sommer a plus de 100 car on prend le max pour les attentions"
    )
    plt.show()


def only_get_best_proportion(dico_proportion, all_methods, name_property):
    new_final_dict = dict()
    for method in all_methods:
        new_dict = dict()
        for one_name_prop in name_property:
            max_prop_value = 0
            for key, dict_prop in dico_proportion.items():
                if method in key and dict_prop[one_name_prop] > max_prop_value:
                    max_prop_value = dict_prop[one_name_prop]
            new_dict[one_name_prop] = max_prop_value
        new_final_dict[method] = new_dict
    return new_final_dict


def only_get_best_cumulative_proportion(dico_proportion, all_methods, name_property):
    new_final_dict = dict()
    for method in all_methods:
        max_dict = dict()
        max_prop_value = 0
        for key, dict_prop in dico_proportion.items():
            if method in key:
                somme_this_method = 0
                for one_name_prop in name_property:
                    somme_this_method += dict_prop[one_name_prop]
                if somme_this_method > max_prop_value:
                    max_prop_value = somme_this_method
                    max_dict = dict_prop
        new_final_dict[method] = max_dict
    return new_final_dict


def seperate_head_in_different_starts(dico_raw_attn, name):
    nb_head = 480
    list_of_dict = [dict() for _ in range(nb_head)]
    list_of_keys = [name + "_head_num_" + str(k) for k in range(nb_head)]
    for key, value in dico_raw_attn.items():
        for index, elem in enumerate(value):
            list_of_dict[index][key] = elem
    return list_of_keys, list_of_dict


def get_proportion(dictionary_scores, dataframe_label):
    counter_pos = 0
    counter_total = 0
    counter_sum_interest = 0
    non_pres = 0
    for _, row in dataframe_label.iterrows():
        sequence = row["sequence"]
        sequence = sequence[:1024]
        res_of_interest = [r for r in row["residue_of_interests"] if r < 1024]
        counter_sum_interest += len(res_of_interest)
        assert (np.array(res_of_interest) > 1024).sum() == 0
        try:
            # On transforme les scores en valeurs absolue pour avoir la proportion d'interet qu'elle soit positive ou négative
            scores = np.abs(np.array(dictionary_scores[sequence]))
            scores_pos = scores[res_of_interest].sum()
            score_total = scores.sum()
            counter_pos += scores_pos
            counter_total += score_total
        except:
            non_pres += 1

    if non_pres != 0:
        logging.info("%s séquences ignorées", non_pres)
    prop = (counter_pos / counter_total) * 100
    prop_normalize = prop / counter_sum_interest
    return prop, prop_normalize


def calc_correlation(dico_scores):
    dico_corr = dict()
    for method1, value1 in dico_scores.items():
        dico_corr[method1] = dict()
        for method2, value2 in dico_scores.items():
            corr = stats.spearmanr(value1, value2).correlation
            dico_corr[method1][method2] = corr
    fichier2 = open("dico_correlation_interpretability_methods.pkl", "wb")
    pkl.dump(dico_corr, fichier2)
    print(dico_corr)


def get_correct_score(test_data, dico_scores, threshold):
    test_data = del_more_than_max_len(test_data)
    dico_correctness_seq = dict()
    for seq, label in test_data:
        pred = dico_scores[seq] > threshold
        pred = np.array(pred)
        label = np.array(label)
        correct = pred & label
        dico_correctness_seq[seq] = correct
    return dico_correctness_seq


"""
def how_many_key_not_present(dico_scores, data):
    nb = 0
    for seq, lab in data:
        print(seq)
        if seq not in dico_scores.keys():
            nb += 1
            
            maxi_sim = 0
            maxi_seq = None
            for seq2 in dico_scores.keys():
                similarity = similar(seq, seq2)
                if similarity > maxi_sim:
                    maxi_sim = similarity
                    maxi_seq = seq2
            print("Most similar sequence of :")
            print(seq)
            print("Is")
            print(maxi_seq)
            print("With a score of", maxi_sim)
            

    print("In total there are", len(data), "sequences annotated")
    print("And there are", len(dico_scores.keys()), "with a score")
    print("There are", nb, "sequences that are not annotated")
"""


def del_more_than_max_len(data, cut_to_max_len):
    new_data = []
    for seq, lab in data:
        if len(seq) <= 1024:
            new_data.append([seq, lab])
        elif cut_to_max_len:
            new_data.append([seq[:1024], lab[:1024]])
    return new_data


def get_best_head(train_data_ori, raw_attn_scores):
    nb_head = 480
    train_data = del_more_than_max_len(train_data_ori)
    best_ap = 0
    best_head = None
    for num_head in tqdm(range(nb_head)):
        y_scores = []
        y_true = []
        for seq, label in train_data:
            y_scores += list(raw_attn_scores[seq][num_head])
            y_true += label
        AP = average_precision_score(y_true, y_scores)
        if AP > best_ap:
            best_ap = AP
            best_head = num_head
    return best_head


def get_threshold(train_data_ori, dico_scores):
    prediction_scores, label_each_residue = concat_all_scores_and_label(
        train_data_ori, dico_scores
    )

    thresholds = np.linspace(min(prediction_scores), max(prediction_scores), 100)

    all_f1_scores = []
    for th in thresholds:
        # print("len label :", len(label_each_residue))
        # print("len pred :", len(prediction_scores))
        dico = classification_report(
            label_each_residue, prediction_scores > th, output_dict=True
        )
        all_f1_scores.append(dico["1"]["f1-score"])

    id_best = np.argmax(all_f1_scores)
    best_threshold = thresholds[id_best]
    return best_threshold


def perso_calc_threshold_recall_precision(list_of_label, list_of_scores):
    all_threashold = np.sort(np.unique(list_of_scores))
    all_precision = []
    all_recall = []
    for threshold in all_threashold:
        prediction = list_of_scores >= threshold
        dico_report = metrics.classification_report(
            list_of_label, prediction, output_dict=True
        )
        precision = dico_report["1"]["precision"]
        recall = dico_report["1"]["recall"]

        all_precision.append(precision)
        all_recall.append(recall)
    return (
        np.array(all_precision),
        np.array(all_recall),
        np.array(all_threashold),
    )


def affiche_precision_recall(recall, precision, label="No"):
    # Re order precision recall point
    order_x = np.argsort(recall)
    recall = recall[order_x]
    precision = precision[order_x]
    plt.plot(recall, precision, label=label, alpha=0.5)


def get_AP_and_AUPRGC(test_data_ori, scores):

    prediction_scores, label_each_residue = concat_all_scores_and_label(
        test_data_ori, scores
    )

    average_precision = average_precision_score(
        label_each_residue, prediction_scores, average="macro"
    )

    # if np.isnan(average_precision):
    #     print("ERREUR")
    #     print("AP :", average_precision)
    #     print("len label :", len(label_each_residue))
    #     print("len scores :", len(prediction_scores))
    #     print("Scores :", prediction_scores)
    #     print("Labels :", label_each_residue)
    #     print("Sum label :", np.sum(label_each_residue))
    precision, recall, thresholds = precision_recall_curve(
        label_each_residue, prediction_scores, pos_label=1
    )
    assert precision[-1] == 1 and recall[-1] == 0
    # if interval:
    #     precision = precision[:-1]
    #     recall = recall[:-1]

    # quantile = np.quantile(prediction_scores, q=0.99)
    # print("Quantile à 99% :", quantile)
    # print("Value of thresholds 0 :", thresholds[0])
    # mask = thresholds < quantile
    """
    # To clean curve and don't take first estimation of precision and recall that have not much data, Only good for visualisation and 
    # need adjustment on the number of value
    precision = precision[:-25]
    recall = recall[:-25]
    thresholds = thresholds[:-25]
    """

    arr_label_each_residue = np.array(label_each_residue)
    arr_prediction_scores = np.array(prediction_scores)
    prg_curve = prg.create_prg_curve(arr_label_each_residue, arr_prediction_scores)
    # prg.plot_prg(prg_curve)
    precision_gain = prg_curve["precision_gain"]
    recall_gain = prg_curve["recall_gain"]
    # thresholds = arr_prediction_scores
    PRG_AUC = prg.calc_auprg(prg_curve)
    # PR_AUC = metrics.auc(recall, precision)
    # print("PRG_AUC :", PRG_AUC)

    all_F_gain = (1 / 2) * precision_gain + (1 / 2) * recall_gain
    all_F_gain = np.nan_to_num(all_F_gain)
    # print("F_gain :", F_gain)
    # print("Max F_gain :", np.max(F_gain))
    max_F_gain = np.max(all_F_gain)

    return (
        average_precision,
        PRG_AUC,
        max_F_gain,
        precision_gain,
        recall_gain,
        precision,
        recall,
    )


def get_token_f1(test_data_ori, scores):
    prediction_scores, label_each_residue = dico_score_to_2d_list(test_data_ori, scores)

    nb_max = 50
    list_nb_token = list(range(1, nb_max + 1, 1))
    list_token_f1 = []

    for score_seq, label_seq in zip(prediction_scores, label_each_residue):
        order = np.flip(np.argsort(score_seq))
        ind_to_take = order[:nb_max]
        somme_TP = 0
        somme_FP = 0
        somme_FN = 0
        tmp_f1 = []
        for ind in ind_to_take:
            l = label_seq[ind]
            if l == 1:
                somme_TP += 1
            elif l == 0:
                somme_FP += 1
            tmp_f1.append(calc_f1(somme_TP, somme_FP, somme_FN))
        assert len(tmp_f1) == nb_max
        list_token_f1.append(tmp_f1)
    list_token_f1 = np.array(list_token_f1)
    list_token_f1 = list_token_f1.mean(axis=0)
    return list_nb_token, list_token_f1


def calc_f1(TP, FP, FN):
    if TP == 0 and FP + FN == 0:
        return 0
    else:
        return TP / (TP + (1 / 2) * (FP + FN))


def dico_score_to_2d_list(test_data_ori, scores):
    test_data = del_more_than_max_len(test_data_ori, cut_to_max_len=True)
    prediction_scores = []
    label_each_residue = []
    for seq, lab in test_data:
        try:
            prediction_scores.append(list(scores[seq]))
            label_each_residue.append(lab)
        except:
            print("Seq not available")
    return prediction_scores, label_each_residue


def concat_all_scores_and_label(test_data_ori, scores):
    test_data = del_more_than_max_len(test_data_ori, cut_to_max_len=True)
    prediction_scores = []
    label_each_residue = []
    for seq, lab in test_data:
        try:
            prediction_scores += list(scores[seq])
            label_each_residue += lab
        except:
            print("Seq not available")
    return prediction_scores, label_each_residue


def get_f1(test_data_ori, scores, threshold):
    test_data = del_more_than_max_len(test_data_ori)
    prediction_scores = []
    label_each_residue = []
    for seq, lab in test_data:
        label_each_residue += lab
        prediction_scores += list(scores[seq])

    dico = classification_report(
        label_each_residue, prediction_scores > threshold, output_dict=True
    )
    return dico["1"]["f1-score"]


def get_scores(base_path, method, which_dataset="catalytic_site"):
    dico_scores, _ = torch.load(
        base_path
        + method
        + "/score_with_fine_tune_models_EnzBert_SwissProt_2021_04_on_dataset_"  # "/score_with_fine_tune_models_ProtBert_model_SwissProt_not_trainned_on_dataset_", "/score_with_fine_tune_models_ProtBert_model_SwissProt_not_trainned_on_dataset_"
        + which_dataset
        + ".pkl",
        map_location=torch.device("cpu"),
    )
    if "LIME" not in method:
        if len(list(dico_scores.values())[0].shape) == 2 and "Raw" not in method:
            for key, value in dico_scores.items():
                dico_scores[key] = dico_scores[key][0]
        if (
            method != "integrated_grad"
            and "Raw" not in method
            and "LRP" not in method
            and "Gradients" not in method
            and "InputXGrad" not in method
            and "follow_by" not in method
            and "pageRank" not in method
            and "flowMax" not in method
        ):
            for key, value in dico_scores.items():
                dico_scores[key] = dico_scores[key][1:]
        if method == "Gradients":
            for key, value in dico_scores.items():
                dico_scores[key] = dico_scores[key].numpy()
        if "LRP_rollout_cls" in method:
            dico_scores = convert_all_dict_LRP_cls(dico_scores)
    # print(np.array(list(dico_scores.values())[0]).shape)
    return dico_scores


def convert_all_LRP_to_seq_shape(square_map):
    # We delete the first dim
    # square_map = square_map[0]
    # square_map = square_map[1:, 1:]
    # We take only the cls
    # score_residue = square_map[0]
    # score_residue = square_map[0, 0, 1:]              # -> Avreage_precision : 0.05768389246239737
    # score_residue = square_map[0, 1:, 0]              # -> Avreage_precision : 0.07291939140640874
    # score_residue = square_map[0, 1:, 1:].max(axis=0) # -> Avreage_precision : 0.0345743601644789
    # score_residue = square_map[0, 1:, 1:].max(axis=1) # -> Avreage_precision : 0.03457436016447896
    # score_residue = (square_map[0].max(axis=0))[1:]   # -> Avreage_precision : 0.03457436016447896
    # score_residue = (square_map[0].max(axis=1))[1:]   # -> Avreage_precision : 0.03457436016447896
    score_residue = square_map[0, 1:, 0]
    return score_residue


def convert_all_dict_LRP_cls(dico_scores):
    new_dict = dict()
    for key, value in dico_scores.items():
        new_val = convert_all_LRP_to_seq_shape(value)
        new_dict[key] = new_val
    return new_dict


def normalize_all_dict(dico_scores, norm_type):
    dioc_norm_scores = dict()
    for seq, one_vec in dico_scores.items():
        if norm_type == "min_max":
            norm_vec = min_max_normalize(one_vec)
        elif norm_type == "std_mean":
            norm_vec = std_mean_normalize(one_vec)
        elif norm_type == "unit_length_norm1":
            norm_vec = unit_length_norm1_normalize(one_vec)
        elif norm_type == "unit_length_norm2":
            norm_vec = unit_length_norm2_normalize(one_vec)
        elif norm_type == "exacerbate_weight":
            norm_vec = unit_length_norm2_normalize(one_vec)
            temp = 0.02
            norm_vec = np.exp(temp * norm_vec) / np.sum(np.exp(temp * norm_vec))
        else:
            raise RuntimeError("Nomalization type unknown")
        dioc_norm_scores[seq] = norm_vec
    return dioc_norm_scores


def min_max_normalize(vec_score):
    vec_score = np.array(vec_score)
    # normalize scores
    numerateur = vec_score - vec_score.min()
    denominateur = vec_score.max() - vec_score.min()
    norm_score = np.divide(
        numerateur, denominateur, out=np.zeros_like(numerateur), where=denominateur != 0
    )
    return norm_score


def std_mean_normalize(vec_score):
    vec_score = np.array(vec_score, dtype=np.float64)
    mean = np.mean(vec_score)
    std = np.std(vec_score)
    # normalize scores
    numerateur = vec_score - mean
    norm_score = np.divide(
        numerateur, std, out=np.zeros_like(numerateur), where=std != 0
    )
    return norm_score


def unit_length_norm1_normalize(vec_score):
    vec_score = np.array(vec_score, dtype=np.float64)
    norm_vec_score = np.sum(np.abs(vec_score))
    norm_vec = np.divide(
        vec_score,
        norm_vec_score,
        out=np.zeros_like(vec_score),
        where=norm_vec_score != 0,
    )
    return norm_vec


def unit_length_norm2_normalize(vec_score):
    vec_score = np.array(vec_score, dtype=np.float64)
    norm_vec_score = np.sqrt(np.sum(np.power(vec_score, 2)))
    norm_vec = np.divide(
        vec_score,
        norm_vec_score,
        out=np.zeros_like(vec_score),
        where=norm_vec_score != 0,
    )
    return norm_vec


def scores_distibution(which_dataset):
    base_path = "data/residues_of_interest/"

    all_methods = [
        name for name in os.listdir(base_path) if os.path.isdir(base_path + name)
    ]
    print(all_methods)

    only_basic_method = [
        "max_follow_by_mean_order3",
        "max_follow_by_mean_order2",
        "max_follow_by_max_order3",
        "max_follow_by_mean_order1",
        "mean_follow_by_max_order3",
        "mean_follow_by_max_order2",
        "max_follow_by_max_order2",
        "mean_follow_by_mean_order1",
        "mean_follow_by_mean_order3",
        "mean_follow_by_mean_order2",
        "max_follow_by_max_order1",
        "mean_follow_by_max_order1",
    ]

    dico_all_scores = dict()
    for method in only_basic_method:
        print(method)
        if (
            "Transformer_untrainned" in method
            or "Attn_last_layer" in method
            or "integrated_grad" in method
            or "LRP_rollout_cls" in method
        ):
            continue
        dico_scores, _ = torch.load(
            base_path
            + method
            + "/score_with_fine_tune_models_EnzBert_SwissProt_2021_04_on_dataset_"
            + which_dataset
            + ".pkl",
            map_location=torch.device("cpu"),
        )
        all_scores = list(dico_scores.values())
        if not isinstance(all_scores[0], list) and len(all_scores[0].shape) == 2:
            all_scores = [a.flatten() for a in all_scores]

        all_scores = np.concatenate(all_scores)
        all_scores = all_scores.flatten()
        # print(all_scores.shape)
        # print(np.percentile(all_scores, 0.01))
        print(np.percentile(all_scores, 0.001))
        # print("std:", np.std(all_scores))
        dico_all_scores[method] = list(all_scores)

    fig, ax = plt.subplots()
    ax.boxplot(dico_all_scores.values())
    ax.set_xticklabels(dico_all_scores.keys())
    plt.xticks(rotation="vertical")
    plt.show()