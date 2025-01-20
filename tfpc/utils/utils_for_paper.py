from collections import Counter

from regex import D
from utils.json_loader import load_json_into_pandas_dataframe
from utils.utils import create_dict_per_level_for_metric
from utils.evaluation import (
    get_metric_on_ECPred40_test,
    calc_all_metric_for_ECPred_with_model_proba_and_vocab,
)
from interpretability.evaluate_interpretability_method import (
    compare_intepretability_methods_V2,
)
from analysis.training_analyser import TrainingAnalyser
from utils.launch_interpretability import launch_interpretability
from utils.metrics_ECPred import MacroAvg
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import os
import time
import sys
import ast

sys.path.append("models_architectures/")


def get_nb_prot_each_main_ec_class(dataset_name, all_sets, colname_ec, separator):
    path_data = "data/datasets/" + dataset_name + "/" + dataset_name + "_"
    print("For", dataset_name, "dataset:")
    for type_d in all_sets:
        print("On dataset:", type_d)
        df = load_json_into_pandas_dataframe(path_data + type_d + ".json")
        print(Counter(df[colname_ec].apply(lambda x: x.split(separator)[0])))


def generate_latex_result_from_prediction_on_ECPred40(dir_res):
    only_enzyme_ECPred = pd.read_csv(
        dir_res + "metric_ECPred_on_levels_others_on_ECPred40_dataset.csv"
    )
    print(only_enzyme_ECPred)
    dico_only_enzyme_ECPred = create_dict_per_level_for_metric(
        "macro avg", only_enzyme_ECPred
    )
    only_enzyme_EnzBert = pd.read_csv(
        dir_res + "metric_EnzBert_on_levels_others_on_ECPred40_dataset.csv"
    )
    dico_only_enzyme_EnzBert = create_dict_per_level_for_metric(
        "macro avg", only_enzyme_EnzBert
    )

    discimination_lvl_0_ECPred = pd.read_csv(
        dir_res + "metric_ECPred_on_levels_0_on_ECPred40_dataset.csv"
    )
    dico_ECPred_level_0 = create_dict_per_level_for_metric(
        "macro avg", discimination_lvl_0_ECPred
    )
    discimination_lvl_0_EnzBert = pd.read_csv(
        dir_res + "metric_EnzBert_on_levels_0_on_ECPred40_dataset.csv"
    )
    dico_EnzBert_level_0 = create_dict_per_level_for_metric(
        "macro avg", discimination_lvl_0_EnzBert
    )
    print(only_enzyme_EnzBert)
    print("-" * 89)
    print(discimination_lvl_0_EnzBert)

    round_digit = 3
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Metric & level & Macro-f1 & Macro-precision & Macro-recall & Accuracy \\\ ")
    # Level 0
    # print(dico_ECPred_level_0)
    res_ECPred = dico_ECPred_level_0[0]
    res_EnzBert = dico_EnzBert_level_0[0]
    print("\\midrule ")
    print(
        "ECPred &"
        + str(0)
        + " & "
        + str(round(res_ECPred["f1"], round_digit))
        + " & "
        + str(round(res_ECPred["precision"], round_digit))
        + " & "
        + str(round(res_ECPred["recall"], round_digit))
        + " & "
        + str(round(res_ECPred["accuracy"], round_digit))
        + "\\\ "
    )
    print(
        "EnzBert &"
        + str(0)
        + " & "
        + str(round(res_EnzBert["f1"], round_digit))
        + " & "
        + str(round(res_EnzBert["precision"], round_digit))
        + " & "
        + str(round(res_EnzBert["recall"], round_digit))
        + " & "
        + str(round(res_EnzBert["accuracy"], round_digit))
        + "\\\ "
    )
    # Level 1 to 4
    for lvl in range(1, 5):
        res_ECPred = dico_only_enzyme_ECPred[lvl]
        res_EnzBert = dico_only_enzyme_EnzBert[lvl]
        print("\\midrule ")
        print(
            "ECPred &"
            + str(lvl)
            + " & "
            + str(round(res_ECPred["f1"], round_digit))
            + " & "
            + str(round(res_ECPred["precision"], round_digit))
            + " & "
            + str(round(res_ECPred["recall"], round_digit))
            + " & "
            + str(round(res_ECPred["accuracy"], round_digit))
            + "\\\ "
        )
        print(
            "EnzBert &"
            + str(lvl)
            + " & "
            + str(round(res_EnzBert["f1"], round_digit))
            + " & "
            + str(round(res_EnzBert["precision"], round_digit))
            + " & "
            + str(round(res_EnzBert["recall"], round_digit))
            + " & "
            + str(round(res_EnzBert["accuracy"], round_digit))
            + "\\\ "
        )

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Accuracy for level 1 and 2 prediction on the $EC40$ test set.}")
    print("\\label{tab:result_EC40}")
    print("\\end{table}")


@torch.no_grad()
def get_all_prediction_weights(
    model_path,
    vocab_path,
    which_dataset,
    which_set,
    colname_sequences,
    folder_save_prediction,
):
    # colname_sequences="sequence" or "primary"
    max_seq_len = 1024
    data_path = (
        "data/datasets/"
        + which_dataset
        + "/"
        + which_dataset
        + "_"
        + which_set
        + ".json"
    )
    dataframe = load_json_into_pandas_dataframe(data_path)
    list_sequences = dataframe[colname_sequences]

    # Load the model
    model_name = model_path.split("/")[-2]
    if torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device("cuda"))
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"))
    model = model.eval()
    vocab = torch.load("data/models/" + vocab_path)

    fichier_result_weights = open(
        folder_save_prediction
        + "all_weights_model_"
        + model_name
        + "_on_dataset_"
        + which_dataset
        + "_"
        + which_set
        + ".csv",
        "w",
    )
    fichier_result_weights.write("sequence;all_weights\n")
    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        fichier_result_weights.write(
            sequence + ";" + str(list(proba_pred.cpu().numpy())) + "\n"
        )

    fichier_result_weights.close()


def generate_metric_file_ECPred40():
    get_metric_on_ECPred40_test(
        which_model="EnzBert",
        which_level="0",
        metric_obj=MacroAvg(),
        thresholds=1,
    )
    get_metric_on_ECPred40_test(
        which_model="ECPred", which_level="0", metric_obj=MacroAvg(), thresholds=None
    )
    get_metric_on_ECPred40_test(
        which_model="EnzBert",
        which_level="others",
        metric_obj=MacroAvg(),
        thresholds=None,
    )

    get_metric_on_ECPred40_test(
        which_model="ECPred",
        which_level="others",
        metric_obj=MacroAvg(),
        thresholds=None,
    )


def get_metric_on_EC40_test(metric_obj):
    """
    Arguments :
    which_level: "0" or "others"
    """
    FOLDER_XP = "data/models/fine_tune_models/"
    FOLDER_Prediction = "data/predictions/"
    threshold_type = "multiplicative_zero_class"

    model_name = "EnzBert_EC40"
    path_class_vocab = FOLDER_XP + model_name + "/classif_EC_pred_lvl_2_vocab.pth"
    class_vocab = torch.load(path_class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    which_set = "test"
    weight_predicted_path = (
        FOLDER_Prediction
        + "all_weights_model_"
        + model_name
        + "_on_dataset_EC40_"
        + which_set
        + ".csv"
    )
    df_output_valid_weights = pd.read_csv(weight_predicted_path, sep=";")
    df_output_valid_weights["all_weights"] = [
        ast.literal_eval(weights_list)
        for weights_list in df_output_valid_weights["all_weights"]
    ]

    dico_prediction = dict()
    for _, row in df_output_valid_weights.iterrows():
        sequence = row["sequence"]
        all_weights = row["all_weights"]
        prediction = inverse_class_vocab[np.argmax(all_weights)]
        dico_prediction[sequence] = prediction

    df_true_label = pd.read_json("data/datasets/EC40/EC40_test.json")

    metric_obj = calc_all_metric_for_ECPred_with_model_proba_and_vocab(
        dico_prediction,
        df_true_label,
        metric_obj,
    )

    metric_obj.get_all_metrics(level_max=2)
    dir_res = "data/predictions/"
    metric_obj.log_all_metrics(
        filepath=dir_res
        + "metric_"
        + "EnzBert"
        + "_on_levels_"
        + "1_and_2"
        + "_on_ECPred40_dataset.csv",
        which_level="others",
    )


def compute_feature_importance_scores():
    FOLDER_XP = "data/models/fine_tune_models/"
    all_method_that_we_want_to_launch = [
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
        "grad_and_inputXgrad",
        "attn_last_layer",
        "gradCam",
        "LRP_with_rollout_cls_and_sum_col",
        "rollout",
        "integrated_grad",
        "lime",
    ]

    path_choosen = FOLDER_XP + "EnzBert_SwissProt_2021_04"

    training_analyser = TrainingAnalyser(path_choosen)
    print(training_analyser.config)
    dataset_name = "catalytic_site"

    print(
        "I will execute all interpretability methods that work on GPU(all except LRP because not enought GPU RAM)"
    )
    nb_seq = "all"
    fichier = open("data/residues_of_interest/time_to_execute.csv", "w")
    fichier.write("Method_name,nb_seq,time_elsapsed")
    for method_name in all_method_that_we_want_to_launch:
        print("Je lance la méthode", method_name)
        start_time = time.time()
        launch_interpretability(
            method_name, path_choosen, training_analyser, dataset_name, nb_seq
        )
        time_elsapsed = time.time() - start_time
        fichier.write(method_name + "," + str(nb_seq) + "," + str(time_elsapsed) + "\n")
    fichier.close()


def compare_intepretability_methods():
    # List of all method
    base_path = "data/residues_of_interest/"
    all_methods = os.listdir(base_path)
    all_methods = [a for a in all_methods if "." not in a]
    dataset_name = "catalytic_site"

    new_all_method_list = [
        "mean_follow_by_mean_order1",
        "InputXGrad",
        "Gradients",
        "integrated_grad",
        "LIME_with_5000_corupt_seq_with_m_replacment_character",
        "Attn_last_layer",
        "GradCam",
        "Rollout",
        "LRP_rollout_cls",
    ]
    compare_intepretability_methods_V2(new_all_method_list, which_dataset=dataset_name)

    # Time to execute
    print("Time to execute each methods in second:")
    fichier = open("data/residues_of_interest/time_to_execute.csv", "r")
    print(fichier.read())
