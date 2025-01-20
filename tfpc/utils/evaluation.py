import numpy as np
import torch
from utils.json_loader import load_json_into_pandas_dataframe
from tqdm import tqdm
from utils.metrics_ECPred import ECPredF1, MacroAvg
import sys
import pandas as pd
import ast
import copy
import pickle as pkl
import matplotlib.pyplot as plt


def find_all_recalibration_weights(
    model_name,
    vocab_class_path,
):

    class_vocab = torch.load(vocab_class_path)
    print("class_vocab :", class_vocab)

    data_path_valid = "data/datasets/ECPred_dataset/ECPred_dataset_valid.json"
    df_valid = load_json_into_pandas_dataframe(data_path_valid)
    prior_ec_class = df_valid["EC Number"].value_counts().to_dict()
    # Add one to all class occurence to avoid proba 0 for are class
    prior_ec_class = {k: v + 1 for k, v in prior_ec_class.items()}
    number_of_elements = sum(list(prior_ec_class.values()))
    prior_ec_class = {k: v / number_of_elements for k, v in prior_ec_class.items()}
    mini_proba = float(np.min(list(prior_ec_class.values())))
    prior_ec_class_order_vocabulary = np.array(
        [
            float(prior_ec_class[ec_class])
            if ec_class in prior_ec_class.keys()
            else mini_proba
            for ec_class in class_vocab.keys()
        ]
    )
    print(prior_ec_class_order_vocabulary[:10])
    fichier_prior = open(
        "data/predictions/prior_ec_class_order_vocabulary_" + model_name + ".pkl", "wb"
    )
    pkl.dump(prior_ec_class_order_vocabulary, fichier_prior)


def find_best_threshold(
    weight_predicted_path,
    threshold_type,
    which_metrics,
    vocab_class_path,
    num_trial=25,
):

    metric = getattr(sys.modules[__name__], which_metrics)()

    class_vocab = torch.load(vocab_class_path)
    print("class_vocab :", class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    df_output_valid_weights = pd.read_csv(weight_predicted_path, sep=";")
    df_output_valid_weights["all_weights"] = [
        ast.literal_eval(weights_list)
        for weights_list in tqdm(df_output_valid_weights["all_weights"])
    ]

    data_path_valid = "data/datasets/ECPred_dataset/ECPred_dataset_valid.json"
    df_valid = load_json_into_pandas_dataframe(data_path_valid)

    max_metric_valid = 0
    best_threshold = None
    if threshold_type == "multiplicative_zero_class":
        all_possible_thresholds = np.linspace(1, 10, num=num_trial)  # max : 3
    elif threshold_type == "simple_thresholds_max_proba":
        all_possible_thresholds = np.linspace(0, 1, num=num_trial)
    elif threshold_type == "simple_thresholds_proba_zero":
        all_possible_thresholds = np.linspace(0, 1, num=num_trial)
    elif threshold_type == "double_thresholds":
        all_possible_thresholds = np.array(
            np.meshgrid(
                np.linspace(0.4, 1, num=num_trial), np.linspace(0, 0.1, num=num_trial)
            )
        ).T.reshape(-1, 2)
    else:
        raise RuntimeError("Threshold type unknown")
    for thresholds in tqdm(all_possible_thresholds):
        df_copy = df_output_valid_weights.copy()
        dico_prediction = get_dico_pred(
            df_copy,
            threshold_type,
            thresholds,
            class_vocab,
            inverse_class_vocab,
        )
        metric_obj = calc_all_metric_for_ECPred_with_model_proba_and_vocab(
            dico_prediction,
            df_valid,
            metric,
        )
        metric_value = metric_obj.get_main_metric()
        if metric_value > max_metric_valid:
            max_metric_valid = metric_value
            best_threshold = [thresholds]

    print("Max metric :", max_metric_valid)
    print("Best thresholds :", best_threshold)

    return best_threshold


def calc_all_metric_for_ECPred_with_model_proba_and_vocab(
    dico_prediction,
    df_true_label,
    metric_obj,
):
    metric_obj.reset_metric()
    col_names = df_true_label.columns

    if "ec_number" in col_names and "sequence" in col_names:
        dico_true_labels = {
            row["sequence"]: row["ec_number"] for _, row in df_true_label.iterrows()
        }
    elif "EC Number" in col_names and "sequence" in col_names:
        dico_true_labels = {
            row["sequence"]: row["EC Number"] for _, row in df_true_label.iterrows()
        }
    elif "label" in col_names and "primary" in col_names:
        dico_true_labels = {
            row["primary"]: row["label"] for _, row in df_true_label.iterrows()
        }
    else:
        raise RuntimeError("Col name unknwon")

    for sequence, true_label in dico_true_labels.items():
        if len(sequence) <= 40 or len(sequence) > 1024:
            continue
        if sequence in dico_prediction.keys():
            prediction = dico_prediction[sequence]
        else:
            prediction = dico_prediction[sequence[:1024]]

        metric_obj.step(true_label, prediction)

    return metric_obj


def get_dico_pred(
    df_output_weights, threshold_type, thresholds, class_vocab, inverse_class_vocab
):
    dico_prediction = dict()
    for _, row in df_output_weights.iterrows():
        sequence = row["sequence"]
        all_weights = row["all_weights"]

        prediction = get_pred_from_weights(
            all_weights, threshold_type, thresholds, class_vocab, inverse_class_vocab
        )
        dico_prediction[sequence] = prediction
    return dico_prediction


def get_dico_pred_with_prior(
    df_output_weights, class_vocab, inverse_class_vocab, prior_vector
):
    dico_prediction = dict()
    for _, row in df_output_weights.iterrows():
        sequence = row["sequence"]
        all_weights = np.array(row["all_weights"])
        all_proba = np.exp(all_weights) / np.sum(np.exp(all_weights))
        assert len(all_proba) == len(prior_vector)
        # new_list_class_weight = (0.6 * all_weights) + (0.4 * prior_vector)
        new_list_class_weight = all_proba * prior_vector

        assert len(new_list_class_weight) == len(prior_vector)
        prediction = inverse_class_vocab[np.argmax(new_list_class_weight)]
        dico_prediction[sequence] = prediction
    return dico_prediction


def get_pred_from_weights(
    list_class_weight, threshold_type, thresholds, vocab, inverse_vocab
):
    tmp_list_class_weight = copy.deepcopy(list_class_weight)
    new_list_class_weight = tmp_list_class_weight

    new_list_class_weight = recalibrate_proba(
        tmp_list_class_weight, thresholds, threshold_type, vocab
    )
    prediction = inverse_vocab[np.argmax(new_list_class_weight)]
    return prediction


@torch.no_grad()
def get_all_weights_EC40(
    model_path,
    vocab_path,
    which_set,
):
    max_seq_len = 1024  # 2048  # 1024
    if which_set == "test":
        data_path = "data/datasets/EC_prediction/EC_prediction_test.json"
    else:
        raise RuntimeError("Set unknwon, you have to choose test")
    dataframe = load_json_into_pandas_dataframe(data_path)
    list_sequences = dataframe["primary"]

    # Load the model
    model_name = model_path.split("/")[-2]
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model = model.eval()
    vocab = torch.load("data/models/" + vocab_path)

    fichier_result_proba_dist = open(
        "data/predictions/all_weights_model_"
        + model_name
        + "_on_set_"
        + which_set
        + ".csv",
        "w",
    )
    fichier_result_proba_dist.write("sequence;all_weights\n")
    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        fichier_result_proba_dist.write(
            sequence + ";" + str(list(proba_pred.numpy())) + "\n"
        )

    fichier_result_proba_dist.close()


##########################################################################################
### DIFERENT RECALIBRATION TYPE
##########################################################################################


def recalibrate_proba(list_class_weight, thresholds, recalibration_type, vocab):
    ind_class_zeros = vocab["0.0.0.0"]
    if recalibration_type == "double_thresholds":
        new_list_class_weight = recalibrate_with_double_thresholds(
            list_class_weight, thresholds, ind_class_zeros
        )
    elif recalibration_type == "simple_thresholds_max_proba":
        new_list_class_weight = recalibrate_with_simple_thresholds_max_proba(
            list_class_weight, thresholds, ind_class_zeros
        )
    elif recalibration_type == "simple_thresholds_proba_zero":
        new_list_class_weight = recalibrate_with_simple_thresholds_proba_zero(
            list_class_weight, thresholds, ind_class_zeros
        )
    elif recalibration_type == "multiplicative_zero_class":
        new_list_class_weight = recalibrate_with_multiplicative_zero_class(
            list_class_weight, thresholds, ind_class_zeros
        )
    else:
        raise RuntimeError("Recalibraiton unknwon")
    return new_list_class_weight


def recalibrate_with_double_thresholds(list_class_weight, thresholds, ind_class_zeros):
    list_all_proba = np.exp(list_class_weight) / sum(np.exp(list_class_weight))
    proba_class_zero = list_all_proba[ind_class_zeros]
    list_all_proba[ind_class_zeros] = 0
    m_proba = np.max(list_all_proba)
    if m_proba < thresholds[0] and proba_class_zero > thresholds[1]:
        list_class_weight[ind_class_zeros] = 100
    else:
        list_class_weight[ind_class_zeros] = -100
    return list_class_weight


def recalibrate_with_simple_thresholds_max_proba(
    list_class_weight, thresholds, ind_class_zeros
):
    list_all_proba = np.exp(list_class_weight) / sum(np.exp(list_class_weight))
    list_all_proba[ind_class_zeros] = 0
    m_proba = np.max(list_all_proba)
    if m_proba < thresholds:
        list_class_weight[ind_class_zeros] = 10
    else:
        list_class_weight[ind_class_zeros] = -10

    return list_class_weight


def recalibrate_with_simple_thresholds_proba_zero(
    list_class_weight, thresholds, ind_class_zeros
):
    list_all_proba = np.exp(list_class_weight) / sum(np.exp(list_class_weight))
    proba_class_zero = list_all_proba[ind_class_zeros]
    if proba_class_zero > thresholds:
        list_class_weight[ind_class_zeros] = 100
    else:
        list_class_weight[ind_class_zeros] = -100

    return list_class_weight


def recalibrate_with_multiplicative_zero_class(
    list_class_weight, thresholds, ind_class_zeros
):
    list_class_weight[ind_class_zeros] *= thresholds
    return list_class_weight


def get_metric_on_EC40_test(metric_obj):
    """
    Arguments :
    which_level: "0" or "others"
    """
    FOLDER_XP = "data/models/fine_tune_models/"
    FOLDER_Prediction = "data/predictions/"
    threshold_type = "multiplicative_zero_class"

    model_name = "ProtBert_EC40_layer_norm_proba_r1"
    path_class_vocab = FOLDER_XP + model_name + "/classif_EC_pred_lvl_2_vocab.pth"
    class_vocab = torch.load(path_class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    which_set = "test"
    weight_predicted_path = (
        FOLDER_Prediction
        + "all_weights_model_"
        + model_name
        + "_on_set_"
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

    df_true_label = pd.read_json("data/datasets/EC_prediction/EC_prediction_test.json")

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


def get_metric_on_ECPred40_test(which_model, which_level, metric_obj, thresholds):
    """
    Arguments :
    which_level: "0" or "others"
    """
    print("I generate metric for model", which_model, "at level", which_level)
    FOLDER_XP = "data/models/fine_tune_models/"
    FOLDER_Prediction = "data/predictions/"
    threshold_type = "multiplicative_zero_class"
    if which_model == "ECPred":
        if which_level == "0":
            path_file_ECPred = "ECPred_predictions/ECPred_predictions.csv"
        elif which_level == "others":
            path_file_ECPred = (
                "ECPred_predictions/ECPred_predictions_a_priori_enzyme.csv"
            )
        else:
            raise RuntimeError("Level unknown")
        df_predictions = pd.read_csv(FOLDER_Prediction + path_file_ECPred)
        df_predictions.loc[
            (df_predictions.prediction == "non Enzyme"), "prediction"
        ] = "0.0.0.0"
        dico_predictions = {
            row["sequence"]: row["prediction"] for _, row in df_predictions.iterrows()
        }
    elif which_model == "EnzBert":
        # print("Temporary change model_name and path_class_vocab for supplementary XP")
        model_name = "EnzBert_ECPred40"  # "ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r2"
        path_class_vocab = (
            FOLDER_XP + model_name + "/classif_EC_pred_lvl_4_vocab.pth"
        )  #
        class_vocab = torch.load(path_class_vocab)
        inverse_class_vocab = {v: k for k, v in class_vocab.items()}

        which_set = "test"
        weight_predicted_path = (
            FOLDER_Prediction
            + "all_weights_model_"
            + model_name
            + "_on_dataset_ECPred40_"
            + which_set
            + ".csv"
        )
        df_output_valid_weights = pd.read_csv(weight_predicted_path, sep=";")
        df_output_valid_weights["all_weights"] = [
            ast.literal_eval(weights_list)
            for weights_list in df_output_valid_weights["all_weights"]
        ]

        # Only predict the class the ECPred predict
        # print("Tmp allow not allowed pred")
        # df_output_valid_weights["all_weights"] = [
        #     avoid_not_allowed_ec_number(weights_list, class_vocab)
        #     for weights_list in df_output_valid_weights["all_weights"]
        # ]

        if thresholds == "prior":
            fichier_prior = open(
                "data/predictions/prior_ec_class_order_vocabulary_"
                + model_name
                + ".pkl",
                "rb",
            )
            prior_ec_class_order_vocabulary = pkl.load(fichier_prior)
            if which_level == "others":
                prior_ec_class_order_vocabulary[class_vocab["0.0.0.0"]] = -999999
            dico_predictions = get_dico_pred_with_prior(
                df_output_valid_weights,
                class_vocab,
                inverse_class_vocab,
                prior_ec_class_order_vocabulary,
            )
        else:
            if which_level == "others":
                thresholds = 0  # Like we never predict class zero, bc never less than 0

            dico_predictions = get_dico_pred(
                df_output_valid_weights,
                threshold_type,
                thresholds,
                class_vocab,
                inverse_class_vocab,
            )
    else:
        raise RuntimeError("Model unknown")

    df_true_label = pd.read_json("data/datasets/ECPred40/ECPred40_test.json")
    if which_level == "others":
        # We need to ignore level 0 here, so delete all non enzyme
        df_true_label = df_true_label[
            df_true_label["EC Number"] != "0.0.0.0"
        ]  # ec_number

    metric_obj = calc_all_metric_for_ECPred_with_model_proba_and_vocab(
        dico_predictions,
        df_true_label,
        metric_obj,
    )
    if which_level == "0":
        all_metrics = metric_obj.get_metric_at_lvl(0)
        print("macro avg :", all_metrics["macro avg"])
        # print("weighted avg :", all_metrics["weighted avg"])
        print("Accuracy/micro-avg :", all_metrics["accuracy"])
    else:
        metric_obj.get_all_metrics(level_max=4)
    dir_res = "data/predictions/"
    metric_obj.log_all_metrics(
        filepath=dir_res
        + "metric_"
        + which_model
        + "_on_levels_"
        + which_level
        + "_on_ECPred40_dataset.csv",
        which_level=which_level,
    )


def generate_calibration_plot():
    FOLDER_XP = "data/models/fine_tune_models/"
    FOLDER_Prediction = "data/predictions/"

    model_name = "ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r2"
    path_class_vocab = FOLDER_XP + model_name + "/classif_EC_pred_lvl_4_vocab.pth"
    class_vocab = torch.load(path_class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    which_set = "test"  # "valid"
    weight_predicted_path = (
        FOLDER_Prediction
        + "all_weights_model_"
        + model_name
        + "_on_set_"
        + which_set
        + ".csv"
    )
    df_output_valid_weights = pd.read_csv(weight_predicted_path, sep=";")
    df_output_valid_weights["all_weights"] = [
        ast.literal_eval(weights_list)
        for weights_list in df_output_valid_weights["all_weights"]
    ]

    dico_prediction = dict()
    dico_proba = dict()
    for _, row in df_output_valid_weights.iterrows():
        sequence = row["sequence"]
        all_weights = np.array(row["all_weights"])
        all_proba = np.exp(all_weights) / np.sum(np.exp(all_weights))
        prediction = inverse_class_vocab[np.argmax(all_proba)]
        dico_proba[sequence] = np.max(all_proba)
        dico_prediction[sequence] = prediction

    df_true_label = pd.read_json(
        "data/datasets/ECPred_dataset/new_test_set_without_fragments.json"  # valid.json"
    )

    nb_bin = 10  # 5
    step = 1 / nb_bin
    list_res = [[] for _ in range(nb_bin)]
    all_proba_per_range = [[] for _ in range(nb_bin)]
    list_all_proba = []
    list_all_correct = []
    for _, row in df_true_label.iterrows():
        sequence = row["sequence"]
        ec_number = row["ec_number"]  # "EC Number"
        sequence = sequence[:1024]
        pred = dico_prediction[sequence]
        proba = dico_proba[sequence]
        correct = ec_number == pred
        num_bucket = int(proba / step)
        list_res[num_bucket].append(correct)
        all_proba_per_range[num_bucket].append(proba)
        list_all_proba.append(proba)
        list_all_correct.append(correct)

    proba_model = []
    accuracy = []
    dico_value_boxplot = dict()
    for k in range(nb_bin):
        # list_res[k] = np.mean(list_res[k])
        min_proba = k * step
        max_proba = (k + 1) * step
        proba_model.append(np.mean(all_proba_per_range[k]))
        accuracy.append(np.mean(list_res[k]))
        dico_value_boxplot[str(accuracy[-1])[:5]] = all_proba_per_range[k]
        print("Proba range:", min_proba, ":", max_proba)
        print("Nb predictions in this proba range:", len(list_res[k]))
        print("Accuracy on this range:", np.mean(list_res[k]))
        print("Mean proba on this range:", np.mean(all_proba_per_range[k]))

    plt.plot(proba_model, accuracy, label="model calibration")
    plt.plot(proba_model, proba_model, label="perfect calibration")
    plt.xlabel("Probability of the model")
    plt.ylabel("Actual accuracy")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(dico_value_boxplot.values())
    ax.set_xticklabels(dico_value_boxplot.keys())
    plt.xticks(rotation=90)
    plt.ylabel("Probability of the model")
    plt.xlabel("Actual accuracy")
    plt.show()

    # Version not by bin but by enought value to calculate accuracy -> To be more precise at the end where all the model proba are
    min_estimation = 100
    list_all_proba = np.array(list_all_proba)
    list_all_correct = np.array(list_all_correct)
    ind_order_proba = np.argsort(list_all_proba)
    order_proba = list_all_proba[ind_order_proba]
    order_correct = list_all_correct[ind_order_proba]
    proba_model = []
    accuracy = []
    for k in range(int(len(ind_order_proba) / min_estimation) + 1):
        estimated_accuracy = np.mean(
            order_correct[k * min_estimation : (k + 1) * min_estimation]
        )
        model_probability = np.mean(
            order_proba[k * min_estimation : (k + 1) * min_estimation]
        )
        proba_model.append(model_probability)
        accuracy.append(estimated_accuracy)
    plt.plot(proba_model, accuracy, label="model calibration")
    plt.plot(proba_model, proba_model, label="perfect calibration")
    plt.xlabel("Probability of the model")
    plt.ylabel("Actual accuracy")
    plt.legend()
    plt.show()


def avoid_not_allowed_ec_number(weight, vocab):
    fichier = open("data/predictions/allow_ec_number.txt")
    ec_allowed_to_predict = fichier.readlines()
    ec_allowed_to_predict = [ec[:-1] for ec in ec_allowed_to_predict]
    minimum_weights = np.min(weight)
    for ec in vocab.keys():
        if ec not in ec_allowed_to_predict and ec != "0.0.0.0":
            weight[vocab[ec]] = minimum_weights  # -10
    return weight
