import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
from utils.json_loader import load_json_into_pandas_dataframe


def calc_proportion_class_training(vocab, type_data):
    if type_data == "BENZ":
        data_path = "data/datasets/BENZ_dataset/BENZ_dataset_train.json"
    elif type_data == "My_dataset_SwissProt_04":
        data_path = "data/datasets/SwissProt_2021_01/SwissProt_2021_01_train.json"
    else:
        raise RuntimeError(
            "Dataset of training not define, add and elif to add path to the train set for this type_data"
        )
    dataframe = load_json_into_pandas_dataframe(data_path)
    list_labels = dataframe["ec_number"]
    occurence_each_label = dict(Counter(list_labels))
    nb_total_occ = np.sum(occurence_each_label.values())
    weights_list = []
    print("Len vocab :", len(vocab))
    for k in range(len(vocab)):
        try:
            """
            weight = 1 / (
                occurence_each_label[vocab[k]] + 0.05 * len(vocab)
            )  # np.mean([1, 1 / (occurence_each_label[vocab[k]] + 1)])

            weight = np.mean(
                [1, 1 / (occurence_each_label[vocab[k]] + 1)]
            )  # 1 / occurence_each_label[vocab[k]]
            """
            # weight = 1 / (occurence_each_label[vocab[k]] + 223)
            # weight = 1 / (occurence_each_label[vocab[k]] + 100000)
            # weight = 1 / (occurence_each_label[vocab[k]] + 1)
            # print(vocab[k])
            # weight = 1 + 0.01 * occurence_each_label[vocab[k]] #No impact on time-based eval

            if vocab[k] == "0.0.0.0":
                print("Not an enzyme weight set to zero")
                weight = 0
            else:
                weight = 1

        except:
            weight = 1e-6
        weights_list.append(weight)
    return weights_list, occurence_each_label


def return_weight_set(vocab, weight_class_zero):
    data_path = "data/datasets/BENZ_dataset/BENZ_dataset_train.json"
    dataframe = load_json_into_pandas_dataframe(data_path)
    list_labels = dataframe["ec_number"]
    occurence_each_label = dict(Counter(list_labels))
    weights_list = []
    print("Len vocab :", len(vocab))
    for k in range(len(vocab)):
        if vocab[k] == "0.0.0.0":
            print("Not an enzyme weight set")
            weight = weight_class_zero
        else:
            weight = 1
        weights_list.append(weight)
    return weights_list



@torch.no_grad()
def calc_all_metric_for_ECPred_with_model_proba_and_vocab(
    dico_model_proba,
    vocab,
    inverse_class_vocab,
    list_sequences,
    list_labels,
    max_seq_len,
):
    TP = np.zeros(4)  # 0
    FN = np.zeros(4)  # 0
    TN = np.zeros(4)  # 0
    FP = np.zeros(4)  # 0
    correct = np.zeros(4)
    incorrect = np.zeros(4)

    fichier_resultat = open(
        "data/predictions/model_ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r1_on_dataset_ECPred_dataset_with_class_zero_True_with_optimal_weight.csv",
        "w",
    )
    fichier_resultat.write("sequence,prediction\n")
    fichier_result_proba_dist = open(
        "data/predictions/all_proba_model_ProtBert_ECPred_dataset_layer_norm_proba_balanced_class_r1_on_dataset_ECPred_dataset.csv",
        "w",
    )
    fichier_result_proba_dist.write("sequence;all_proba\n")
    # We calc the metric on all the dev set
    for sequence, label in tqdm(
        zip(list_sequences, list_labels), total=len(list_sequences)
    ):
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
        # On modifie la classe zéro pour qu'elle soit moins présente
        if recalibrate:
            proba_pred = recalibrate_proba(proba_pred, weight_list)
        pred_indice = int(np.argmax(proba_pred.cpu().numpy()))
        pred_class = inverse_class_vocab[pred_indice]

        fichier_resultat.write(sequence + "," + pred_class + "\n")

        class_zero = "0.0.0.0"

        for lvl in range(1, 5):
            label_truncated = ".".join(label.split(".")[:lvl])
            pred_truncated = ".".join(pred_class.split(".")[:lvl])
            class_zero_truncated = ".".join(class_zero.split(".")[:lvl])
            if pred_truncated == label_truncated:
                correct[lvl - 1] += 1
            else:
                incorrect[lvl - 1] += 1
            if (
                label_truncated == class_zero_truncated
                and pred_truncated == class_zero_truncated
            ):
                TN[lvl - 1] += 1
            elif (
                label_truncated == class_zero_truncated
                and pred_truncated != class_zero_truncated
            ):
                FP[lvl - 1] += 1
            elif label_truncated == pred_truncated:
                TP[lvl - 1] += 1
            else:
                FN[lvl - 1] += 1
    fichier_resultat.close()
    fichier_result_proba_dist.close()
    return TP, FN, TN, FP, correct, incorrect


@torch.no_grad()
def calc_all_metrics_on_positive(
    model_path,
    vocab_path,
    count_correct_if_multiple_class,
    recalibrate=True,
    type_data="BENZ",
    max_lvl=4,
    class_separator_model=".",
    class_seperator_dataset=".",
):  # recalibrate = True
    max_seq_len = 1024  # 1024

    # Load the data
    if type_data == "BENZ":
        data_path = "data/datasets/BENZ_dataset/Correct_positive_BENZ_test_set.json"
        dataframe = load_json_into_pandas_dataframe(data_path)
        list_sequences = dataframe["sequence"]
        list_labels = dataframe["ec_number"]
        list_is_seventh = dataframe["is_seventh"]
    elif type_data == "EC40":
        data_path = "data/datasets/EC_prediction/EC_prediction_test.json"
        dataframe = load_json_into_pandas_dataframe(data_path)
        list_sequences = dataframe["primary"]
        list_labels = dataframe["label"]
        list_is_seventh = [False for _ in range(len(dataframe))]
    elif type_data == "ECPred":
        data_path = "data/datasets/ECPred_dataset/ECPred_dataset_test.json"
        dataframe = load_json_into_pandas_dataframe(data_path)
        list_sequences = dataframe["sequence"]
        list_labels = dataframe["EC Number"]
        list_is_seventh = [False for _ in range(len(dataframe))]
    elif type_data == "My_dataset_SwissProt_04":
        data_path = "data/datasets/test_set_diff_SwissProt_2021_01_et_2021_04/time_based_test_set_between_2021_01_and_2021_04.json"
        dataframe = load_json_into_pandas_dataframe(data_path)
        list_sequences = dataframe["sequence"]
        list_labels = dataframe["ec_number"]
        list_is_seventh = [False for _ in range(len(dataframe))]
    else:
        raise RuntimeError("type_data uknow")

    # Load the model
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    vocab = torch.load("data/models/" + vocab_path)
    path_class_vocab = (
        "/".join(model_path.split("/")[:-1])
        + "/classif_EC_pred_lvl_"
        + str(max_lvl)
        + "_vocab.pth"
    )
    class_vocab = torch.load(path_class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    weight_list, occurence_each_label = calc_proportion_class_training(
        inverse_class_vocab, type_data
    )

    TP = np.zeros(max_lvl)
    P = 0
    pred_non_enzyme = 0
    TP_without_seven = np.zeros(max_lvl)
    P_without_seven = 0
    pred_non_enzyme_without_seven = 0

    all_succed_class = []
    all_error_class = []

    all_occ_predicted_error = []
    all_occ_true_label_error = []

    # We calc the metric on all the dev set
    for sequence, label, is_seventh in tqdm(
        zip(list_sequences, list_labels, list_is_seventh), total=len(list_sequences)
    ):
        seq_ori = sequence
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        if recalibrate:
            proba_pred = recalibrate_proba(proba_pred, weight_list)
        pred_indice = int(np.argmax(proba_pred.cpu().numpy()))
        pred_class = inverse_class_vocab[pred_indice]
        # print(pred_class)

        if (
            pred_class
            == "0"
            + class_separator_model
            + "0"
            + class_separator_model
            + "0"
            + class_separator_model
            + "0"
        ):
            pred_non_enzyme += 1
            if not is_seventh:
                pred_non_enzyme_without_seven += 1
        list_pred_ec = label.split(",")
        if label_present_in_training(list_pred_ec, class_vocab):
            if pred_class not in list_pred_ec:
                ind_predicted = class_vocab[pred_class]
                ind_true_label = class_vocab[list_pred_ec[0]]
                print("#" * 89)
                print("Prediction error :")
                print("Predicted :", pred_class)
                print("Correct class :", list_pred_ec[0])
                print("Occurence predicted :", occurence_each_label[pred_class])
                print("Occurence true classes :", occurence_each_label[list_pred_ec[0]])
                print("Proba predicted :", proba_pred[ind_predicted])
                print("Proba true label :", proba_pred[ind_true_label])
                print("#" * 89)
                all_occ_predicted_error.append(occurence_each_label[pred_class])
                all_occ_true_label_error.append(occurence_each_label[list_pred_ec[0]])

        for lvl in range(1, max_lvl + 1):
            label_at_lvl = [
                class_seperator_dataset.join(
                    one_ec.split(class_seperator_dataset)[:lvl]
                )
                for one_ec in list_pred_ec
            ]

            pred_class_at_lvl = class_seperator_dataset.join(
                pred_class.split(class_separator_model)[:lvl]
            )
            # print("pred_class_at_lvl :", pred_class_at_lvl)
            # print("label_at_lvl :", label_at_lvl)
            if len(label_at_lvl) == 1 and lvl == 4:
                if pred_class_at_lvl in label_at_lvl:
                    all_succed_class.append(label_at_lvl[0])
                    all_succed_class = list(set(all_succed_class))
                else:
                    all_error_class.append(label_at_lvl[0])
                    all_error_class = list(set(all_error_class))
            if pred_class_at_lvl in label_at_lvl or (
                len(label_at_lvl) > 1 and count_correct_if_multiple_class
            ):
                TP[lvl - 1] += 1
                if not is_seventh:
                    TP_without_seven[lvl - 1] += 1
            else:
                if lvl == 5:  # 4 car 5 pas possible
                    print("Prediction not correct")
                    print("Len prot :", len(seq_ori))
                    print("True label :", label_at_lvl)
                    print("Predicted label :", pred_class_at_lvl)
                    print("Top labels :")
                    list_max_ind = np.flip(np.argsort(proba_pred.cpu().numpy()))[:5]
                    for ind in list_max_ind:
                        pred_class = inverse_class_vocab[ind]
                        print(proba_pred[ind])
                        print(pred_class)

        P += 1
        if not is_seventh:
            P_without_seven += 1

    print("For the full positive dataset of BENZ :")
    for lvl in range(1, max_lvl + 1):
        TRP = (TP[lvl - 1] / P) * 100  # =Recall
        print("TRP/recall at lvl", lvl, ":", TRP)
    FNR = (pred_non_enzyme / P) * 100
    print("FNR :", FNR)

    print("For the reduced positive dataset of BENZ :")
    for lvl in range(1, max_lvl + 1):
        TRP = (TP_without_seven[lvl - 1] / P_without_seven) * 100  # =Recall
        print("TRP/recall at lvl", lvl, ":", TRP)
    FNR = (pred_non_enzyme_without_seven / P_without_seven) * 100
    print("FNR :", FNR)

    print("all_succed_class :", all_succed_class)
    print("all_error_class : ", all_error_class)

    print("all_occ_predicted_error :", Counter(all_occ_predicted_error))
    print("all_occ_true_label_error :", Counter(all_occ_true_label_error))


def label_present_in_training(list_ec_label, vocab):
    for ec in list_ec_label:
        if ec in vocab.keys():
            return True
    return False


@torch.no_grad()
def calc_ensemble_all_metrics_on_positive(
    list_model_path, vocab_path, recalibrate=True
):
    max_seq_len = 1024
    max_lvl = 4  # 2
    class_separator = "."  # "-"

    # Load the data
    data_path = "data/datasets/BENZ_dataset/Correct_positive_BENZ_test_set.json"
    dataframe = load_json_into_pandas_dataframe(data_path)
    list_sequences = dataframe["sequence"]
    list_labels = dataframe["ec_number"]
    list_is_seventh = dataframe["is_seventh"]

    # Load the model
    list_models = []
    for path_choosen in list_model_path:
        complete_model_path = (
            path_choosen + "/classif_EC_pred_lvl_" + str(max_lvl) + ".pth"
        )
        model = torch.load(complete_model_path, map_location=torch.device("cpu"))
        model = model.eval()
        list_models.append(model)
    vocab = torch.load("data/models/" + vocab_path)
    path_class_vocab = (
        "/".join(complete_model_path.split("/")[:-1])
        + "/classif_EC_pred_lvl_"
        + str(max_lvl)
        + "_vocab.pth"
    )
    class_vocab = torch.load(path_class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    weight_list = calc_proportion_class_training(inverse_class_vocab)

    TP = np.zeros(max_lvl)
    P = 0
    pred_non_enzyme = 0
    TP_without_seven = np.zeros(max_lvl)
    P_without_seven = 0

    # We calc the metric on all the dev set
    for sequence, label, is_seventh in tqdm(
        zip(list_sequences, list_labels, list_is_seventh), total=len(list_sequences)
    ):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        list_pred = []
        for model in list_models:
            proba_pred = model(input_batch)[0]
            if recalibrate:
                proba_pred = recalibrate_proba(proba_pred, weight_list)
            list_pred.append(proba_pred.cpu().numpy())
        list_pred = np.array(list_pred)
        proba_pred = np.mean(list_pred, axis=0)
        # print(proba_pred.shape)
        pred_indice = int(np.argmax(proba_pred))
        pred_class = inverse_class_vocab[pred_indice]
        # print(pred_class)
        if (
            pred_class
            == "0"
            + class_separator
            + "0"
            + class_separator
            + "0"
            + class_separator
            + "0"
        ):
            pred_non_enzyme += 1

        list_pred_ec = label.split(",")

        for lvl in range(1, max_lvl + 1):
            label_at_lvl = [
                ".".join(one_ec.split(".")[:lvl]) for one_ec in list_pred_ec
            ]
            pred_class_at_lvl = ".".join(pred_class.split(class_separator)[:lvl])

            if pred_class_at_lvl in label_at_lvl:
                TP[lvl - 1] += 1
                if not is_seventh:
                    TP_without_seven[lvl - 1] += 1
        P += 1
        if not is_seventh:
            P_without_seven += 1

    print("For the full positive dataset of BENZ :")
    for lvl in range(1, max_lvl + 1):
        TRP = (TP[lvl - 1] / P) * 100  # =Recall
        print("TRP/recall at lvl", lvl, ":", TRP)
    FNR = (pred_non_enzyme / P) * 100
    print("FNR :", FNR)

    print("For the reduced positive dataset of BENZ :")
    for lvl in range(1, max_lvl + 1):
        TRP = (TP_without_seven[lvl - 1] / P_without_seven) * 100  # =Recall
        print("TRP/recall at lvl", lvl, ":", TRP)
    FNR = (pred_non_enzyme / P_without_seven) * 100
    print("FNR :", FNR)


@torch.no_grad()
def calc_all_metrics_on_negative(model_path, vocab_path, recalibrate=True):
    # Load the data
    data_path = "data/datasets/BENZ_dataset/Correct_negative_BENZ_test_set.json"
    dataframe = load_json_into_pandas_dataframe(data_path)
    list_sequences = dataframe["sequence"]

    # Load the model
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model = model.eval()
    vocab = torch.load("data/models/" + vocab_path)
    path_class_vocab = (
        "/".join(model_path.split("/")[:-1]) + "/classif_EC_pred_lvl_4_vocab.pth"
    )
    class_vocab = torch.load(path_class_vocab)
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    weight_list = calc_proportion_class_training(inverse_class_vocab)
    max_seq_len = 1024

    FP = 0
    N = 0

    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences, total=len(list_sequences)):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        if recalibrate:
            proba_pred = recalibrate_proba(proba_pred, weight_list)
        pred_indice = int(np.argmax(proba_pred.cpu().numpy()))
        pred_class = inverse_class_vocab[pred_indice]
        # print(pred_class)
        if pred_class != "0.0.0.0":
            FP += 1

        N += 1
        # FPR = (FP / N) * 100
        # print("Tmp FPR :", FPR)
    print("For the full negative dataset of BENZ :")
    FPR = (FP / N) * 100
    print("FPR :", FPR)
