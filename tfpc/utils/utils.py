"""
This module define some usefull function that are use in different file
"""

import sys
from os import path
import os
import subprocess
import logging
from collections import namedtuple
from types import MethodType
import ast
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from dataset_manager.training_dataset_manager import TrainingDatasetManager
from collections import Counter

# from logger.metrics_logger import MetricsLogger
from utils.functional_pytorch import perso_multi_head_attention_forward
from utils.functional_pytorch import perso_gradient_multi_head_attention_forward
from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
import pickle as pkl

# import inspect
# import torch.nn.functional as F
from utils.protBert_get_attn import forward
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from interpretability.utils import load_sequences, load_model_and_vocab
from interpretability.evaluate_interpretability_method import (
    get_scores,
    normalize_all_dict,
)
from utils.json_loader import load_json_into_pandas_dataframe


def function_not_implemented():
    raise RuntimeError("The function you ask is not implemented for now")


def select_output_lm(output, indices_masks):
    output_pred = None
    for num_batch, list_indices in enumerate(indices_masks):
        if output_pred is None:
            output_pred = output[num_batch][list_indices]
        else:
            output_pred = torch.cat((output_pred, output[num_batch][list_indices]))

    return output_pred


def move_tensor_to_device(unknonw_object, device):
    if isinstance(unknonw_object, list):
        for k, _ in enumerate(unknonw_object):
            if torch.is_tensor(unknonw_object[k]):
                unknonw_object[k] = unknonw_object[k].to(device)
    if torch.is_tensor(unknonw_object):
        unknonw_object = unknonw_object.to(device)
    return unknonw_object


@torch.no_grad()
def performance_with_more_and_more_mask(path_model, path_vocab, base_path, all_methods):
    dico_associated_best_norm = {
        "Rollout": "None",
        "GradCam": "None",
        "Attn_last_layer": "None",
        "Integreted_gradients": "unit_length_norm2",
        "LIME_5000_m": "unit_length_norm2",
        "LIME_25000_variations": "unit_length_norm2",
        "LIME_5000_X": "unit_length_norm2",
        "LIME_5000_p": "unit_length_norm2",
        "weighted_pageRank_each_head_follow_by_max": "unit_length_norm2",
        "Gradients": "unit_length_norm1",
        "InputXGrad": "unit_length_norm1",
        "LRP_rollout_cls": "unit_length_norm2",
        "LRP_rollout_cls_start_at_0": "unit_length_norm2",
        "LRP_rollout_sum_col_start_at_0": "unit_length_norm2",
        "LRP_rollout_sum_col": "unit_length_norm2",
        "max_follow_by_max": "None",  # A vérifier quel est le meilleurs pour cette méthode max_follow_by_max
        "mean_follow_by_max": "None",
        "mean_follow_by_mean": "None",
        "max_follow_by_mean": "None",
        "Raw_attention_sum_col": "None",
        "Raw_attention_cls_max": "Unknown",
    }
    all_normalization = [
        "unit_length_norm1",
        "unit_length_norm2",
        "std_mean",
        "min_max",
    ]
    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    which_dataset = "selected_res"
    max_len = 1024
    percent_mask_step = 2  # By x% mask step
    max_prop_mask = 95
    dataframe = load_json_into_pandas_dataframe(
        "data/datasets/swissprot_and_catalytic_atlas_selected_site/swissprot_and_catalytic_atlas_selected_site_train.json"
    )

    list_sequences = dataframe["sequence"]
    list_labels = dataframe["ec_number"]

    # Load scores from different interpretability methods
    logging.info("We load the interpretability score from dump file")
    dico_scores = dict()
    for method in all_methods:
        try:
            dico_scores[method] = get_scores(
                base_path, method, which_dataset=which_dataset
            )
        except:
            if which_dataset != "catalytic_site":
                dico_scores[method] = get_scores(
                    base_path, method, which_dataset="catalytic_site"
                )
            else:
                dico_scores[method] = get_scores(
                    base_path, method, which_dataset="selected_res"
                )
        for keys in dico_scores[method]:
            dico_scores[method][keys] = list(dico_scores[method][keys])

    # Add method with normalization
    logging.info("We calc dict with normalization")
    for method in all_methods:
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

    model, vocab, pad_indice = load_model_and_vocab(path_model, path_vocab)
    path_class_vocab = (
        "/".join(path_model.split("/")[:-1]) + "/classif_EC_pred_lvl_2_vocab.pth"
    )

    class_vocab = torch.load(path_class_vocab, map_location=torch.device("cpu"))
    inverse_class_vocab = {v: k for k, v in class_vocab.items()}

    dico_of_list_acc = dict()
    for method_name, scores_this_method in dico_scores.items():
        print(method_name)
        if "LIME" in method_name:
            is_LIME = True
        else:
            is_LIME = False
        list_acc = []
        for percent_mask in tqdm(range(0, max_prop_mask, percent_mask_step)):
            list_sequences_masked = modify_seq(
                list_sequences, scores_this_method, percent_mask, max_len, is_LIME
            )
            acc = performance_on_new_data(
                model,
                vocab,
                inverse_class_vocab,
                list_sequences_masked,
                list_labels,
            )
            list_acc.append(acc)
            print(list_acc)
        # plt.plot(list_acc, label=method_name)
        dico_of_list_acc[method_name] = list_acc

    # plt.show()
    if not os.path.isdir("data/eval_interpretability_evolution_acc"):
        os.mkdir("data/eval_interpretability_evolution_acc")
    torch.save(dico_of_list_acc, "data/eval_interpretability_evolution_acc/res.pkl")


def plot_result_eval_interpretability_decrease_perf():
    dico_of_list_acc = torch.load("data/eval_interpretability_evolution_acc/res.pkl")
    step = 2
    max_pourcent = 95
    pourcentage = list(range(0, max_pourcent, step))
    for method_name, list_acc in dico_of_list_acc.items():
        print(method_name)
        resp = input("Want to plot ?(Y/n)")
        if resp == "Y":
            plt.plot(pourcentage, list_acc, label=method_name)
    plt.xlabel("Pourcentage de token remplacer par mask")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    print(dico_of_list_acc)


def modify_seq(
    list_sequences, scores_this_method, percent_mask, max_len, is_LIME=False
):
    new_list_seq = []
    for seq in list_sequences:
        cut_seq = seq[:max_len]
        nb_token_to_mask = int(len(cut_seq) * (percent_mask / 100))
        if is_LIME:
            token_to_mask = np.flip(np.argsort(scores_this_method[seq]))[
                :nb_token_to_mask
            ]
        else:
            token_to_mask = np.flip(np.argsort(scores_this_method[cut_seq][:max_len]))[
                :nb_token_to_mask
            ]
        seq = list(cut_seq)
        for ind in token_to_mask:
            seq[ind] = "m"
        seq = "".join(seq)
        new_list_seq.append(seq)
    return new_list_seq


@torch.no_grad()
def performance_on_new_data(
    model, vocab, inverse_class_vocab, list_sequences, list_labels
):
    max_seq_len = 1024
    nb_correct = 0
    nb_total = 0
    # We calc the metric on all the dev set
    for sequence, labels in tqdm(
        zip(list_sequences, list_labels), total=len(list_sequences)
    ):
        if labels is None:
            continue
        labels = np.array(labels[:max_seq_len])
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        pred_indice = int(np.argmax(proba_pred.cpu().numpy()))
        pred_class = inverse_class_vocab[pred_indice]

        label = "-".join(labels[0].split(".")[:2])
        if pred_class == label:
            nb_correct += 1
        nb_total += 1
    acc = (nb_correct / nb_total) * 100
    return acc


# Calc metric juste on contact dataset in an unsupervised maneer
@torch.no_grad()
def calc_metric_on_auxiliary_data(
    dataloader,
    model,
    metric_logger,
    nb_batch_seen,
    typeEval,
):
    if hasattr(model, "module"):
        model = model.module
    model = model.eval()
    # On modifie la fonction de la bibliotheque pytorch pour permettre de récuperer l'attention de chaque tête et pas uniquement par layer
    ancienne_fonciton_lib = torch.nn.functional.multi_head_attention_forward
    torch.nn.functional.multi_head_attention_forward = (
        perso_multi_head_attention_forward
    )

    # Step 1 : Put probe in the network to extract all attention map
    # Step 2 : Forward pass on 20 proteines
    # Step 3 : Symetrization + APC
    # Step 4 : Fit an sklearn L1 logic regression with the 20 proteines
    # Step 5 : Forward pass on all the dev dataloard and record in metric_logger list with couple (nb_batch_seen,value_L/5 pred)

    # Creation d'un crochet pour récupérer l'attention lors de la passe forward
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    if hasattr(model, "transformer_embedder"):
        # Si c'est un modèle finetunné
        the_encoder = model.transformer_embedder.transformer_encoder
    else:
        # Si c'est un modèle juste pré entrainé
        the_encoder = model.transformer_encoder

    # Pour chaque couche on met un crochet pour recuperer les attentions
    NB_LAYER = len(the_encoder.layers)
    NB_HEAD = the_encoder.layers[0].self_attn.__dict__["num_heads"]

    list_hook = []
    for k in range(NB_LAYER):
        list_hook.append(
            the_encoder.layers[k].self_attn.register_forward_hook(
                get_activation("self_attn_" + str(k))
            )
        )

    subset_size = 100
    cuda = next(model.parameters()).is_cuda
    dataset_x_logistc_reg = []
    dataset_y_logistc_reg = []
    for input_batch, target_batch in dataloader:
        batch_size = input_batch.shape[0]
        if len(dataset_x_logistc_reg) < subset_size:
            if cuda:
                input_batch = input_batch.cuda()
            model(input_batch)
            attentions_map = new_get_all_attentions_map(
                activation, NB_LAYER, NB_HEAD, batch_size, target_batch
            )
            dataset_x_logistc_reg += attentions_map
            dataset_y_logistc_reg += target_batch
        else:
            break

    logging.info("We get the result on the %s proteins", subset_size)
    # We transform into nd array
    dataset_x_logistc_reg = [np.array(d) for d in dataset_x_logistc_reg]
    dataset_y_logistc_reg = [np.array(d) for d in dataset_y_logistc_reg]

    logging.info("We finish to pass into np array")
    # We symetrize and do an AVERAGE PRODUCT CORRECTION
    dataset_x_logistc_reg = apply_symetrize_and_APC(dataset_x_logistc_reg)
    logging.info("We finish to apply symetrize and APC")
    dataset_x_logistc_reg, dataset_y_logistc_reg = convert_for_logistic(
        dataset_x_logistc_reg,
        dataset_y_logistc_reg,
    )

    logging.info("We will train an l1 logistic regression")
    # We train a logistic regression with an l1 penatly
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.15,  # 1
        multi_class="ovr",
    ).fit(dataset_x_logistc_reg, dataset_y_logistc_reg)
    nb_contact = np.sum(dataset_y_logistc_reg)
    logging.debug(
        "There is %s contact on %s potential site",
        nb_contact,
        len(dataset_y_logistc_reg),
    )
    logging.debug("There is %s coeff", clf.coef_)
    logging.debug("Non zeros at indice %s", np.nonzero(clf.coef_)[1])
    logging.info("We finish to train the regression")

    all_precision = [0, 0]
    subset_eval_train = 5000
    compteur = 0
    # We calc the metric on all the dev set
    for input_batch, target_batch in dataloader:
        if compteur > subset_eval_train:
            break

        batch_size = input_batch.shape[0]
        if cuda:
            input_batch = input_batch.cuda()
        model(input_batch)
        attentions_map = new_get_all_attentions_map(
            activation, NB_LAYER, NB_HEAD, batch_size, target_batch
        )
        attentions_map = apply_symetrize_and_APC(attentions_map)
        for num_batch in range(batch_size):
            attn_this_batch = np.array(attentions_map[num_batch])
            len_prot = int(attn_this_batch.shape[1])
            x_features = attn_this_batch.transpose((1, 2, 0))
            this_target = target_batch[num_batch]

            indice_right_triangle = np.triu_indices(len_prot)
            # Matrice symetrique donc on prend que la partie triangle
            x_features = x_features[indice_right_triangle]
            this_target = this_target[indice_right_triangle]
            mask = this_target != -1
            this_target = this_target[mask]
            x_features = x_features[mask]
            if len(x_features) == 0:
                continue

            prediction = clf.predict_proba(x_features)
            prediction = torch.tensor([p[1] for p in prediction])
            how_many_top = min(len_prot, len(prediction))
            top_L_prediction_indice = torch.topk(prediction, how_many_top)[1]
            correct = (this_target[top_L_prediction_indice]).sum()

            all_precision[0] += correct.cpu().numpy()
            all_precision[1] += len_prot
        compteur += 1

    # We delete the hook from the model
    for hook in list_hook:
        hook.remove()
    # On rétablie la fonction correct dans la librairie pytorch
    torch.functional.multi_head_attention_forward = ancienne_fonciton_lib
    precision = (all_precision[0] / all_precision[1]) * 100
    logging.info("The metric value on auxiliary task is %s", precision)
    metric_logger.append(precision)
    model = torch.nn.DataParallel(model)
    model = model.train()


"""
def apply_symetrize_and_APC(data_x):
    #data_x of shape : [nb_exemples,nb_head*nb_layer,len_prot,len_prot]
    new_data_x = []
    for x in data_x:
        all_attn_map = []
        for attn_map in x:
            len_prot = attn_map.shape[0]
            symmetrize_attn_map = 0.5 * (attn_map + attn_map.T)
            APC_result = np.zeros(symmetrize_attn_map.shape)
            for i in range(len_prot):
                for j in range(len_prot):
                    Fi = np.sum(symmetrize_attn_map[:, j])
                    Fj = np.sum(symmetrize_attn_map[i, :])
                    F = np.sum(symmetrize_attn_map)
                    APC_result[i, j] = symmetrize_attn_map[i, j] - ((Fi * Fj) / F)
            all_attn_map.append(APC_result)
        new_data_x.append(all_attn_map)
    return new_data_x
"""


def apply_symetrize_and_APC(data_x):
    """
    data_x of shape : [nb_exemples,nb_head*nb_layer,len_prot,len_prot]
    """
    new_data_x = []
    for x in data_x:
        all_attn_map = []
        for attn_map in x:
            symmetrize_attn_map = 0.5 * (attn_map + attn_map.T)
            Q = np.sum(symmetrize_attn_map, axis=1).reshape(-1, 1)
            W = np.sum(symmetrize_attn_map, axis=0).reshape(1, -1)
            F = np.sum(symmetrize_attn_map)
            symmetrize_attn_map = symmetrize_attn_map - (np.matmul(Q, W) / F)
            all_attn_map.append(symmetrize_attn_map)
        new_data_x.append(all_attn_map)
    return new_data_x


def convert_for_logistic(data_x, data_y):
    """
    Convert shape :
    data_x of shape : [nb_exemples,nb_head*nb_layer,len_prot,len_prot]
    data_y of shape : [nb_exemple,len_prot,len_prot]
    To shape :
    new_data_x of shape : [nb_exemples*len_prot*len_prot,nb_head*nb_layer]
    new_data_y of shape : [nb_exemple*len_prot*len_prot]
    """
    new_data_x = []
    new_data_y = []
    for prot_x, prot_y in zip(data_x, data_y):
        prot_x = np.array(prot_x)
        prot_y = np.array(prot_y)
        len_prot = prot_y.shape[0]
        for i in range(len_prot):
            for j in range(i + 6, len_prot):
                if prot_y[i, j] != -1:
                    x_feature = prot_x[:, i, j]
                    y_label = prot_y[i, j]
                    new_data_x.append(x_feature)
                    new_data_y.append(y_label)
    return new_data_x, new_data_y


def transform_for_logist(x_data):
    x_data = np.array(x_data)
    new_data_x = []
    len_prot = x_data.shape[1]
    for i in range(len_prot):
        for j in range(len_prot):
            x_feature = x_data[:, i, j]
            new_data_x.append(x_feature)
    return new_data_x


def new_transform_for_logist(x_data):
    x_data = np.array(x_data)
    x_data = x_data.reshape((-1, x_data.shape[1] * x_data.shape[2]))
    x_data = x_data.transpose()
    return x_data


def get_all_attentions_map(activation, NB_LAYER, NB_HEAD, batch_size, target_batch):
    attention_map = []
    for num_batch in range(batch_size):
        len_prot_without_pad = len(target_batch[num_batch])
        attention_map_per_batch = []
        for layer in range(NB_LAYER):
            for head in range(NB_HEAD):
                matrice_attention_layer_l_batch_b = (
                    activation["self_attn_" + str(layer)][1][num_batch][head]
                    .detach()
                    .cpu()
                    .numpy()
                )
                matrice_attention_layer_l_batch_b = matrice_attention_layer_l_batch_b[
                    :len_prot_without_pad, :len_prot_without_pad
                ]
                attention_map_per_batch.append(matrice_attention_layer_l_batch_b)
        attention_map.append(attention_map_per_batch)
    return attention_map


def new_get_all_attentions_map(activation, NB_LAYER, NB_HEAD, batch_size, target_batch):
    attention_map = []
    for num_batch in range(batch_size):
        len_prot_without_pad = len(target_batch[num_batch])
        attention_map_per_batch = None
        for layer in range(NB_LAYER):
            matrice_attention_layer_l_batch_b = (
                activation["self_attn_" + str(layer)][1][num_batch]
                .detach()
                .cpu()
                .numpy()
            )
            matrice_attention_layer_l_batch_b = matrice_attention_layer_l_batch_b[
                :, :len_prot_without_pad, :len_prot_without_pad
            ]
            if attention_map_per_batch is None:
                attention_map_per_batch = matrice_attention_layer_l_batch_b
            else:
                attention_map_per_batch = np.concatenate(
                    (attention_map_per_batch, matrice_attention_layer_l_batch_b)
                )
        attention_map.append(attention_map_per_batch)
    return attention_map


@torch.no_grad()
def calc_metric_and_loss_on_data(
    dataloader,
    model,
    loss_logger,
    metric_logger,
    weights_modifiers,
    metrics_manager,
    ind_task,
    nb_batch_seen,
    typeEval,
):
    """
    This function calculate the metric on the dev/test dataloader, add each value to the metric manager,
    at the end calculate total metric value and log it into the metric_logger in the dev/test attribute.
    And after that reset the metric manager.
    """
    model = model.eval()
    # Show dev exemples on model and add loss and exemple to the metric manager
    compteur = 0
    for input_batch, target_batch in dataloader:
        if typeEval == "test":
            print("Avancement :", compteur, "/", len(dataloader))
        output_batch = model(input_batch)

        loss = weights_modifiers.one_eval_batch(ind_task, output_batch, target_batch)
        if loss_logger is not None:
            loss_logger.add_loss_tmp(typeEval, loss)
        metrics_manager.update_metrics(
            typeEval, ind_task, output_batch.cpu(), target_batch
        )
        compteur += 1
    if loss_logger is not None:
        loss_logger.finish_add_loss_tmp(typeEval, nb_batch_seen)

    for metric_name in metric_logger.all_metrics_names:
        value_metric = metrics_manager.get_metric_value(typeEval, ind_task, metric_name)
        metric_logger.add_metric(typeEval, metric_name, nb_batch_seen, value_metric)
        logging.info(
            "%s on %s on task number %s is %s",
            metric_name,
            typeEval,
            ind_task,
            str(value_metric),
        )

    metrics_manager.reset_all_metrics(typeEval, ind_task)
    model = model.train()


def get_material_and_infos():
    dico_info_version_and_material = dict()
    dico_info_version_and_material["pwd"] = os.getcwd()
    dico_info_version_and_material["Python VERSION"] = sys.version
    dico_info_version_and_material["pyTorch VERSION"] = torch.__version__
    dico_info_version_and_material["CUDNN VERSION"] = torch.backends.cudnn.version()
    dico_info_version_and_material["Number CUDA Devices:"] = torch.cuda.device_count()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--format=csv",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        result = result.stdout.decode("utf-8")
    except FileNotFoundError as except_value:
        logging.debug("%s", except_value)
        result = "nvidia-smi not installed"
    dico_info_version_and_material["__Devices"] = result
    if torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
    else:
        cuda_device = "No"
    dico_info_version_and_material["Active CUDA Device: GPU"] = cuda_device
    dico_info_version_and_material["Available devices"] = torch.cuda.device_count()
    return dico_info_version_and_material


def get_attribute_to_save_train_logger_and_save():
    attribute_to_save = [
        "all_nb_batch_log",
        "all_nb_batch_save",
        "nb_batch_seen",
        "num_batch_each_epoch",
        "all_losses_logger",
        "all_metrics_logger",
        "train_metrics_counter",
        "train_metrics_each",
        "material",
        "time_tracker",
        "finished_XP",
    ]
    return attribute_to_save


def get_slope(metric_val):
    if len(metric_val) > 3:
        filtered_metric = lowess(
            [el[1] for el in metric_val],
            [el[0] for el in metric_val],
            frac=1 / 3,
        )
        x_filter = [el[0] for el in filtered_metric]
        y_filter = [el[1] for el in filtered_metric]
        slope = (y_filter[-1] - y_filter[-2]) / (x_filter[-1] - x_filter[-2])
    else:
        raise RuntimeError("Not enough metric value to calculate slope correctly")
    return slope


def get_first_metric_dev_value(metrics_logger):
    all_metric_name = metrics_logger.get_all_metric_names()
    name_first_metric = all_metric_name[0]
    metric_val = metrics_logger.get_metric_couple("dev", name_first_metric)
    return metric_val


def return_pretty_time(time_in_sec):
    if time_in_sec > 60:
        secondes = time_in_sec % 60
        minutes = time_in_sec // 60
        if minutes > 60:
            heures = minutes // 60
            minutes = minutes % 60
            if heures > 24:
                jours = heures // 24
                heures = heures % 24
                return (
                    str(jours)
                    + " jours : "
                    + str(heures)
                    + " heures : "
                    + str(minutes)
                    + " minutes : "
                    + print_float(secondes)
                    + " secondes"
                )
            else:
                return (
                    str(heures)
                    + " heures :"
                    + str(minutes)
                    + " minutes : "
                    + print_float(secondes)
                    + " secondes"
                )
        else:
            return str(minutes) + " minutes : " + print_float(secondes) + " secondes"
    else:
        return str(time_in_sec)[:3] + "s"


def print_float(number):
    my_number = str(number)
    tab = my_number.split(".")
    return ".".join([tab[0], tab[1][:2]])


def save_fig(name, folder_saving):
    saving_folder = check_if_folder_exist(folder_saving)
    plt.savefig(saving_folder + name + ".pdf")


def check_if_folder_exist(folder_saving):
    if not path.exists(folder_saving):
        os.mkdir(folder_saving)
    return folder_saving


def square_pad_sequence(list_of_tensor, padding_value):
    batch_size = len(list_of_tensor)
    max_len = 0
    for t_value in list_of_tensor:
        longueur = t_value.shape[0]
        if longueur > max_len:
            max_len = longueur
    tenseur = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)
    for indice, t_value in enumerate(list_of_tensor):
        longueur_prot = t_value.shape[0]
        t_value = t_value.long()
        tenseur[indice, :longueur_prot, :longueur_prot] += t_value
        tenseur[indice, longueur_prot:, longueur_prot:] = padding_value
    return tenseur


def get_dataloader(
    dataset_name,
    batch_size,
    limit_size_input_prot,
    col_name_input,
    col_name_output,
    dataset_type,
    path_vocab,
):
    # Create dataloader for contact TEST dataset
    Config_named_tuple = namedtuple(
        "Config_named_tuple",
        ["root_datasets", "vocab", "tasks", "root_models", "starting_model_path_vocab"],
    )
    Task_named_tuple = namedtuple(
        "Task_named_tuple",
        [
            "data_path",
            "unique_task_name",
            "batch_size",
            "limit_size_input_prot",
            "col_name_input",
            "col_name_output",
            "dataset_type",
        ],
    )
    dico_type = {"name": dataset_type, "params": []}
    task_param = Task_named_tuple(
        dataset_name,
        "task",
        batch_size,
        limit_size_input_prot,
        col_name_input,
        col_name_output,
        dico_type,
    )
    config_task = Config_named_tuple(
        "data/datasets/",
        torch.load("data/models/" + path_vocab),
        [task_param],
        "data/models/",
        path_vocab,
    )
    dataset_manager = TrainingDatasetManager(config_task)
    dataloader = dataset_manager.get_train_dataloader(0)
    return dataloader


def get_pred_contact_from_attention_map(path_model, path_vocab):
    model = torch.load(path_model, map_location=torch.device("cpu"))
    dataloader = get_dataloader(
        "proteinnet",
        2,
        1024,
        "primary",
        "tertiary",
        "classification_contact_per_amino_acid",
        path_vocab,
    )
    all_metric = dict()
    all_metric["name"] = "contact"
    metric_logger = []
    nb_batch_seen = 0
    typeEval = "test"
    calc_metric_on_auxiliary_data(
        dataloader, model, metric_logger, nb_batch_seen, typeEval
    )
    return metric_logger[0]


def set_hook_to_get_attention_map(model):
    if hasattr(model, "module"):
        model = model.module
    model = model.eval()
    # On modifie la fonction de la bibliotheque pytorch pour permettre de récuperer l'attention de chaque tête et pas uniquement par layer
    ancienne_fonciton_lib = torch.nn.functional.multi_head_attention_forward
    torch.nn.functional.multi_head_attention_forward = perso_gradient_multi_head_attention_forward  # anciennement  perso_multi_head_attention_forward

    # Creation d'un crochet pour récupérer l'attention lors de la passe forward
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    is_protBert = False
    if hasattr(model, "transformer_embedder"):
        # Si c'est un modèle finetunné
        if hasattr(model.transformer_embedder, "ProtBert_BFD"):
            the_encoder = model.transformer_embedder.ProtBert_BFD.encoder
            is_protBert = True
        else:
            the_encoder = model.transformer_embedder.transformer_encoder
    elif hasattr(model, "ProtBert_BFD"):
        the_encoder = model.ProtBert_BFD.encoder
        print(the_encoder.layer[0].attention)
        is_protBert = True
    else:
        # Si c'est un modèle juste pré entrainé
        the_encoder = model.transformer_encoder

    if is_protBert:
        # Pour chaque couche on met un crochet pour recuperer les attentions
        NB_LAYER = len(the_encoder.layer)
        NB_HEAD = the_encoder.layer[0].attention.self.num_attention_heads
        print(NB_LAYER)
        print(NB_HEAD)

        list_hook = []
        for k in range(NB_LAYER):
            the_encoder.layer[k].attention.self.output_attentions = True
            the_encoder.layer[k].attention.self.forward = MethodType(
                forward, the_encoder.layer[k].attention.self
            )

            list_hook.append(
                the_encoder.layer[k].attention.self.register_forward_hook(
                    get_activation("self_attn_" + str(k))
                )
            )
    else:
        # Pour chaque couche on met un crochet pour recuperer les attentions
        NB_LAYER = len(the_encoder.layers)
        NB_HEAD = the_encoder.layers[0].self_attn.__dict__["num_heads"]

        list_hook = []
        for k in range(NB_LAYER):
            list_hook.append(
                the_encoder.layers[k].self_attn.register_forward_hook(
                    get_activation("self_attn_" + str(k))
                )
            )
    return activation, NB_LAYER, NB_HEAD, is_protBert, list_hook, ancienne_fonciton_lib


@torch.no_grad()
def calc_metric_1D_from_attn_map(
    path_model, path_vocab, which_dataset="catalytic_site"
):

    train_size = 1000
    max_seq_len = 1024
    penalty_logistic_reg = "l1"
    coeff_reg_logistic_reg = 1
    type_multiclass = "ovr"
    solver_logistic_reg = "liblinear"
    class_weight_logistic_reg = "balanced"
    # Just for catalytic residue prediction
    dico_find_roles = dict()

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################

    # We load the dataset with the sequence and as a label a liste of indices of interest
    # catalytic_site_train is the catalytic residue from Mechanism and Catalytic Site Atlas
    if which_dataset == "catalytic_site":
        input_col_name = "sequence"
        output_col_name = "catalytic_residue_position"
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/catalytic_site/catalytic_site_train.json"
        )
    elif which_dataset == "binding_site":
        input_col_name = "sequence"
        output_col_name = "label"
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/binding_site/binding_site_train.json"
        )
    else:
        logging.error(
            "%s : Ce dataset n'est pas encore supporté par la fonction", which_dataset
        )
    dataframe = dataframe.sample(frac=1, random_state=1).reset_index(drop=True)

    logging.info("Le dataframe contient %s protéines.", len(dataframe))
    # We load the model and the vocab
    if torch.cuda.is_available():
        model = torch.load(path_model)
        model = model.cuda()
    else:
        model = torch.load(path_model, map_location=torch.device("cpu"))
    vocab = torch.load("data/models/" + path_vocab)
    pad_indice = vocab["p"]

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################

    (
        activation,
        nb_layer,
        nb_head,
        is_protBert,
        list_hook,
        ancienne_fonciton_lib,
    ) = set_hook_to_get_attention_map(model)

    ##################################################################################################
    # STEP 3 : We train a logistic regression to detect amino acid that we want from the attention map#
    ##################################################################################################

    cuda = next(model.parameters()).is_cuda
    dataset_x_logistc_reg = []
    dataset_y_logistc_reg = []
    for index, row in dataframe.iterrows():
        for indi in row[output_col_name]:
            if indi > len(row[input_col_name]):
                print(row)
                raise RuntimeError("Valeur indice impossible")
        input_batch = torch.tensor([vocab[l] for l in row[input_col_name]]).unsqueeze(0)
        target_batch = row[output_col_name]
        batch_size = input_batch.shape[0]
        if len(dataset_x_logistc_reg) < train_size:
            if cuda:
                input_batch = input_batch.cuda()

            model(input_batch)
            attentions_map = get_attentions_map_simple(
                activation,
                nb_layer,
                nb_head,
                batch_size,
                input_batch,
                pad_indice,
                is_protBert,
            )

            for attn in attentions_map:
                assert len(attn[0]) == len(row[input_col_name])
                dataset_x_logistc_reg += [attn.mean(axis=1)]

            dataset_y_logistc_reg.append(target_batch)
        else:
            break

    if which_dataset == "catalytic_site":
        new_dataset_y_logistc_reg = []
        new_dataset_x_logistc_reg = []
        for attn_map, list_position in zip(
            dataset_x_logistc_reg, dataset_y_logistc_reg
        ):
            seq_len = len(attn_map[0])
            target = np.zeros(seq_len)
            target[list_position] = 1
            attn_map = attn_map.T
            for k in range(seq_len):
                new_dataset_x_logistc_reg.append(list(attn_map[k]))
                new_dataset_y_logistc_reg.append(target[k])

        dataset_x_logistc_reg = np.array(new_dataset_x_logistc_reg)
        dataset_y_logistc_reg = np.array(new_dataset_y_logistc_reg)
    elif which_dataset == "binding_site":
        new_dataset_y_logistc_reg = []
        new_dataset_x_logistc_reg = []
        for attn_map, target in zip(dataset_x_logistc_reg, dataset_y_logistc_reg):
            seq_len = len(attn_map[0])
            attn_map = attn_map.T
            for k in range(seq_len):
                new_dataset_x_logistc_reg.append(list(attn_map[k]))
                new_dataset_y_logistc_reg.append(target[k])

        dataset_x_logistc_reg = np.array(new_dataset_x_logistc_reg)
        dataset_y_logistc_reg = np.array(new_dataset_y_logistc_reg)
    else:
        logging.debug("ERREUR which_dataset")

    logging.info("We get the result on the %s proteins", train_size)

    logging.info("We finish to pass into np array")
    logging.info("We will train an l1 logistic regression")

    print(dataset_x_logistc_reg.shape)
    print(dataset_y_logistc_reg.shape)
    # We train a logistic regression with an l1 penatly
    clf = LogisticRegression(
        penalty=penalty_logistic_reg,
        solver=solver_logistic_reg,
        C=coeff_reg_logistic_reg,
        multi_class=type_multiclass,
        class_weight=class_weight_logistic_reg,
    ).fit(dataset_x_logistc_reg, dataset_y_logistc_reg)
    catalytic_res = np.sum(dataset_y_logistc_reg)
    logging.debug(
        "There is %s catalytic residue on %s potential site",
        catalytic_res,
        len(dataset_y_logistc_reg),
    )
    logging.debug("There is %s coeff", clf.coef_)
    logging.debug("Non zeros at indice %s", np.nonzero(clf.coef_)[1])
    logging.info("We finish to train the regression")

    #####################################################################################################
    # STEP 4 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################

    nb_positif_correct = 0
    nb_pred = 0
    nb_catalytic = 0
    pred_positive = 0
    compteur = 0
    list_ident_correc = []
    # We calc the metric on all the dev set
    for index, row in dataframe.iterrows():
        if compteur < train_size:
            compteur += 1
            continue
        if compteur % 50 == 0:
            print((compteur / len(dataframe)) * 100, "% effectué")
        len_seq = len(row[input_col_name])
        if len_seq > max_seq_len:
            row[input_col_name] = row[input_col_name][:max_seq_len]
            new_cat = []
            for ind in row[output_col_name]:
                if ind < max_seq_len:
                    new_cat.append(ind)
            row[output_col_name] = new_cat
        len_seq = len(row[input_col_name])

        input_batch = torch.tensor([vocab[l] for l in row[input_col_name]]).unsqueeze(0)
        target_batch = row[output_col_name]

        if which_dataset == "catalytic_site":
            target = np.zeros(len_seq)
            target[target_batch] = 1
        elif which_dataset == "binding_site":
            target = np.array(target_batch)
        else:
            logging.debug("ERREUR which_dataset2")

        batch_size = input_batch.shape[0]
        if cuda:
            input_batch = input_batch.cuda()
        model(input_batch)
        attentions_map = get_attentions_map_simple(
            activation,
            nb_layer,
            nb_head,
            batch_size,
            input_batch,
            pad_indice,
            is_protBert,
        )

        attentions_map = attentions_map[0].mean(axis=1)

        assert len(attentions_map[0]) == len_seq
        attentions_map = attentions_map.T
        if which_dataset == "catalytic_site":
            all_roles = row["roles"].split(";")
            roles_per_indice = {}
            for r in all_roles:
                ind = r.split(":")[0]
                list_roles = r.split(":")[1]
                roles_per_indice[ind] = list_roles
        for k in range(len_seq):
            x_features = attentions_map[k]
            prediction = clf.predict_proba([x_features])[0]
            pred_catalytic = np.argmax(prediction)
            pred_positive += pred_catalytic
            if pred_catalytic == 1:
                nb_positif_correct += pred_catalytic == target[k]
                if target[k] == 1 and which_dataset == "catalytic_site":
                    identifiant_site = compteur * max_seq_len + k
                    list_ident_correc.append(identifiant_site)
                    if str(k) in roles_per_indice.keys():
                        tab = ast.literal_eval(roles_per_indice[str(k)])
                        for r in tab:
                            if r in dico_find_roles.keys():
                                dico_find_roles[r] += 1
                            else:
                                dico_find_roles[r] = 1
                    else:
                        if "unknown" in dico_find_roles.keys():
                            dico_find_roles["unknown"] += 1
                        else:
                            dico_find_roles["unknown"] = 1

            nb_catalytic += target[k]
            nb_pred += 1

        compteur += 1

    rappel = (nb_positif_correct / nb_catalytic) * 100
    precision = (nb_positif_correct / pred_positive) * 100
    f1_score = 2 * ((precision * rappel) / (precision + rappel))
    logging.info(
        "There are "
        + str(nb_pred)
        + " amino acids to test, and "
        + str(nb_catalytic)
        + " catalytics sites."
    )
    logging.info("On catalytic site detection :")
    logging.info("The recall is %s", rappel)
    logging.info("The precision is %s", precision)
    logging.info("The F1 score is %s", f1_score)

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
    return rappel, precision, f1_score, dico_find_roles, list_ident_correc


def get_attentions_map_simple(
    activation, NB_LAYER, NB_HEAD, batch_size, input_batch, pad_indice, is_protBert
):
    attention_map = []
    for num_batch in range(batch_size):
        len_prot_without_pad = (input_batch[num_batch] != pad_indice).sum()
        attention_map_per_batch = None
        for layer in range(NB_LAYER):
            matrice_attention_layer_l_batch_b = (
                activation["self_attn_" + str(layer)][1][num_batch]
                .detach()
                .cpu()
                .numpy()
            )

            matrice_attention_layer_l_batch_b = matrice_attention_layer_l_batch_b[
                :, :len_prot_without_pad, :len_prot_without_pad
            ]
            matrice_attention_layer_l_batch_b = np.array(
                matrice_attention_layer_l_batch_b
            )

            if attention_map_per_batch is None:
                attention_map_per_batch = matrice_attention_layer_l_batch_b
            else:
                attention_map_per_batch = np.concatenate(
                    (attention_map_per_batch, matrice_attention_layer_l_batch_b)
                )

        attention_map.append(np.array(attention_map_per_batch))
    return attention_map


def get_attentions_map_and_gradient(
    activation, NB_LAYER, NB_HEAD, batch_size, input_batch, pad_indice, is_protBert
):
    attention_map = []
    gradient_attention_map = []
    for num_batch in range(batch_size):
        len_prot_without_pad = (input_batch[num_batch] != pad_indice).sum()
        attention_map_per_batch = None
        gradient_attention_map_per_batch = None
        for layer in range(NB_LAYER):
            matrice_attention_layer_l_batch_b = activation["self_attn_" + str(layer)][
                1
            ][num_batch].cpu()
            gradient_matrice_attention_layer_l_batch_b = activation[
                "self_attn_" + str(layer)
            ][1].grad[num_batch]

            matrice_attention_layer_l_batch_b = matrice_attention_layer_l_batch_b[
                :, :len_prot_without_pad, :len_prot_without_pad
            ]
            gradient_matrice_attention_layer_l_batch_b = (
                gradient_matrice_attention_layer_l_batch_b[
                    :, :len_prot_without_pad, :len_prot_without_pad
                ]
            )

            if attention_map_per_batch is None:
                attention_map_per_batch = matrice_attention_layer_l_batch_b
                gradient_attention_map_per_batch = (
                    gradient_matrice_attention_layer_l_batch_b
                )
            else:
                attention_map_per_batch = torch.cat(
                    (attention_map_per_batch, matrice_attention_layer_l_batch_b)
                )
                gradient_attention_map_per_batch = torch.cat(
                    (
                        gradient_attention_map_per_batch,
                        gradient_matrice_attention_layer_l_batch_b,
                    )
                )

        attention_map.append(attention_map_per_batch.detach().numpy())
        gradient_attention_map.append(gradient_attention_map_per_batch.numpy())
    return attention_map, gradient_attention_map


@torch.no_grad()
def scoreBERTOLOGY_from_attn_map(
    path_model,
    path_vocab,
    which_dataset,
    random_result=False,
    grad_attn_for_suppervised=False,
    seuil=None,
):  # which_dataset = "binding_site" or "catalytic_site"
    max_seq_len = 1024
    folder_save = "data/result_xp_prop_attn_map_unsupervised/"
    # /result_prop_attn_map_unsupervised_data/models/pre_trained_models/30_layer_uniparc+BFD/model_embedder.pth
    extract_name = "_".join(path_model.split("/")[2:4])
    name_output_file = (
        "result_prop_attn_map_unsupervised_"
        + extract_name
        + "_"
        + which_dataset
        + "_random_"
        + str(random_result)
        + "_seuil_"
        + str(seuil)
        + "_grad_attn_"
        + str(grad_attn_for_suppervised)
    )

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################

    # We load the dataset with the sequence and as a label a liste of indices of interest
    # catalytic_site_train is the catalytic residue from Mechanism and Catalytic Site Atlas
    if which_dataset == "catalytic_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/catalytic_site/catalytic_site_train.json"
        )
        list_catalytic_site = []
        for index, row in dataframe.iterrows():
            arr = np.zeros((len(row["sequence"]))).astype(int)
            tab = row["catalytic_residue_position"]
            arr[tab] = 1
            list_catalytic_site.append(list(arr))
        # Exemple row["roles"] for one line : 90:['metal ligand', 'electrostatic stabiliser'];88:['electrostatic stabiliser'];90:['metal ligand'];63:['proton donor', 'proton acceptor'];93:['metal ligand'];61:['metal ligand']

        dataframe = dataframe.assign(label=list_catalytic_site)
    elif which_dataset == "binding_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/binding_site_with_roles/binding_site_with_roles_train.json"
        )
    else:
        logging.error(
            "%s : Ce dataset n'est pas encore supporté par la fonction", which_dataset
        )
    dataframe = dataframe.sample(frac=1, random_state=1).reset_index(drop=True)

    logging.info("Le dataframe contient %s protéines.", len(dataframe))
    # We load the model and the vocab
    if torch.cuda.is_available():
        model = torch.load(path_model)
        model = model.cuda()
    else:
        model = torch.load(path_model, map_location=torch.device("cpu"))

    vocab = torch.load("data/models/" + path_vocab)
    pad_indice = vocab["p"]

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################

    (
        activation,
        nb_layer,
        nb_head,
        is_protBert,
        list_hook,
        ancienne_fonciton_lib,
    ) = set_hook_to_get_attention_map(model)

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)
    #####################################################################################################
    # STEP 3 : We create a dict with all possibles roles#
    #####################################################################################################
    roles_per_attn_head = []
    for _ in range(nb_layer):
        for _ in range(nb_head):
            dict_attn_roles_head_k = dict()
            for raw_roles in dataframe["roles"]:
                if raw_roles == "":
                    continue
                tab_r = raw_roles.split(";")
                for t in tab_r:
                    list_roles_res_i = ast.literal_eval(t.split(":")[1])
                    for role in list_roles_res_i:
                        if role not in dict_attn_roles_head_k.keys():
                            dict_attn_roles_head_k[role] = 0
            roles_per_attn_head.append(dict_attn_roles_head_k)

    #####################################################################################################
    # STEP 4 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################

    sum_each_head_on_binding = np.zeros((nb_layer, nb_head))
    if random_result:
        sum_each_head_on_random = np.zeros((nb_layer, nb_head))
        if grad_attn_for_suppervised:
            sum_each_head_gradient_on_random = np.zeros((nb_layer, nb_head))
            sum_each_head_gradient_mul_attn_on_random = np.zeros((nb_layer, nb_head))
    if grad_attn_for_suppervised:
        sum_each_head_gradient_on_binding = np.zeros((nb_layer, nb_head))
        sum_each_head_gradient_mul_attn_on_binding = np.zeros((nb_layer, nb_head))
        sum_each_head_total_gradient = np.zeros((nb_layer, nb_head))
        sum_each_head_total_gradient_mul_attn = np.zeros((nb_layer, nb_head))
    sum_each_head_total = np.zeros((nb_layer, nb_head))
    compteur = 0
    # We calc the metric on all the dev set
    for index, row in dataframe.iterrows():
        if compteur % 10 == 0:
            print((compteur / len(dataframe)) * 100, "% effectué")
        len_seq = len(row["sequence"])
        if len_seq > max_seq_len:
            row["sequence"] = row["sequence"][:max_seq_len]
            row["label"] = row["label"][:max_seq_len]
            if which_dataset == "binding_site":
                row["mask"] = row["mask"][:max_seq_len]
        len_seq = len(row["sequence"])

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in row["sequence"]]
        ).unsqueeze(0)
        target = row["label"]
        tab_raw_roles = row["roles"].split(";")
        if tab_raw_roles == [""]:
            roles = dict()
        else:
            roles = {
                int(tab.split(":")[0]): ast.literal_eval(tab.split(":")[1])
                for tab in tab_raw_roles
            }

        if which_dataset == "binding":
            mask = row["mask"]
        else:
            mask = [1 for _ in range(len(target))]
        # Traitement target et mask
        mask_ori = mask
        mask = np.array([mask])
        mask_carrer = np.matmul(mask.T, mask).astype(bool)

        target = np.array(target) & np.array(mask_ori)

        target = np.array([target])
        one_vector = np.ones((len(target[0])))
        one_vector = one_vector.reshape((-1, len(one_vector)))
        target_carrer = np.matmul(one_vector.T, target).astype(bool)
        if random_result:
            target_random = target.copy()
            np.random.shuffle(target_random[0])
            target_carrer_random = np.matmul(one_vector.T, target_random).astype(
                bool
            )  # Tirage aléatoire de position "intéressante"

        batch_size = 1
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        out = model(input_batch)
        if grad_attn_for_suppervised:
            for k in range(nb_layer):
                activation["self_attn_" + str(k)][1].retain_grad()
            argmax = torch.argmax(out[0])
            out[0][argmax].backward()

        if grad_attn_for_suppervised:
            attentions_map, gradient_attention_map = get_attentions_map_and_gradient(
                activation,
                nb_layer,
                nb_head,
                batch_size,
                input_batch,
                pad_indice,
                is_protBert,
            )

            gradient_attention_map = gradient_attention_map[0]
            gradient_attention_map = gradient_attention_map[:, 1:, 1:]

            # On prend que les gradients positif, c'est à dire ceux qui augmente la probabilité de la classe majoritaire.
            # gradient_attention_map[gradient_attention_map < 0] = 0
            new_grad = []
            for grad in gradient_attention_map:
                grad[grad < 0] = 0
                new_grad.append(grad)
            gradient_attention_map = new_grad
        else:
            attentions_map = get_attentions_map_simple(
                activation,
                nb_layer,
                nb_head,
                batch_size,
                input_batch,
                pad_indice,
                is_protBert,
            )
        attentions_map = attentions_map[0]
        attentions_map = attentions_map[:, 1:, 1:]

        if seuil is not None:
            attentions_map[attentions_map < seuil] = 0

        for num_layer in range(nb_layer):
            for num_head in range(nb_head):
                somme_all = attentions_map[num_layer * nb_head + num_head]
                somme_all = somme_all[mask_carrer]
                somme_all = somme_all.sum()
                if grad_attn_for_suppervised:
                    somme_all_gradient = gradient_attention_map[
                        num_layer * nb_head + num_head
                    ]

                    somme_all_gradient = somme_all_gradient[mask_carrer]
                    somme_all_gradient = somme_all_gradient.sum()

                    somme_all_gradiant_mul_attn = np.multiply(
                        attentions_map[num_layer * nb_head + num_head],
                        gradient_attention_map[num_layer * nb_head + num_head],
                    )
                    somme_all_gradiant_mul_attn = somme_all_gradiant_mul_attn[
                        mask_carrer
                    ]
                    somme_all_gradiant_mul_attn = somme_all_gradiant_mul_attn.sum()

                res = attentions_map[num_layer * nb_head + num_head]
                if grad_attn_for_suppervised:
                    res_gradient = gradient_attention_map[
                        num_layer * nb_head + num_head
                    ]

                for position_residue, list_roles_one_res in roles.items():
                    if position_residue < max_seq_len:
                        for role in list_roles_one_res:
                            somme_attn_per_col = res.sum(axis=0)
                            roles_per_attn_head[num_layer * nb_head + num_head][
                                role
                            ] += somme_attn_per_col[position_residue]
                res_new = res[target_carrer]
                new_res = res_new.sum()

                if grad_attn_for_suppervised:
                    res_new_gradient = res_gradient[target_carrer]
                    new_res_gradient = res_new_gradient.sum()

                    res_new_gradient_mul_attn = np.multiply(res_gradient, res)[
                        target_carrer
                    ]
                    new_res_grad_mul_attn = res_new_gradient_mul_attn.sum()
                # print("new_res :", new_res)
                # print("somme_all :", somme_all)
                # assert new_res <= somme_all
                if random_result:
                    resR = res[target_carrer_random]
                    res_random = resR.sum()
                    sum_each_head_on_random[num_layer][num_head] += res_random
                    if grad_attn_for_suppervised:
                        # Grad
                        resG = res_gradient[target_carrer_random]
                        res_random_grad = resG.sum()
                        sum_each_head_gradient_on_random[num_layer][
                            num_head
                        ] += res_random_grad
                        # Grad mul attn
                        resGA = res_new_gradient_mul_attn = np.multiply(
                            res_gradient, res
                        )[target_carrer_random]
                        res_random_grad_mul_attn = resGA.sum()
                        sum_each_head_gradient_mul_attn_on_random[num_layer][
                            num_head
                        ] += res_random_grad_mul_attn
                if grad_attn_for_suppervised:
                    sum_each_head_gradient_on_binding[num_layer][
                        num_head
                    ] += new_res_gradient
                    sum_each_head_gradient_mul_attn_on_binding[num_layer][
                        num_head
                    ] += new_res_grad_mul_attn
                    sum_each_head_total_gradient[num_layer][
                        num_head
                    ] += somme_all_gradient
                    sum_each_head_total_gradient_mul_attn[num_layer][
                        num_head
                    ] += somme_all_gradiant_mul_attn
                sum_each_head_on_binding[num_layer][num_head] += new_res
                sum_each_head_total[num_layer][num_head] += somme_all

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

    if random_result and grad_attn_for_suppervised:
        all_data_to_save = [
            sum_each_head_on_random,
            sum_each_head_gradient_on_random,
            sum_each_head_gradient_mul_attn_on_random,
            sum_each_head_on_binding,
            sum_each_head_gradient_on_binding,
            sum_each_head_gradient_mul_attn_on_binding,
            sum_each_head_total,
            sum_each_head_total_gradient,
            sum_each_head_total_gradient_mul_attn,
            roles_per_attn_head,
        ]
    elif random_result:
        all_data_to_save = [
            sum_each_head_on_random,
            sum_each_head_on_binding,
            sum_each_head_total,
            roles_per_attn_head,
        ]
    elif grad_attn_for_suppervised:
        all_data_to_save = [
            sum_each_head_on_binding,
            sum_each_head_gradient_on_binding,
            sum_each_head_gradient_mul_attn_on_binding,
            sum_each_head_total,
            sum_each_head_total_gradient,
            sum_each_head_total_gradient_mul_attn,
            roles_per_attn_head,
        ]
    else:
        all_data_to_save = [
            sum_each_head_on_binding,
            sum_each_head_total,
            roles_per_attn_head,
        ]
    pkl.dump(all_data_to_save, open(folder_save + name_output_file + ".pkl", "wb"))
    return np.divide(sum_each_head_on_binding, sum_each_head_total)


def calc_InputXGrad(path_model, path_vocab, which_dataset, frac):
    max_seq_len = 1024
    folder_save = "data/residues_of_interest/Gradients/"
    folder_save_grad = "data/residues_of_interest/InputXGrad/"
    extract_name = "_".join(path_model.split("/")[2:4])
    name_output_file = "score_with_" + extract_name + "_on_dataset_" + which_dataset

    if which_dataset == "catalytic_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/catalytic_site/catalytic_site_train.json"
        )
        list_catalytic_site = []
        for index, row in dataframe.iterrows():
            arr = np.zeros((len(row["sequence"]))).astype(int)
            tab = row["catalytic_residue_position"]
            arr[tab] = 1
            list_catalytic_site.append(list(arr))
        # Exemple row["roles"] for one line : 90:['metal ligand', 'electrostatic stabiliser'];88:['electrostatic stabiliser'];90:['metal ligand'];63:['proton donor', 'proton acceptor'];93:['metal ligand'];61:['metal ligand']

        dataframe = dataframe.assign(label=list_catalytic_site)
    elif which_dataset == "binding_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/binding_site_with_roles/binding_site_with_roles_train.json"
        )
    else:
        logging.error(
            "%s : Ce dataset n'est pas encore supporté par la fonction", which_dataset
        )
    dataframe = dataframe.sample(frac=frac, random_state=1).reset_index(drop=True)

    # We load the model and the vocab
    if torch.cuda.is_available():
        model = torch.load(path_model)
        model = model.cuda()
    else:
        model = torch.load(path_model, map_location=torch.device("cpu"))

    model = model.eval()
    vocab = torch.load("data/models/" + path_vocab)
    pad_indice = vocab["p"]

    ######################################################
    # STEP 2 : We hook the model to get embedding#
    ######################################################
    # Creation d'un crochet pour récupérer l'attention lors de la passe forward

    print(model.transformer_embedder.ProtBert_BFD.embeddings.word_embeddings)

    # Creation d'un crochet pour récupérer l'attention lors de la passe forward
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    model.transformer_embedder.ProtBert_BFD.embeddings.word_embeddings.register_forward_hook(
        get_activation("embedding")
    )

    ############################################################
    # STEP 3 : We evaluate the model on the rest of the dataset#
    ############################################################

    # We calc the metric on all the dev set
    compteur = 0

    dico_scores = dict()
    dico_scores_grad = dict()
    dico_labels = dict()
    for index, row in tqdm(dataframe.iterrows(),total=len(dataframe)):
        compteur += 1
        len_seq = len(row["sequence"])
        if len_seq > max_seq_len:
            row["sequence"] = row["sequence"][:max_seq_len]
            row["label"] = row["label"][:max_seq_len]
            if which_dataset == "binding_site":
                row["mask"] = row["mask"][:max_seq_len]
        len_seq = len(row["sequence"])
        sequence = row["sequence"]
        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in row["sequence"]]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        out = model(input_batch)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

        activation["embedding"].retain_grad()
        argmax = torch.argmax(out[0])

        out[0][argmax].backward()

        input_embedding = activation["embedding"][0].detach()
        gradient_embedding = activation["embedding"].grad[0]
        # On enleve le token CLS
        input_embedding = input_embedding[1:]
        gradient_embedding = gradient_embedding[1:]

        out_score = torch.mul(gradient_embedding, input_embedding)
        out_score = torch.pow(torch.pow(out_score, 2).sum(dim=1), 1 / 2)
        gradient_embedding = torch.pow(
            torch.pow(gradient_embedding, 2).sum(dim=1), 1 / 2
        )

        labels = row["label"]
        labels = np.array(labels[:max_seq_len])
        dico_scores_grad[sequence] = gradient_embedding
        dico_scores[sequence] = out_score
        dico_labels[sequence] = labels
        """
        out_score = out_score / out_score.sum()

        nb_total_res += len(row["sequence"])
        top_k = int(0.05 * len(row["sequence"]))
        nb_total_pred += top_k

        best_indice_pred = np.flip(np.argsort(out_score.numpy()))[:top_k]
        indice_target = np.nonzero(row["label"])[0]

        indice_random = np.arange(len(out_score))
        np.random.shuffle(indice_random)
        indice_random = indice_random[:top_k]

        prop_on_site += out_score[indice_target].sum()
        total_sum += out_score.sum()

        top_10 = np.flip(np.argsort(out_score.numpy()))[:ok_top_num]
        is_ok = False
        for t in top_10:
            if t in indice_target:
                is_ok = True
        if is_ok:
            nb_prot_ok += 1

        top_10_random = np.arange(len(out_score))
        np.random.shuffle(top_10_random)
        top_10_random = top_10_random[:ok_top_num]
        is_ok_random = False
        for t in top_10_random:
            if t in indice_target:
                is_ok_random = True
        if is_ok_random:
            nb_prot_ok_random += 1
        nb_total_prot += 1

        nb_find_InputXGrad += len(
            list(set(best_indice_pred).intersection(set(indice_target)))
        )
        nb_find_random += len(
            list(set(list(indice_random)).intersection(set(list(indice_target))))
        )
        nb_total += len(indice_target)
        """
    """
    recall_InputXGrad = nb_find_InputXGrad / nb_total
    recall_random = nb_find_random / nb_total
    precision_InputXGrad = nb_find_InputXGrad / nb_total_pred
    precision_random = nb_find_random / nb_total_pred
    print("recall_InputXGrad :", recall_InputXGrad * 100, "%")
    print("Precision recall_InputXGrad :", precision_InputXGrad * 100, "%")
    print("recall_random :", recall_random * 100, "%")
    print("Precision random :", precision_random * 100, "%")

    proportion_propriete = float(prop_on_site / total_sum)
    base_prop = nb_total / nb_total_res
    print("proportion_propriete : ", proportion_propriete * 100, "%")
    print("Proportion des résidues avec la propriété :", base_prop * 100, "%")

    proportion_prot_ok = nb_prot_ok / nb_total_prot
    proportion_prot_ok_random = nb_prot_ok_random / nb_total_prot
    print(
        "Proportion with at least one residue detected in the top3 :",
        proportion_prot_ok * 100,
        "%",
    )
    print(
        "Proportion with at least one residue detected in random 3 :",
        proportion_prot_ok_random * 100,
        "%",
    )
    """
    results = [dico_scores, dico_labels]
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)

    torch.save(results, open(folder_save + name_output_file + ".pkl", "wb"))

    results = [dico_scores_grad, dico_labels]
    if not os.path.exists(folder_save_grad):
        os.mkdir(folder_save_grad)
    torch.save(results, open(folder_save_grad + name_output_file + ".pkl", "wb"))


def encode(sequence):
    global vocab
    tv = torch.tensor([vocab["c"]] + [vocab[l] for l in sequence]).unsqueeze(0)
    return tv


# define a prediction function
def prediction_function(batch_sequences):
    global model
    batch_resul = None
    for sequence in tqdm(batch_sequences):
        input = encode(sequence)
        if torch.cuda.is_available():
            input = input.cuda()
        outputs = model(input).detach().cpu()  # .numpy()
        if batch_resul is None:
            batch_resul = outputs
        else:
            batch_resul = torch.cat((batch_resul, outputs))
    return batch_resul.numpy()


def get_label_from_lime(model, explainer, sequence, nb_ex_per_prot):
    input_batch = encode(sequence)
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
    outputs = model(input_batch).detach().cpu()
    argmax = int(torch.argmax(outputs[0]))

    exp = explainer.explain_instance(
        sequence,
        prediction_function,
        num_samples=nb_ex_per_prot,
        num_features=100,
        top_labels=1,
    )  # default num_samples=5000

    map_result = exp.as_map()[argmax]
    list_of_position = np.array([m[0] for m in map_result])
    list_of_weight = np.array([m[1] for m in map_result])

    scores = np.zeros((len(sequence)))
    for pos, score in zip(list_of_position, list_of_weight):
        scores[pos] = score

    return scores


def print_recall_precision(nb_find_InputXGrad, nb_find_random, nb_total, nb_total_pred):
    recall_InputXGrad = nb_find_InputXGrad / nb_total
    recall_random = nb_find_random / nb_total
    precision_InputXGrad = nb_find_InputXGrad / nb_total_pred
    precision_random = nb_find_random / nb_total_pred
    print("Recall_InputXGrad :", recall_InputXGrad * 100, "%")
    print("Precision InputXGrad :", precision_InputXGrad * 100, "%")
    print("Recall random :", recall_random * 100, "%")
    print("Precision random :", precision_random * 100, "%")


@torch.no_grad()
def calc_LIME_interpretability(
    path_model, path_vocab, which_dataset, choose_frac, nb_ex_per_prot, base_string
):
    global model
    max_seq_len = 1024
    folder_save = (
        "data/residues_of_interest/LIME_with_"
        + str(nb_ex_per_prot)
        + "corupt_seq_with_"
        + base_string
        + "_replacment_character"
        + "/"
    )
    extract_name = "_".join(path_model.split("/")[2:4])

    name_output_file = "score_with_" + extract_name + "_on_dataset_" + which_dataset

    max_seq_len = 1024
    if which_dataset == "catalytic_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/catalytic_site/catalytic_site_train.json"
        )
        list_catalytic_site = []
        for index, row in dataframe.iterrows():
            arr = np.zeros((len(row["sequence"]))).astype(int)
            tab = row["catalytic_residue_position"]
            arr[tab] = 1
            list_catalytic_site.append(list(arr))
        # Exemple row["roles"] for one line : 90:['metal ligand', 'electrostatic stabiliser'];88:['electrostatic stabiliser'];90:['metal ligand'];63:['proton donor', 'proton acceptor'];93:['metal ligand'];61:['metal ligand']

        dataframe = dataframe.assign(label=list_catalytic_site)
    elif which_dataset == "binding_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/binding_site_with_roles/binding_site_with_roles_train.json"
        )
    else:
        logging.error(
            "%s : Ce dataset n'est pas encore supporté par la fonction", which_dataset
        )
    dataframe = dataframe.sample(frac=choose_frac).reset_index(drop=True)

    # We load the model and the vocab
    if torch.cuda.is_available():
        logging.info("J'utilise le GPU")
        model = torch.load(path_model)
        model = model.cuda()
    else:
        model = torch.load(path_model, map_location=torch.device("cpu"))

    model = model.eval()

    global vocab
    vocab = torch.load("data/models/" + path_vocab)

    ############################################################
    # STEP 2 : We evaluate the model on the rest of the dataset#
    ############################################################

    # We calc the metric on all the dev set
    compteur = 0

    explainer = LimeTextExplainer(char_level=True, bow=False, mask_string=base_string)

    dico_scores = dict()
    dico_labels = dict()

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        compteur += 1
        sequence = row["sequence"]
        len_seq = len(row["sequence"])
        if len_seq > max_seq_len:
            row["sequence"] = row["sequence"][:max_seq_len]
            row["label"] = row["label"][:max_seq_len]
            if which_dataset == "binding_site":
                row["mask"] = row["mask"][:max_seq_len]
        len_seq = len(row["sequence"])

        list_of_scores = get_label_from_lime(
            model, explainer, row["sequence"], nb_ex_per_prot
        )

        indice_target = row["label"]

        dico_scores[sequence] = list(list_of_scores)
        dico_labels[sequence] = list(indice_target)

    results = [dico_scores, dico_labels]
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)
    torch.save(results, open(folder_save + name_output_file + ".pkl", "wb"))


def create_dict_per_level_for_metric(metric_type, dataframe):
    dico = {}
    for _, row in dataframe.iterrows():
        row_metric_type = row["metric_type"]
        if row_metric_type == metric_type:
            dico[row["level"]] = row
    return dico


def annalyse_dataset_difficulty(
    dataset_name, name_col_ec_train, name_col_ec_test, name_col_seq, test_name
):
    import collections

    dataset_folder = "data/datasets/" + dataset_name + "/"
    df_test = pd.read_json(dataset_folder + test_name + ".json")
    print(
        "Mean len test sequences :",
        np.mean([len(seq) for seq in df_test[name_col_seq]]),
    )
    print(df_test.columns)
    df_train = pd.read_json(dataset_folder + dataset_name + "_train.json")
    print(df_train.columns)
    # Cut at level 2 to compare the two dataset at the same level
    df_test[name_col_ec_test] = df_test[name_col_ec_test].apply(
        lambda ec: ".".join(ec.split(".")[:2])
    )
    df_train[name_col_ec_train] = df_train[name_col_ec_train].apply(
        lambda ec: ".".join(ec.split(".")[:2])
    )
    print(df_test.head())
    test_counter = collections.Counter(df_test[name_col_ec_test])
    train_counter = collections.Counter(df_train[name_col_ec_train])
    all_ec_test = list(test_counter.keys())
    nb_ex_train = []
    nb_ex_test = []
    for ec in all_ec_test:
        nb_ex_test.append(test_counter[ec])
        nb_ex_train.append(train_counter[ec])

    order = np.argsort(nb_ex_test)
    nb_ex_test = [nb_ex_test[k] for k in order]
    nb_ex_train = [nb_ex_train[k] for k in order]
    all_ec_test = [all_ec_test[k] for k in order]
    x = np.array(list(range(len(all_ec_test))))
    plt.bar(x - 0.15, nb_ex_test, 0.3, label="test_number")
    plt.bar(x + 0.15, nb_ex_train, 0.3, label="train_number")
    plt.xticks(x, all_ec_test, rotation=90)
    plt.legend()
    plt.show()

