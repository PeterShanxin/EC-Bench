import torch
import logging
import pandas as pd
from utils.functional_pytorch import perso_gradient_multi_head_attention_forward
from types import MethodType
import numpy as np
import torch
from utils.protBert_get_attn import forward
import sys
from tqdm import tqdm
import csv
import pickle as pkl
from utils.json_loader import load_json_into_pandas_dataframe


def get_attentions_map_simple(activation, NB_LAYER, input_batch, pad_indice):
    len_prot_without_pad = (input_batch[0] != pad_indice).sum()
    attention_map_per_batch = None
    for layer in range(NB_LAYER):
        matrice_attention_layer_l_batch_b = (
            activation["self_attn_" + str(layer)][1][0].detach().cpu().numpy()
        )

        matrice_attention_layer_l_batch_b = matrice_attention_layer_l_batch_b[
            :, :len_prot_without_pad, :len_prot_without_pad
        ]
        matrice_attention_layer_l_batch_b = np.array(matrice_attention_layer_l_batch_b)

        if attention_map_per_batch is None:
            attention_map_per_batch = matrice_attention_layer_l_batch_b
        else:
            attention_map_per_batch = np.concatenate(
                (attention_map_per_batch, matrice_attention_layer_l_batch_b)
            )

    return np.array(attention_map_per_batch)


def get_combi_head(activation, NB_LAYER, input_batch, pad_indice):
    len_prot_without_pad = (input_batch[0] != pad_indice).sum()
    combi_head_per_batch = None
    for layer in range(NB_LAYER):
        matrice_combination_layer_l_batch_b = (
            activation["combi_head_" + str(layer)][1][0].detach().cpu().numpy()
        )

        matrice_combination_layer_l_batch_b = matrice_combination_layer_l_batch_b[
            :, :len_prot_without_pad, :len_prot_without_pad
        ]
        matrice_combination_layer_l_batch_b = np.array(
            matrice_combination_layer_l_batch_b
        )

        if combi_head_per_batch is None:
            combi_head_per_batch = matrice_combination_layer_l_batch_b
        else:
            combi_head_per_batch = np.concatenate(
                (combi_head_per_batch, matrice_combination_layer_l_batch_b)
            )

    return np.array(combi_head_per_batch)


def get_cls_token_repr(activation, NB_LAYER):
    all_cls_tokens = None
    for layer in range(NB_LAYER):
        matrice_repr_layer_l = (
            activation["residue_repr_layer_" + str(layer)][0][0][0]
            .detach()
            .cpu()
            .numpy()
        )
        matrice_repr_layer_l = matrice_repr_layer_l.reshape(
            -1, matrice_repr_layer_l.shape[0]
        )

        if all_cls_tokens is None:
            all_cls_tokens = matrice_repr_layer_l
        else:
            all_cls_tokens = np.concatenate((all_cls_tokens, matrice_repr_layer_l))

    return np.array(all_cls_tokens)


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

    isProtBert = False
    if hasattr(model, "transformer_embedder"):
        # Si c'est un modèle finetunné
        if hasattr(model.transformer_embedder, "ProtBert_BFD"):
            the_encoder = model.transformer_embedder.ProtBert_BFD.encoder
            isProtBert = True
        else:
            the_encoder = model.transformer_embedder.transformer_encoder
    elif hasattr(model, "ProtBert_BFD"):
        the_encoder = model.ProtBert_BFD.encoder
        print(the_encoder.layer[0].attention)
        isProtBert = True
    else:
        # Si c'est un modèle juste pré entrainé
        the_encoder = model.transformer_encoder

    if isProtBert:
        # Pour chaque couche on met un crochet pour recuperer les attentions
        NB_LAYER = len(the_encoder.layer)
        NB_HEAD = the_encoder.layer[0].attention.self.num_attention_heads

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

    return activation, NB_LAYER, NB_HEAD, list_hook, ancienne_fonciton_lib


def get_weight_head_combi(model):
    isProtBert = False
    if hasattr(model, "transformer_embedder"):
        # Si c'est un modèle finetunné
        if hasattr(model.transformer_embedder, "ProtBert_BFD"):
            the_encoder = model.transformer_embedder.ProtBert_BFD.encoder
            isProtBert = True
        else:
            the_encoder = model.transformer_embedder.transformer_encoder
    elif hasattr(model, "ProtBert_BFD"):
        the_encoder = model.ProtBert_BFD.encoder
        isProtBert = True
    else:
        # Si c'est un modèle juste pré entrainé
        the_encoder = model.transformer_encoder

    if isProtBert:
        # Pour chaque couche on met un crochet pour recuperer les attentions
        NB_LAYER = len(the_encoder.layer)

        list_weight_head_combi = []
        for k in range(NB_LAYER):
            list_weight_head_combi.append(
                the_encoder.layer[k].attention.output.dense.weight
            )

    else:
        raise RuntimeError("Not implemented yet")

    return list_weight_head_combi


def set_hook_to_get_repr_all_layers(model):
    if hasattr(model, "module"):
        model = model.module
    model = model.eval()

    # Creation d'un crochet pour récupérer l'attention lors de la passe forward
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    if hasattr(model, "transformer_embedder"):
        # Si c'est un modèle finetunné
        if hasattr(model.transformer_embedder, "ProtBert_BFD"):
            the_encoder = model.transformer_embedder.ProtBert_BFD.encoder
        else:
            raise RuntimeError("We need a ProtBert model")
    elif hasattr(model, "ProtBert_BFD"):
        the_encoder = model.ProtBert_BFD.encoder
    else:
        # Si c'est un modèle juste pré entrainé
        the_encoder = model.transformer_encoder

    # Pour chaque couche on met un crochet pour recuperer les attentions
    NB_LAYER = len(the_encoder.layer)
    NB_HEAD = the_encoder.layer[0].attention.self.num_attention_heads

    list_hook = []
    for k in range(NB_LAYER):
        list_hook.append(
            the_encoder.layer[k].register_forward_hook(
                get_activation("residue_repr_layer_" + str(k))
            )
        )

    return activation, NB_LAYER, NB_HEAD, list_hook


def set_hook_to_get_repr_all_layers_before_FFD(model):
    if hasattr(model, "module"):
        model = model.module
    model = model.eval()

    # Creation d'un crochet pour récupérer l'attention lors de la passe forward
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    if hasattr(model, "transformer_embedder"):
        # Si c'est un modèle finetunné
        if hasattr(model.transformer_embedder, "ProtBert_BFD"):
            the_encoder = model.transformer_embedder.ProtBert_BFD.encoder
        else:
            raise RuntimeError("We need a ProtBert model")
    elif hasattr(model, "ProtBert_BFD"):
        the_encoder = model.ProtBert_BFD.encoder
    else:
        # Si c'est un modèle juste pré entrainé
        the_encoder = model.transformer_encoder

    # Pour chaque couche on met un crochet pour recuperer les attentions
    NB_LAYER = len(the_encoder.layer)
    NB_HEAD = the_encoder.layer[0].attention.self.num_attention_heads

    list_hook = []
    for k in range(NB_LAYER):
        list_hook.append(
            the_encoder.layer[k].attention.register_forward_hook(
                get_activation("residue_repr_layer_" + str(k))
            )
        )

    return activation, NB_LAYER, NB_HEAD, list_hook


def get_info_GPU_memory():
    t = torch.cuda.get_device_properties(0).total_memory / 1000000000
    r = torch.cuda.memory_reserved(0) / 1000000000
    a = torch.cuda.memory_allocated(0) / 1000000000
    f = r - a  # free inside reserved
    print("#" * 89)
    print("Total memory :", t, "Gb")
    print("Reserved memory :", t, "Gb")
    print("Allocated memory :", a, "Gb")
    print("Free memory :", f, "Gb")
    print("#" * 89)


def load_sequences(which_dataset, nb_seq, with_label=False, with_roles=False):
    # We load the dataset with the sequence and as a label a liste of indices of interest
    # catalytic_site_train is the catalytic residue from Mechanism and Catalytic Site Atlas
    col_sequence = "sequence"
    if which_dataset == "catalytic_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/catalytic_site/catalytic_site_train.json"
        )
        list_catalytic_site = []
        for _, row in dataframe.iterrows():
            arr = np.zeros((len(row["sequence"]))).astype(int)
            tab = row["catalytic_residue_position"]
            arr[tab] = 1
            list_catalytic_site.append(list(arr))
        # Exemple row["roles"] for one line : 90:['metal ligand', 'electrostatic stabiliser'];88:['electrostatic stabiliser'];90:['metal ligand'];63:['proton donor', 'proton acceptor'];93:['metal ligand'];61:['metal ligand']

        dataframe = dataframe.assign(label=list_catalytic_site)
        col_label_name = "label"
        col_roles_name = "roles"
    elif which_dataset == "binding_site_uniprot_without_catalytic_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/catalytic_site/M-CSA_infos_for_3D_representations.json"
        )
        col_sequence = "sequence_uniprot"
        col_label_name = "residue_of_interests"
        col_roles_name = ""

        list_selected_site = []
        for _, row in dataframe.iterrows():
            arr = np.zeros((len(row[col_sequence]))).astype(int)
            tab = np.array(row["resid_binding_uniprot"]) - 1
            tab = tab.astype(int)
            arr[tab] = 1
            list_selected_site.append(list(arr))

        dataframe = dataframe.assign(residue_of_interests=list_selected_site)

    elif which_dataset == "binding_site":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/binding_site_with_roles/binding_site_with_roles_train.json"
        )
        col_label_name = "label"
        col_roles_name = "roles"
    elif which_dataset == "selected_res":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/swissprot_and_catalytic_atlas_selected_site/swissprot_and_catalytic_atlas_selected_site_train.json"
        )
        list_selected_site = []
        for _, row in dataframe.iterrows():
            arr = np.zeros((len(row["sequence"]))).astype(int)
            tab = row["residue_of_interests"]
            arr[tab] = 1
            list_selected_site.append(list(arr))
        # Exemple row["roles"] for one line : 90:['metal ligand', 'electrostatic stabiliser'];88:['electrostatic stabiliser'];90:['metal ligand'];63:['proton donor', 'proton acceptor'];93:['metal ligand'];61:['metal ligand']

        dataframe = dataframe.assign(residue_of_interests=list_selected_site)
        col_label_name = "residue_of_interests"
        col_roles_name = "types"
    elif which_dataset == "EC_pred":
        dataframe = load_json_into_pandas_dataframe(
            "data/datasets/EC_prediction/EC_prediction_valid.json"
        )
        col_label_name = "label"
        col_roles_name = "roles"
    else:
        logging.error(
            "%s : Ce dataset n'est pas encore supporté par la fonction", which_dataset
        )
    if nb_seq == "all":
        nb_seq = len(dataframe)
    frac_for_nb_seq = nb_seq / len(dataframe)
    dataframe = dataframe.sample(frac=frac_for_nb_seq, random_state=1).reset_index(
        drop=True
    )
    if which_dataset == "EC_pred":
        return list(dataframe["primary"])
    else:
        if with_label:
            if with_roles:
                return (
                    list(dataframe[col_sequence]),
                    list(dataframe[col_label_name]),
                    list(dataframe[col_roles_name]),
                )
            else:
                return list(dataframe[col_sequence]), list(dataframe[col_label_name])
        else:
            return list(dataframe[col_sequence])


def load_model_and_vocab(path_model, path_vocab):
    # We load the model and the vocab
    if torch.cuda.is_available():
        model = torch.load(path_model)
        model = model.cuda()
    else:
        model = torch.load(path_model, map_location=torch.device("cpu"))

    model = model.eval()
    vocab = torch.load("data/models/" + path_vocab)
    pad_indice = vocab["p"]
    return model, vocab, pad_indice
