import torch
from tqdm import tqdm
import csv
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from interpretability.utils import (
    get_attentions_map_simple,
    set_hook_to_get_attention_map,
    load_sequences,
    load_model_and_vocab,
)


@torch.no_grad()
def histogram_of_attn_depending_on_property(
    path_model, path_vocab, which_dataset, nb_seq
):  # which_dataset = "binding_site" or "catalytic_site"
    max_seq_len = 1024

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences, list_labels = load_sequences(which_dataset, nb_seq, with_label=True)
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
    all_pic_binding = np.zeros((51))
    all_pic_not_binding = np.zeros((51))

    nb_pic_binding = []
    nb_pic_not_binding = []

    nb_binding = 0
    nb_not_binding = 0

    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences):
        sequence = sequence[:max_seq_len]
        label = list_labels[compteur]

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

        # On enleve le CLS
        attentions_map = attentions_map[:, 1:, 1:]

        num_head = 268
        # for num_head in range(480):
        if len(sequence) > 50:
            head_i = attentions_map[num_head]
            nb_residues = head_i.shape[0]
            for num_residue in range(nb_residues):
                attn_on_res_j = head_i[:, num_residue]
                if np.max(attn_on_res_j) > 0.4:
                    attn_on_res_j[attn_on_res_j < 0.1] = 0
                    peaks, _ = find_peaks(attn_on_res_j, height=0)

                    # plt.plot(peaks, attn_on_res_j[peaks], "x")
                    if label[num_residue] == 1:
                        nb_pic_binding.append(len(peaks))
                        # all_pic_binding += center_on_pic
                        # plt.plot(attn_on_res_j, linestyle="dashed", label="binding")
                        nb_binding += 1
                    else:
                        nb_pic_not_binding.append(len(peaks))
                        # all_pic_not_binding += center_on_pic
                        # plt.plot(attn_on_res_j, label="not_binding")
                        nb_not_binding += 1
            # plt.show()
        compteur += 1

    plt.boxplot([nb_pic_binding, nb_pic_not_binding], labels=["binding", "not_binding"])
    plt.legend()
    plt.show()
    """
    all_pic_binding = all_pic_binding / nb_binding
    all_pic_not_binding = all_pic_not_binding / nb_not_binding
    plt.plot(all_pic_binding, color="red", label="binding")
    plt.plot(all_pic_not_binding, color="blue", label="not binding")
    plt.legend()
    plt.show()
    """
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