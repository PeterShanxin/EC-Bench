import torch
from tqdm import tqdm
import csv
import pickle as pkl
import os

from transformers.file_utils import FAISS_IMPORT_ERROR
from LRP.convert_ProtBert_to_LRP_model import convertProtBertToLRPModel
from LRP.ExplanationGenerator import Generator
import numpy as np

from interpretability.utils import (
    load_sequences,
    load_model_and_vocab,
    get_info_GPU_memory,
)


def score_on_residue_with_LRP_rollout(
    path_model, path_vocab, which_dataset, nb_seq, method="cls_and_somme_col"
):
    """
    Input :
    which_dataset = "binding_site" or "catalytic_site"
    n_steps : 500
    """
    max_seq_len = 1024

    if method == "cls_and_somme_col":
        folder_save = "data/residues_of_interest/LRP_rollout_cls/"
        folder_save2 = "data/residues_of_interest/LRP_rollout_sum_col/"
    elif method == "rollout":
        folder_save = "data/residues_of_interest/Rollout/"
    elif method == "gradCam":
        folder_save = "data/residues_of_interest/GradCam/"
    elif method == "attn_last_layer":
        folder_save = "data/residues_of_interest/Attn_last_layer/"
    else:
        raise RuntimeError("Method not supported")
    extract_name = "_".join(path_model.split("/")[2:4])
    name_output_file = "score_with_" + extract_name + "_on_dataset_" + which_dataset

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences, list_labels = load_sequences(
        which_dataset, nb_seq, with_label=True, with_roles=False
    )
    model, vocab, _ = load_model_and_vocab(path_model, path_vocab)

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################

    model = convertProtBertToLRPModel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    explanations = Generator(model)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################
    compteur = 0

    dico_scores = dict()
    dico_scores2 = dict()
    dico_scores3 = dict()
    dico_labels = dict()

    # We calc the metric on all the dev set
    for sequence, labels in tqdm(
        zip(list_sequences, list_labels), total=len(list_sequences)
    ):
        labels = np.array(labels[:max_seq_len])
        sequence = sequence[:max_seq_len]
        # print("Taille séquence sans CLS :", len(sequence))

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        # generate an explanation for the input
        if method == "cls_and_somme_col":
            try:
                expl = explanations.generate_LRP(
                    input_ids=input_batch, start_layer=11
                )  # start_layer=0
            except:
                print("I think it's an RAM error")
                input_batch = input_batch.cpu()
                model = model.cpu()
                expl = explanations.generate_LRP(
                    input_ids=input_batch, start_layer=11, force_cpu=True
                )  # start_layer=0
                model = model.cuda()

            # rollout_cls = expl[0, 0, 1:]
            # dico_scores[sequence] = rollout_cls.detach().cpu().numpy()
            rollout_somme_col = expl[0]
            rollout_somme_col = rollout_somme_col.sum(dim=0)
            rollout_somme_col = rollout_somme_col[1:]
            dico_scores2[sequence] = rollout_somme_col.detach().cpu().numpy()

        elif method == "rollout":
            expl = explanations.generate_rollout(input_ids=input_batch, start_layer=0)
        elif method == "gradCam":
            expl = explanations.generate_attn_gradcam(input_ids=input_batch)
        elif method == "attn_last_layer":
            expl = explanations.generate_attn_last_layer(input_ids=input_batch)
        dico_scores[sequence] = expl.detach().cpu().numpy()
        dico_labels[sequence] = labels

        compteur += 1

    ##########################################################################
    # STEP 5 : We delete the hook and get the model back to the original state#
    ##########################################################################

    results = [
        dico_scores,
        dico_labels,
    ]

    if not os.path.exists(folder_save):
        os.mkdir(folder_save)

    torch.save(results, open(folder_save + name_output_file + ".pkl", "wb"))

    if method == "cls_and_somme_col":
        results2 = [
            dico_scores2,
            dico_labels,
        ]

        if not os.path.exists(folder_save2):
            os.mkdir(folder_save2)
        torch.save(results2, open(folder_save2 + name_output_file + ".pkl", "wb"))
