import torch
from tqdm import tqdm
import csv
import pickle as pkl
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import numpy as np
import inspect
import os

from interpretability.utils import (
    get_attentions_map_simple,
    set_hook_to_get_attention_map,
    load_sequences,
    load_model_and_vocab,
)


def get_scores(lig, model, input_batch, ref_token, n_steps):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    output = model(input_batch)
    output = output.cpu().detach().numpy()

    pred_class = int(np.argmax(output))

    # pred_class = torch.tensor([pred_class])
    seq_length = input_batch.shape[1]
    # generate reference indices for each sample
    token_reference = TokenReferenceBase(reference_token_idx=ref_token)
    reference_indices = token_reference.generate_reference(
        seq_length, device=device
    ).unsqueeze(0)

    attributions = lig.attribute(
        input_batch,
        reference_indices,
        n_steps=n_steps,
        target=pred_class,
        internal_batch_size=1,
    )

    attributions = attributions.sum(dim=2).squeeze(0)
    # attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    # [1:] because we don't take the cls token into account
    return attributions[1:]


@torch.no_grad()
def score_on_residue_with_integreted_gradients(
    path_model, path_vocab, which_dataset, nb_seq, n_steps=500, base_string="p"
):
    """
    Input :
    which_dataset = "binding_site" or "catalytic_site"
    n_steps : 500
    """
    max_seq_len = 1024

    folder_save = "data/residues_of_interest/Integreted_gradients/"
    extract_name = "_".join(path_model.split("/")[2:4])
    name_output_file = "score_with_" + extract_name + "_on_dataset_" + which_dataset

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

    lig = LayerIntegratedGradients(
        model, model.transformer_embedder.ProtBert_BFD.embeddings
    )

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

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        model(input_batch)
        list_res = get_scores(lig, model, input_batch, vocab[base_string], n_steps)

        dico_scores[sequence] = list_res
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
