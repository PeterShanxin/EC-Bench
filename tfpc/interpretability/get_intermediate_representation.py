import torch
from tqdm import tqdm
import numpy as np
from interpretability.utils import (
    set_hook_to_get_repr_all_layers,
    load_sequences,
    load_model_and_vocab,
    set_hook_to_get_repr_all_layers_before_FFD,
)
import pickle as pkl
from sklearn.decomposition import NMF
import inspect
from interpretability.create_NMF_projector import get_all_tokens_repr


@torch.no_grad()
def get_intermediate_representation_of_residues(
    path_model, path_vocab, which_dataset, nb_seq, where="after_FFD"
):
    max_seq_len = 1024

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences = load_sequences(which_dataset, nb_seq)
    model, vocab, pad_indice = load_model_and_vocab(path_model, path_vocab)

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################
    if where == "after_FFD":
        (
            activation,
            nb_layer,
            nb_head,
            list_hook,
        ) = set_hook_to_get_repr_all_layers(model)
    elif where == "before_FFD":
        (
            activation,
            nb_layer,
            nb_head,
            list_hook,
        ) = set_hook_to_get_repr_all_layers_before_FFD(model)
    else:
        raise RuntimeError("Not correct where, only after_FFD and before_FFD accepted")

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################
    compteur = 0
    all_projection_all_tokens = []
    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        model(input_batch)

        all_token_repr = get_all_tokens_repr(
            activation,
            nb_layer,
        )

        all_token_repr = all_token_repr.T

        # print(all_token_repr.shape)

        all_projection_all_tokens.append(list(all_token_repr))

        compteur += 1

    pkl.dump(
        [list_sequences, all_projection_all_tokens],
        open("data/NMF_projections/projections.pkl", "wb"),
    )

    ##########################################################################
    # STEP 5 : We delete the hook and get the model back to the original state#
    ##########################################################################
    # We delete the hook from the model
    for hook in list_hook:
        hook.remove()

    model = torch.nn.DataParallel(model)
    model = model.train()