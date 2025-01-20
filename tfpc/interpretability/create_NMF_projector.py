import torch
from tqdm import tqdm
import numpy as np
from interpretability.utils import (
    set_hook_to_get_repr_all_layers,
    load_sequences,
    load_model_and_vocab,
)
import pickle as pkl
from sklearn.decomposition import NMF
import inspect


def get_all_tokens_repr(activation, nb_layer):
    all_cls_tokens = None
    for layer in range(nb_layer):
        matrice_repr_layer_l = (
            activation["residue_repr_layer_" + str(layer)][0][0][1:]
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


@torch.no_grad()
def create_NMF_projection(
    path_model, path_vocab, dim_projection, which_dataset, nb_seq
):
    max_seq_len = 1024

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences = load_sequences(which_dataset, nb_seq)
    model, vocab, pad_indice = load_model_and_vocab(path_model, path_vocab)

    """
    print(model.transformer_embedder.ProtBert_BFD.encoder.layer[0])
    print(
        inspect.getsource(
            model.transformer_embedder.ProtBert_BFD.encoder.layer[
                0
            ].intermediate.intermediate_act_fn
        )
    )
    """

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################
    (
        activation,
        nb_layer,
        nb_head,
        list_hook,
    ) = set_hook_to_get_repr_all_layers(model)

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################
    compteur = 0
    all_repr_all_tokens = None
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
        if all_repr_all_tokens is None:
            all_repr_all_tokens = all_token_repr
        else:
            all_repr_all_tokens = np.concatenate((all_repr_all_tokens, all_token_repr))

        compteur += 1

    print(all_repr_all_tokens.shape)
    all_repr_all_tokens[all_repr_all_tokens < 0] = 0
    nb_zero = (all_repr_all_tokens == 0).sum()
    nb_total = all_repr_all_tokens.shape[0] * all_repr_all_tokens.shape[1]
    print("nb_zero :", nb_zero)
    print("nb_total :", nb_total)
    print("Proportion :", (nb_zero / nb_total) * 100, "%")
    model_NMF = NMF(
        n_components=dim_projection,
        init="random",
        random_state=0,
        verbose=1,
        max_iter=200,
    )
    model_NMF.fit(all_repr_all_tokens)

    pkl.dump(
        model_NMF,
        open("data/models_NMF/NMF_binding_size_" + str(dim_projection) + ".pkl", "wb"),
    )

    ##########################################################################
    # STEP 5 : We delete the hook and get the model back to the original state#
    ##########################################################################
    # We delete the hook from the model
    for hook in list_hook:
        hook.remove()

    model = torch.nn.DataParallel(model)
    model = model.train()