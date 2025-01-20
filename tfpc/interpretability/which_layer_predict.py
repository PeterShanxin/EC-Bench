import torch
from tqdm import tqdm
import numpy as np
from interpretability.utils import (
    get_cls_token_repr,
    set_hook_to_get_repr_all_layers,
    load_sequences,
    load_model_and_vocab,
)


def get_final_projection(model):
    try:
        # norm1 = model.norm1
        classe_embedding = model.classe_embedding
    except:
        raise RuntimeError("Only work on layer norm pojection for now")
    return classe_embedding  # norm1,


def project_cls_to_decision(token_cls_all_layers, classe_embedding):
    proj_vector = None
    for token_cls in token_cls_all_layers:
        token_cls = torch.tensor(token_cls)
        output = classe_embedding(token_cls)
        # output = norm1(output)
        output = output.detach().numpy()
        output = output.reshape(-1, output.shape[0])
        if proj_vector is None:
            proj_vector = output
        else:
            proj_vector = np.concatenate((proj_vector, output))
    # We take the classe that was predicted by the last layer
    decision = np.argmax(proj_vector[-1])

    rank_of_pred = []
    # We see which position it was rank in other layer
    for proj in proj_vector:
        sort_prediction = np.flip(np.argsort(proj))
        result = np.where(sort_prediction == decision)
        result = int(result[0])
        rank_of_pred.append(result)
    return rank_of_pred


@torch.no_grad()
def position_final_label_cls(path_model, path_vocab, nb_seq):
    which_dataset = "EC_pred"
    max_seq_len = 1024

    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################
    list_sequences = load_sequences(which_dataset, nb_seq)
    model, vocab, pad_indice = load_model_and_vocab(path_model, path_vocab)

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################

    (
        activation,
        nb_layer,
        nb_head,
        list_hook,
    ) = set_hook_to_get_repr_all_layers(model)

    classe_embedding = get_final_projection(model)  # norm1,

    print("Nb layer :", nb_layer)
    print("Nb head :", nb_head)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################
    compteur = 0
    all_rank = []
    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        model(input_batch)

        token_cls = get_cls_token_repr(activation, nb_layer)

        rank_pred_each_layer = project_cls_to_decision(
            token_cls, classe_embedding  # norm1
        )
        all_rank.append(rank_pred_each_layer)

        compteur += 1

    all_rank = np.array(all_rank)
    all_rank_mean = all_rank.mean(axis=0)
    print(all_rank_mean.shape)
    print(all_rank_mean)
    ##########################################################################
    # STEP 5 : We delete the hook and get the model back to the original state#
    ##########################################################################
    # We delete the hook from the model
    for hook in list_hook:
        hook.remove()

    model = torch.nn.DataParallel(model)
    model = model.train()