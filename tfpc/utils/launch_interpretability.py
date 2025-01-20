from interpretability.generate_residues_of_interests_with_attn import (
    score_on_residue_with_attn_map,
    score_on_residue_with_attn_map_multiprocess,
)

from interpretability.generate_residues_of_interests_with_integreted_gradients import (
    score_on_residue_with_integreted_gradients,
)

from interpretability.generate_residues_of_interests_with_LRP_rollout import (
    score_on_residue_with_LRP_rollout,
)
from utils.utils import (
    calc_InputXGrad,
    calc_LIME_interpretability,
)
import logging


def launch_interpretability(
    choose_method, path_choosen, training_analyser, dataset_name, nb_seq
):
    if (
        "_follow_by_" in choose_method
        or "flowmax" in choose_method
        or "pageRank" in choose_method
    ):
        # Quick and can be done on GPU
        score_on_residue_with_attn_map_multiprocess(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            choose_method,
            one_output_vec=True,
        )
    elif choose_method == "raw_attention_sum":
        score_on_residue_with_attn_map(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            "somme_col",
        )
    elif choose_method == "raw_attention_cls":
        score_on_residue_with_attn_map(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            "attn_on_cls",
        )
    elif choose_method == "grad_and_inputXgrad":
        # Quick and can be done on GPU
        if nb_seq == "all":
            frac = 1
        else:
            frac = nb_seq / 992
        calc_InputXGrad(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            frac,
        )
    elif choose_method == "lime":
        # Very long and can be done on GPU
        nb_ex_per_prot = 5000  # int(input("Number of example to estimate lime ?(between 2 and 10000)"))
        base_string = "m"  # input("Which base string ? (X,c,p), default=X : ")
        if nb_seq == "all":
            frac = 1
        else:
            frac = nb_seq / 992
        calc_LIME_interpretability(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            frac,
            nb_ex_per_prot,
            base_string,
        )
    elif choose_method == "integrated_grad":
        # Quick and can be done on GPU
        nb_ex_per_prot = 50  # int(input("Number of example to estimate lime ?(between 2 and 10000)"))
        base_string = "m"  # input("Which base string ? (X,c,p), default=X : ")

        score_on_residue_with_integreted_gradients(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            nb_ex_per_prot,
            base_string,
        )

    elif choose_method == "LRP_with_rollout_cls_and_sum_col":
        # Long and can't be done on GPU
        # https://github.com/hila-chefer/Transformer-Explainability
        score_on_residue_with_LRP_rollout(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            "cls_and_somme_col",
        )
    elif choose_method == "rollout":
        # Long and can't be done on GPU
        score_on_residue_with_LRP_rollout(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            "rollout",
        )
    elif choose_method == "gradCam":
        # Long and can't be done on GPU
        score_on_residue_with_LRP_rollout(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            "gradCam",
        )
    elif choose_method == "attn_last_layer":
        # Long and can't be done on GPU
        score_on_residue_with_LRP_rollout(
            path_choosen
            + "/"
            + training_analyser.config.tasks[0].unique_task_name
            + ".pth",
            training_analyser.config.starting_model.path_vocab,
            dataset_name,
            nb_seq,
            "attn_last_layer",
        )

    else:
        logging.debug("methode inconue")