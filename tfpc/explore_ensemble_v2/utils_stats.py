import torch
from strategy_obj import (
    strategy_proba,
    strategy_weight,
    strategy_proba_post_scale_temp_tunning,
    strategy_BMA,
    strategy_BMC,
    strategy_Learn_weights,
)
import torch
import pandas as pd
from itertools import combinations
import logging

# Delete the padding in order to compare the prot
def delete_pad(sequence):
    sequence = sequence.replace("p", "")
    sequence = sequence.replace("c", "")
    return sequence


def list_of_result_to_dico(all_result):
    dico_pred = dict()
    for indice, res_mod_i in enumerate(all_result):
        all_seq = [delete_pad(r[0]) for r in res_mod_i]
        all_class_weights = [r[1][:65] for r in res_mod_i]
        for seq, weights in zip(all_seq, all_class_weights):
            if seq not in dico_pred.keys():
                dico_pred[seq] = [weights]
            else:
                dico_pred[seq].append(weights)
    return dico_pred


def get_df_for_cross_val(df, nb_partie_cross_val):
    # On convertie les label vers leurs valeurs dans le vocabulaire du/des models
    dict_label = torch.load(
        "../data/models/fine_tune_models/ProtBert_EC40_classic_r1/classif_EC_pred_lvl_2_vocab.pth"
    )
    # On convertie les labels du dataset
    for ind, row in df.iterrows():
        label = row["label"]
        df.at[ind, "label"] = dict_label[label]

    # On tronque le dataset à 1024 AA en longueur max
    for ind, row in df.iterrows():
        sequence = row["primary"]
        if len(sequence) > 1024:
            df.at[ind, "primary"] = sequence[:1024]

    # On mélange dataframe
    dataframe = df.sample(frac=1)
    # On sépare en nb_partie_cross_val parties
    taille_one_part = int(len(dataframe) / nb_partie_cross_val)
    list_df = []
    for num_partie in range(nb_partie_cross_val):
        indice_test = set(
            list(
                range(num_partie * taille_one_part, (num_partie + 1) * taille_one_part)
            )
        )

        indice_train = list(set(list(range(len(dataframe)))) - indice_test)
        indice_test = list(indice_test)
        df_train = dataframe.iloc[indice_train]
        df_test = dataframe.iloc[indice_test]
        list_df.append([df_train, df_test])
    return list_df


def get_strategy_obj(
    strat_name,
    list_param_strat,
    name_ens,
    ensembles,
    df_train,
    df_test,
    num_cross_val,
    dico_all_results,
):
    if strat_name == "proba":
        return strategy_proba(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
    elif strat_name == "weight":
        return strategy_weight(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
    elif (
        strat_name == "proba_post_calibrate_temp_tunning_ECE"
        or strat_name == "proba_post_calibrate_temp_tunning_KS"
    ):
        return strategy_proba_post_scale_temp_tunning(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
    elif strat_name == "BMA_weight" or strat_name == "BMA_proba":
        return strategy_BMA(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
    elif strat_name == "BMC_weight" or strat_name == "BMC_proba":
        return strategy_BMC(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
    elif "gradient_descent_opti" in strat_name:
        return strategy_Learn_weights(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
    else:
        raise RuntimeError("Strategy name uknown")


def output_summary_of_ensemble_strategy(
    dict_all_ensembles, all_name_strategy, df, nb_partie_cross_val, save_name
):
    list_df_cross_val = get_df_for_cross_val(df, nb_partie_cross_val)
    all_strategy_all_ensembles = []
    for name_ens, ensembles in dict_all_ensembles.items():
        logging.info("Je créer les objets pour l'ensemble %s", name_ens)
        num_cross_val = 0
        default_path_result = "../data/tests_data/"
        all_result = []
        for name_res_i in ensembles:
            res_i = torch.load(default_path_result + name_res_i)
            all_result.append(res_i)
        dico_all_results = list_of_result_to_dico(all_result)
        # We take the opposite to have the biggest dataset in test to have an accurate metric value
        for df_test, df_train in list_df_cross_val:
            logging.info(
                "Je créer et j'entraine les objets pour le %ser/ème split des données",
                num_cross_val + 1,
            )
            all_strategy_one_ensemble = []
            for key, list_param_strat in all_name_strategy.items():
                all_strategy_one_ensemble.append(
                    get_strategy_obj(
                        key,
                        list_param_strat,
                        name_ens,
                        ensembles,
                        df_train,
                        df_test,
                        num_cross_val,
                        dico_all_results,
                    )
                )
            all_strategy_all_ensembles.append(all_strategy_one_ensemble)
            num_cross_val += 1

    all_results = []
    for ind_group_ens, all_strategy_one_ensemble in enumerate(
        all_strategy_all_ensembles
    ):
        logging.info(
            "Total process : %s %%",
            ind_group_ens / len(all_strategy_all_ensembles) * 100,
        )
        for ind_strategy, strategy in enumerate(all_strategy_one_ensemble):
            logging.info(
                "Progression this ensemble : %s %%",
                ind_strategy / len(all_strategy_one_ensemble) * 100,
            )
            for k in range(1, strategy.nb_models + 1):
                indices = list(range(strategy.nb_models))
                all_combi = list(combinations(indices, k))

                for indice_which_net_to_use in all_combi:
                    all_combi_name = []
                    for ind_mod in indice_which_net_to_use:
                        all_combi_name.append(strategy.all_networks_names[int(ind_mod)])
                    which_net_to_use = all_combi_name

                    score = strategy.get_score(which_net_to_use)
                    one_result = {
                        "name_ensembles": strategy.ensembles_names,
                        "nb_model": k,
                        "participant_network": which_net_to_use,
                        "all_temperature_softmax": str(strategy.dico_temp_softmax),
                        "numero_df_cross_val": strategy.num_cross_val,
                        "all_KS_first_highest": strategy.get_all_KS(which_highest=1),
                        "all_ECE_first_highest": strategy.get_all_ECE(which_highest=1),
                        "all_KS_second_highest": strategy.get_all_KS(which_highest=2),
                        "all_ECE_second_highest": strategy.get_all_ECE(which_highest=2),
                        "strategy_name": strategy.name_strat,
                        "accuracy": score,
                    }
                    all_results.append(one_result)

    data = {
        "name_ensembles": [r["name_ensembles"] for r in all_results],
        "nb_model": [r["nb_model"] for r in all_results],
        "participant_network": [r["participant_network"] for r in all_results],
        "all_temperature_softmax": [r["all_temperature_softmax"] for r in all_results],
        "numero_df_cross_val": [r["numero_df_cross_val"] for r in all_results],
        "all_KS_first_highest": [r["all_KS_first_highest"] for r in all_results],
        "all_ECE_first_highest": [r["all_ECE_first_highest"] for r in all_results],
        "all_KS_second_highest": [r["all_KS_second_highest"] for r in all_results],
        "all_ECE_second_highest": [r["all_ECE_second_highest"] for r in all_results],
        "strategy_name": [r["strategy_name"] for r in all_results],
        "accuracy": [r["accuracy"] for r in all_results],
    }
    df_results = pd.DataFrame.from_dict(data)
    print(df_results)

    df_results.to_json(save_name)
