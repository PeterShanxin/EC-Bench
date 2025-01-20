import numpy as np
import torch
from scipy.special import softmax
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import combinations


def calc_KS(strat_obj, dataframe, which_highest, network_name, temperature=1):
    nb_parties = 1000
    arr_proba = np.zeros((nb_parties))
    arr_correct = np.zeros((nb_parties))
    arr_counter = np.zeros((nb_parties))
    for ind, row in dataframe.iterrows():
        sequence = row["primary"]
        label = row["label"]
        pred_all_proba = strat_obj.get_proba(sequence, network_name, temperature)
        indice_choosen = np.flip(np.argsort(pred_all_proba))[which_highest - 1]
        proba_choosen_indice = pred_all_proba[indice_choosen]
        correct = label == indice_choosen
        quotient = int(proba_choosen_indice / nb_parties)
        arr_proba[quotient] += proba_choosen_indice
        arr_correct[quotient] += correct
        arr_counter[quotient] += 1

    somme_proba_pred = 0
    somme_correct = 0
    somme_counter = 0
    max_error = 0
    for k in range(nb_parties):
        somme_proba_pred += arr_proba[k]
        somme_correct += arr_correct[k]
        somme_counter += arr_counter[k]
        sigma_t = somme_proba_pred / somme_counter
        taux_t = somme_correct / somme_counter
        error_t = np.abs(sigma_t - taux_t)
        if error_t > max_error:
            max_error = error_t

    return max_error


def calc_ECE(strat_obj, dataframe, which_highest, network_name, temperature=1):
    nb_categorie = 10
    taille_categorie = 1 / nb_categorie
    all_categorie = [[] for _ in range(nb_categorie)]
    for _, row in dataframe.iterrows():
        sequence = row["primary"]
        label = row["label"]
        pred_all_proba = strat_obj.get_proba(sequence, network_name, temperature)
        indice_choosen = np.flip(np.argsort(pred_all_proba))[which_highest - 1]
        proba_choosen_indice = pred_all_proba[indice_choosen]
        correct = label == indice_choosen
        quotient = int(proba_choosen_indice / taille_categorie)
        all_categorie[quotient].append([proba_choosen_indice, correct])

    new_tab = []
    for cat in all_categorie:
        new_tab.append(np.mean([c[1] for c in cat]))

    # ECE : Expected Calibration Error
    score_ECE = 0
    nb_exemples_total = len(dataframe)
    for cat in all_categorie:
        len_BM = len(cat)
        acc_BM = np.mean([c[0] for c in cat])
        conf_BM = np.mean([c[1] for c in cat])
        if len_BM > 0:
            score_ECE += (len_BM / nb_exemples_total) * np.abs(acc_BM - conf_BM)

    return score_ECE


def calc_acc(strat_obj, dataframe, network_name):
    correct = 0
    total = 0
    for _, row in dataframe.iterrows():
        sequence = row["primary"]
        label = row["label"]
        prediction = np.argmax(
            strat_obj.dico_all_results[sequence][
                strat_obj.dico_name_to_ind_network[network_name]
            ]
        )
        if prediction == label:
            correct += 1
        total += 1
    return correct / total


def calc_acc_ens(strat_obj, dataframe, tirage_poids, combi_list_name):
    correct = 0
    total = 0
    for _, row in dataframe.iterrows():
        sequence = row["primary"]
        label = row["label"]
        prediction = None
        for ind, network_name in enumerate(combi_list_name):
            poids_this_net_work = tirage_poids[ind]
            weight = strat_obj.dico_all_results[sequence][
                strat_obj.dico_name_to_ind_network[network_name]
            ]

            if strat_obj.wich_output == "proba":
                proba = softmax(weight)
                output = proba
            elif strat_obj.wich_output == "weight":
                output = weight
            else:
                raise RuntimeError("wich output not supported")
            output = np.array(output)
            weighted_output_current_network = poids_this_net_work * output
            if prediction is None:
                prediction = weighted_output_current_network
            else:
                prediction += weighted_output_current_network
        prediction = np.argmax(prediction)
        if prediction == label:
            correct += 1
        total += 1
    return correct / total


class base_strategy:
    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        self.all_networks_names = ensembles
        self.ensembles_names = name_ens
        self.nb_models = len(ensembles)
        self.df_train = df_train
        self.df_test = df_test
        self.num_cross_val = num_cross_val
        self.all_KS = None
        self.all_ECE = None
        self.dico_temp_softmax = {ens: 1 for ens in ensembles}
        self.dico_all_results = dico_all_results
        self.dico_name_to_ind_network = dict()
        self.dico_ind_to_name_network = dict()
        for ind, name in enumerate(ensembles):
            self.dico_name_to_ind_network[name] = ind
            self.dico_ind_to_name_network[ind] = name

    def train(self):
        pass

    def get_all_KS(self, which_highest):
        if self.all_KS is None:
            self.all_KS = []
            for network_name in self.all_networks_names:
                self.all_KS.append(
                    calc_KS(
                        self,
                        self.df_test,
                        which_highest,
                        network_name,
                        self.dico_temp_softmax[network_name],
                    )
                )
        return self.all_KS

    def get_all_ECE(self, which_highest):
        if self.all_ECE is None:
            self.all_ECE = []
            for network_name in self.all_networks_names:
                self.all_ECE.append(
                    calc_ECE(
                        self,
                        self.df_test,
                        which_highest,
                        network_name,
                        self.dico_temp_softmax[network_name],
                    )
                )
        return self.all_ECE

    def get_proba(self, sequence, network_name, temperature):
        return softmax(
            np.array(
                self.dico_all_results[sequence][
                    self.dico_name_to_ind_network[network_name]
                ]
            )
            / temperature
        )

    def get_score(self, which_net_to_use):
        total_correct = 0
        for _, row in self.df_test.iterrows():
            sequence = row["primary"]
            label = row["label"]
            pred_clas = self.get_decision(sequence, which_net_to_use)
            correct = pred_clas == label
            total_correct += correct
        return total_correct / len(self.df_test)

    def get_specific_pred(self, sequence, which_net_to_use):
        all_outputs = []
        for name_network in which_net_to_use:
            out = self.dico_all_results[sequence][
                self.dico_name_to_ind_network[name_network]
            ]
            all_outputs.append(out)
        return np.array(all_outputs)


class strategy_proba(base_strategy):
    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        self.name_strat = "proba"
        super().__init__(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )

    def get_decision(self, sequence, which_net_to_use):
        all_outputs = self.get_specific_pred(sequence, which_net_to_use)
        new_outputs = []
        for out in all_outputs:
            new_out = softmax(out)
            new_outputs.append(new_out)
        res = np.mean(new_outputs, axis=0)
        pred_class = np.argmax(res)
        return pred_class


class strategy_weight(base_strategy):
    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        self.name_strat = "weight"
        super().__init__(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )

    def get_decision(self, sequence, which_net_to_use):
        all_outputs = self.get_specific_pred(sequence, which_net_to_use)
        res = np.mean(all_outputs, axis=0)
        pred_class = np.argmax(res)
        return pred_class


class strategy_proba_post_scale_temp_tunning(base_strategy):
    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        super().__init__(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
        self.which_metric = list_param_strat[0]
        if self.which_metric == "KS":
            self.name_strat = "proba_post_calib_with_temp_scaling_KS"
        else:
            self.name_strat = "proba_post_calib_with_temp_scaling_ECE"
        logging.info("Je commence l'entrainement pour %s", self.name_strat)
        self.train()
        logging.info("J'ai finit l'entrainement pour %s", self.name_strat)

    def train(self):
        for network_name in self.all_networks_names:
            logging.info(
                "Training for %s to find the optimal temperature for the softmax",
                network_name,
            )
            all_temperature = np.linspace(0.8, 3, 100)
            temp_opti = None
            error_min = 1
            for ind_temp, temperature in enumerate(all_temperature):
                if ind_temp % 25 == 0:
                    logging.info(
                        "ind_temp : %s %%", ind_temp / len(all_temperature) * 100
                    )
                if self.which_metric == "KS":
                    error = calc_KS(self, self.df_test, 1, network_name, temperature)
                elif self.which_metric == "ECE":
                    error = calc_ECE(self, self.df_test, 1, network_name, temperature)
                else:
                    raise RuntimeError(
                        "Metric not defined, please choose between KS and ECE"
                    )
                if error < error_min:
                    error_min = error
                    temp_opti = temperature
            self.dico_temp_softmax[network_name] = temp_opti

    def get_decision(self, sequence, which_net_to_use):
        all_outputs = self.get_specific_pred(sequence, which_net_to_use)
        new_outputs = []
        for ind, out in enumerate(all_outputs):
            name_network = self.dico_ind_to_name_network[ind]
            temp_opti = self.dico_temp_softmax[name_network]
            new_out = softmax(out / temp_opti)
            new_outputs.append(new_out)
        res = np.mean(new_outputs, axis=0)
        pred_class = np.argmax(res)
        return pred_class


class strategy_BMA(base_strategy):
    """
    Bayesian model averaging strategy with the model weight or probability
    """

    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        super().__init__(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
        self.wich_output = list_param_strat[0]
        if self.wich_output == "proba":
            self.name_strat = "BMA_proba"
        else:
            self.name_strat = "BMA_weight"

        logging.info("Je commence l'entrainement pour %s", self.name_strat)
        self.train()
        logging.info("J'ai finit l'entrainement pour %s", self.name_strat)

    def train(self):
        # Train Bayesian model averaging
        logging.info("Starting the training of BMA")

        self.weights = dict()
        for network_name in self.all_networks_names:
            logging.info(
                "Training for %s to find the optimal temperature for the softmax",
                network_name,
            )
            acc = calc_acc(self, self.df_train, network_name)
            epsilon = 1 - acc

            p_h_sachant_d = 1
            for _, row in self.df_train.iterrows():
                sequence = row["primary"]
                label = row["label"]
                prediction = np.argmax(
                    self.dico_all_results[sequence][
                        self.dico_name_to_ind_network[network_name]
                    ]
                )
                if prediction == label:
                    factor = 1 - epsilon
                else:
                    factor = epsilon
                p_h_sachant_d *= factor
            self.weights[network_name] = p_h_sachant_d
        logging.info("The final weights are : %s", self.weights)

    def get_decision(self, sequence, which_net_to_use):
        all_outputs = self.get_specific_pred(sequence, which_net_to_use)
        new_outputs = []
        for ind, weight in enumerate(all_outputs):
            if self.wich_output == "proba":
                proba = softmax(weight)
                output = proba
            elif self.wich_output == "weight":
                output = weight
            else:
                raise RuntimeError("wich output not supported")
            network_name = which_net_to_use[ind]
            output = self.weights[network_name] * output
            new_outputs.append(output)
        res = np.mean(new_outputs, axis=0)
        pred_class = np.argmax(res)
        return pred_class


class strategy_BMC(base_strategy):
    """
    Bayesian model combination strategy with the model weight or probability
    """

    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        super().__init__(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
        self.wich_output = list_param_strat[0]
        if self.wich_output == "proba":
            self.name_strat = "BMC_proba"
        else:
            self.name_strat = "BMC_weight"

        logging.info("Je commence l'entrainement pour %s", self.name_strat)
        self.total_tirage = 100  # 1000
        self.q = 3  # 25
        self.weights = dict()
        self.train()
        logging.info("J'ai finit l'entrainement pour %s", self.name_strat)

    def train(self):
        # Train Bayesian model combination
        self.dico_name_to_ind_network_this_combi = dict()
        for k in range(1, self.nb_models + 1):
            print("Nb model :", k, "/", self.nb_models)
            indices = list(range(self.nb_models))
            all_combi = list(combinations(indices, k))

            for indice_which_net_to_use in all_combi:
                all_combi_name = []
                for ind_mod in indice_which_net_to_use:
                    all_combi_name.append(self.all_networks_names[int(ind_mod)])
                self.dico_name_to_ind_network_this_combi[str(all_combi_name)] = dict()
                self.dico_name_to_ind_network_this_combi[str(all_combi_name)] = dict()
                for ind, name in enumerate(all_combi_name):
                    self.dico_name_to_ind_network_this_combi[str(all_combi_name)][
                        name
                    ] = ind
                    self.dico_name_to_ind_network_this_combi[str(all_combi_name)][
                        ind
                    ] = name
                self.train_this_combi(all_combi_name)

    def train_this_combi(self, combi_list_name):
        list_alpha = np.ones(len(combi_list_name))

        N_tirage = int(self.total_tirage / self.q)
        best_tirage = list_alpha
        for _ in tqdm(range(N_tirage)):
            all_tirage = np.random.dirichlet(best_tirage, self.q)
            best_p = None
            for tirage_poids in all_tirage:
                p_e_sachant_d = self.eval_combi(tirage_poids, combi_list_name)
                if best_p is None or p_e_sachant_d > best_p:
                    best_p = p_e_sachant_d
                    best_tirage = best_tirage + tirage_poids
        """
        TO TEST
        """
        ind_list = []
        for network_name in combi_list_name:
            network_ind = self.dico_name_to_ind_network_this_combi[
                str(combi_list_name)
            ][network_name]
            ind_list.append(network_ind)
        """
        FIN TO TEST
        """

        self.weights[str(combi_list_name)] = dict()
        for ind, network_name in zip(
            ind_list, combi_list_name
        ):  # TO TEST : self.all_networks_names):
            self.weights[str(combi_list_name)][network_name] = best_tirage[ind]

    def eval_combi(self, tirage_poids, combi_list_name):
        acc = calc_acc_ens(self, self.df_train, tirage_poids, combi_list_name)
        epsilon = 1 - acc
        p_e_sachant_d = 1
        for _, row in self.df_train.iterrows():
            sequence = row["primary"]
            label = row["label"]
            prediction = None
            for ind, network_name in enumerate(combi_list_name):
                poids_this_net_work = tirage_poids[ind]
                weight = self.dico_all_results[sequence][
                    self.dico_name_to_ind_network[network_name]
                ]

                if self.wich_output == "proba":
                    proba = softmax(weight)
                    output = proba
                elif self.wich_output == "weight":
                    output = weight
                else:
                    raise RuntimeError("wich output not supported")
                output = np.array(output)
                weighted_output_current_network = poids_this_net_work * output
                if prediction is None:
                    prediction = weighted_output_current_network
                else:
                    prediction += weighted_output_current_network
            prediction = np.argmax(prediction)
            if prediction == label:
                factor = 1 - epsilon
            else:
                factor = epsilon
            p_e_sachant_d *= factor

        return p_e_sachant_d

    def get_decision(self, sequence, which_net_to_use):
        all_outputs = self.get_specific_pred(sequence, which_net_to_use)
        new_outputs = []
        for ind, weight in enumerate(all_outputs):
            if self.wich_output == "proba":
                proba = softmax(weight)
                output = proba
            elif self.wich_output == "weight":
                output = weight
            else:
                raise RuntimeError("wich output not supported")
            network_name = which_net_to_use[ind]
            output = self.weights[str(which_net_to_use)][network_name] * output
            new_outputs.append(output)
        res = np.mean(new_outputs, axis=0)
        pred_class = np.argmax(res)
        return pred_class


class strategy_Learn_weights(base_strategy):
    """
    Learn weights for each model strategy with the model weight or probability
    """

    def __init__(
        self,
        list_param_strat,
        name_ens,
        ensembles,
        df_train,
        df_test,
        num_cross_val,
        dico_all_results,
    ):
        super().__init__(
            list_param_strat,
            name_ens,
            ensembles,
            df_train,
            df_test,
            num_cross_val,
            dico_all_results,
        )
        self.wich_output = list_param_strat[0]
        self.epochs = list_param_strat[1]
        if self.wich_output == "proba":
            self.name_strat = "gradient_descent_opti_on_proba_" + str(self.epochs)
        else:
            self.name_strat = "gradient_descent_opti_on_weight_" + str(self.epochs)
        logging.info("Je commence l'entrainement pour %s", self.name_strat)
        self.learned_weight = dict()
        self.train()
        logging.info("J'ai finit l'entrainement pour %s", self.name_strat)

    def train(self):
        self.dico_name_to_ind_network_this_combi = dict()
        for k in tqdm(range(1, self.nb_models + 1)):
            indices = list(range(self.nb_models))
            all_combi = list(combinations(indices, k))
            for indice_which_net_to_use in all_combi:
                all_combi_name = []
                for ind_mod in indice_which_net_to_use:
                    all_combi_name.append(self.all_networks_names[int(ind_mod)])
                self.dico_name_to_ind_network_this_combi[str(all_combi_name)] = dict()
                self.dico_name_to_ind_network_this_combi[str(all_combi_name)] = dict()
                for ind, name in enumerate(all_combi_name):
                    self.dico_name_to_ind_network_this_combi[str(all_combi_name)][
                        name
                    ] = ind
                    self.dico_name_to_ind_network_this_combi[str(all_combi_name)][
                        ind
                    ] = name
                self.train_this_combi(all_combi_name)

    def train_this_combi(self, all_combi_name):
        self.learned_weight[str(all_combi_name)] = Learning_weights(len(all_combi_name))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.learned_weight[str(all_combi_name)].parameters(), lr=1e-4
        )
        for _ in range(self.epochs):
            list_loss = []
            for _, row in self.df_train.iterrows():
                sequence = row["primary"]
                label = torch.tensor([row["label"]])
                ouputs_all_models = self.get_specific_pred(sequence, all_combi_name)
                if self.wich_output == "proba":
                    proba = softmax(ouputs_all_models)
                    output = proba
                elif self.wich_output == "weight":
                    output = ouputs_all_models
                else:
                    raise RuntimeError("wich output not supported")
                ind_list = []
                for network_name in all_combi_name:
                    network_ind = self.dico_name_to_ind_network_this_combi[
                        str(all_combi_name)
                    ][network_name]
                    ind_list.append(network_ind)

                output = self.learned_weight[str(all_combi_name)](
                    ouputs_all_models, ind_list
                )
                output = output.unsqueeze(0)
                loss = criterion(output, label)
                list_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        """
        logging.info(
            "The mean loss on the last epoch is %s for %s epochs",
            np.mean(list_loss),
            self.epochs,
        )
        logging.info(
            "After gradient descent the final weights are : %s",
            self.learned_weight[str(all_combi_name)].model_weighting,
        )
        """

    def get_decision(self, sequence, which_net_to_use):
        all_outputs = self.get_specific_pred(sequence, which_net_to_use)
        if self.wich_output == "proba":
            proba = softmax(all_outputs)
            output = proba
        elif self.wich_output == "weight":
            output = all_outputs
        else:
            raise RuntimeError("wich output not supported")

        ind_list = []
        for network_name in which_net_to_use:
            network_ind = self.dico_name_to_ind_network_this_combi[
                str(which_net_to_use)
            ][network_name]
            ind_list.append(network_ind)
        output = self.learned_weight[str(which_net_to_use)](output, ind_list)
        output = output.detach().numpy()
        pred_class = np.argmax(output)
        return pred_class


class Learning_weights(torch.nn.Module):
    def __init__(self, nb_models):
        super(Learning_weights, self).__init__()
        tensor = torch.ones(nb_models)
        self.model_weighting = torch.nn.parameter.Parameter(tensor)

    def forward(self, out_all_models, ind_choose_model):
        out_all_models = torch.tensor(out_all_models)
        output = None
        for ind, out_one_model in zip(ind_choose_model, out_all_models):
            weigthing_out = out_one_model * torch.clamp(
                input=self.model_weighting[ind],
                min=1e-4,
                max=1e15,
            )
            if output is None:
                output = weigthing_out
            else:
                output += weigthing_out
        return output