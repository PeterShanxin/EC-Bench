from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle as pkl
import math

# Delete the padding in order to compare the prot
def delete_pad(sequence):
    sequence = sequence.replace("p", "")
    sequence = sequence.replace("c", "")
    return sequence


def how_many_predict_prot(prot, seq_pred_and_correct_by_all_models, list_of_list_pred):
    how_many_pred = 0
    for ind_model in range(len(list_of_list_pred)):
        if prot in seq_pred_and_correct_by_all_models[ind_model]:
            how_many_pred += 1
    return how_many_pred


def list_of_result_to_dico(all_result):
    dico_pred = dict()
    for indice, res_mod_i in enumerate(all_result):
        all_seq = [delete_pad(r[0]) for r in res_mod_i]
        all_class_weights = [r[1] for r in res_mod_i]
        for seq, weights in zip(all_seq, all_class_weights):
            if seq not in dico_pred.keys():
                dico_pred[seq] = [weights]
            else:
                dico_pred[seq].append(weights)
    return dico_pred


def softmax_with_temp(x, tau):
    """Returns softmax probabilities with temperature tau
    Input:  x -- 1-dimensional array
    Output: s -- 1-dimensional array
    """
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


def acc_of_ensemble(df_test, dico_pred, method, list_all_norm, all_diffs=None, temp=1):
    correct = 0
    all_correct = []
    all_inccorect = []
    for ind, row in df_test.iterrows():
        sequence = row["primary"]
        label = row["label"]
        all_preds = dico_pred[sequence]
        if method == "proba":
            all_preds = [softmax(class_weight) for class_weight in all_preds]
            output_transf = np.sum(all_preds, axis=0)
        elif method == "proba_median":
            all_preds = [softmax(class_weight) for class_weight in all_preds]
            output_transf = np.median(all_preds, axis=0)
        elif method == "weight":
            output_transf = np.sum(all_preds, axis=0)
        elif method == "weight_normalize":
            all_preds = [preds / np.linalg.norm(preds) for preds in all_preds]
            output_transf = np.sum(all_preds, axis=0)
        elif method == "weight_all_normalize":
            all_preds = [
                preds / list_all_norm[indice] for indice, preds in enumerate(all_preds)
            ]
            output_transf = np.sum(all_preds, axis=0)
        elif method == "max":
            output_transf = np.max(all_preds, axis=0)
        elif method == "proba_with_temp":
            all_preds = [
                softmax_with_temp(np.array(class_weight), temp)
                for class_weight in all_preds
            ]
            output_transf = np.sum(all_preds, axis=0)

        pred_class = np.argmax(output_transf)
        if pred_class == label:
            correct += 1
            all_correct.append([sequence, all_preds])
        else:
            all_inccorect.append([sequence, all_preds])

    # TEMPORAIRE annalyse TO DELETE
    if len(list(dico_pred.values())[0]) == 5:
        pkl.dump(all_correct, open("all_correct_" + method + ".pkl", "wb"))
        pkl.dump(all_inccorect, open("all_incorrect_" + method + ".pkl", "wb"))
    # Fin TEMPORAIRE analyse

    acc = (correct / len(df_test)) * 100
    return acc


def calc_post_calib(df_train, dico_pred):
    dico_pred = {your_key: dico_pred[your_key] for your_key in df_train["primary"]}
    nb_network = len(list(dico_pred.values())[0])
    step_pourcentile = 1
    all_describe_function = []
    for num_network in range(nb_network):
        describe_function = []
        dico_weight = {key: value[num_network] for key, value in dico_pred.items()}
        dico_proba = {key: softmax(value) for key, value in dico_weight.items()}
        list_max_proba = [np.max(value) for key, value in dico_proba.items()]
        for k in range(1, 100, step_pourcentile):
            pourcentile_max = np.percentile(list_max_proba, k)
            pourcentil_min = np.percentile(list_max_proba, k - 1)
            list_proba_predite = []
            correct = 0
            total = 0
            for ind, row in df_train.iterrows():
                sequence = row["primary"]
                label = row["label"]
                proba_max = np.max(dico_proba[sequence])
                if proba_max < pourcentile_max and proba_max > pourcentil_min:
                    list_proba_predite.append(proba_max)
                    pred_class = np.argmax(dico_proba[sequence])
                    if pred_class == label:
                        correct += 1
                    total += 1
            acc = correct / total
            diff = np.mean(list_proba_predite) - acc
            describe_function.append([pourcentil_min, diff])
        all_describe_function.append(describe_function)
    return all_describe_function


def calc_temp_calib(df_train, dico_pred, top_k_ECE):
    dico_pred = {your_key: dico_pred[your_key] for your_key in df_train["primary"]}
    nb_network = len(list(dico_pred.values())[0])
    step_pourcentile = 1
    all_best_temp = []
    for num_network in range(nb_network):
        dico_weight = {key: value[num_network] for key, value in dico_pred.items()}
        all_temperature_to_test = np.linspace(0.8, 1.8, num=40)
        mini_error = 100
        best_temp = None
        for temperature_softmax in all_temperature_to_test:
            erreur_ECE = calc_ECE(
                10, temperature_softmax, dico_weight, df_train, top_k_ECE
            )
            if erreur_ECE < mini_error:
                mini_error = erreur_ECE
                best_temp = temperature_softmax
        all_best_temp.append(best_temp)
    print(all_best_temp)
    return all_best_temp


def calc_mlp_calib(df_train, dico_pred):
    dico_pred = {your_key: dico_pred[your_key] for your_key in df_train["primary"]}
    nb_network = len(list(dico_pred.values())[0])
    all_mlps = []
    for num_network in range(nb_network):
        dico_weight = {key: value[num_network] for key, value in dico_pred.items()}
        X_train = list(dico_weight.values())
        y_train = list(df_train["label"])
        for k in range(65):
            one_item = np.zeros((65))
            one_item[k] = 6.0
            one_y_train = k
            X_train.append(one_item)
            y_train.append(one_y_train)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = MLPClassifier(
            hidden_layer_sizes=(65), max_iter=50, warm_start=True, alpha=0.001
        ).fit(X_train, y_train)
        clf.coefs_ = [
            np.identity(65) + 0.1 * np.random.random((65, 65)),
            np.identity(65) + 0.1 * np.random.random((65, 65)),
        ]
        clf.intercepts_ = [
            0.1 * np.random.random((65)),
            0.1 * np.random.random((65)),
        ]
        # clf.n_layers_ = 3
        # clf.out_activation_ = "softmax"
        # clf.n_iter_ = 30
        clf = clf.fit(X_train, y_train)
        all_mlps.append(clf)
    return all_mlps


def calc_ECE(nb_categorie, temperature_softmax, dico_weight, df_train, top_k):
    taille_categorie = 1 / nb_categorie
    all_categorie = [[] for _ in range(nb_categorie)]
    nb_exemples_total = len(df_train)
    for ind, row in df_train.iterrows():
        sequence = row["primary"]
        label = row["label"]

        nb_class = len(dico_weight[sequence])
        indices = np.flip(np.argsort(dico_weight[sequence]))[:top_k]
        for num_class in indices:
            preuve = softmax(np.array(dico_weight[sequence]) / temperature_softmax)[
                num_class
            ]
            correct = label == num_class
            quotient = int(preuve / taille_categorie)
            if quotient < 0:
                quotient = 0
            all_categorie[quotient].append([correct, preuve])

        # prediction = np.argmax(dico_weight[sequence])
        # proba = np.max(softmax(np.array(dico_weight[sequence]) / temperature_softmax))
        # correct = label == prediction
        # quotient = int(proba / taille_categorie)
        # all_categorie[quotient].append([correct, proba])

    # ECE : Expected Calibration Error
    score_ECE = 0

    for cat in all_categorie:
        len_BM = len(cat)
        acc_BM = np.mean([c[0] for c in cat])
        conf_BM = np.mean([c[1] for c in cat])
        if len_BM > 0:
            score_ECE += (len_BM / nb_exemples_total) * np.abs(acc_BM - conf_BM)

    score_ECE = score_ECE * 100
    return score_ECE


"""
#OLD recalibrate non optize
def recalibrate(all_preds, all_describe_function):
    nb_network = len(all_preds)
    new_list = []
    for num_network in range(nb_network):
        proba_this_network = all_preds[num_network]
        describe_function = all_describe_function[num_network]
        new_proba = []
        for proba in proba_this_network:
            c = 0
            while c < len(describe_function) and proba > describe_function[c][0]:
                c += 1
            # TODO : vérifier c-1 si c'est correct et pas c ou autre
            diff = describe_function[c - 1][1]
            new_proba.append(proba - diff)
        new_list.append(np.array(new_proba))
    return new_list
"""


def recalibrate(all_preds, all_describe_function):
    # Attention fixed threashold is to avoid rescaling where we don't have enought data but it's not pretty
    fixed_threashold_proba = 0.1
    nb_network = len(all_preds)
    new_list = []
    for num_network in range(nb_network):
        proba_this_network = all_preds[num_network]
        describe_function = all_describe_function[num_network]
        seuil_pre = 0
        for elem in describe_function:
            seuil = elem[0]
            diff = elem[1]
            mask1 = proba_this_network < seuil
            mask2 = proba_this_network > seuil_pre
            mask3 = proba_this_network > fixed_threashold_proba
            mask_final = mask1 & mask2 & mask3
            proba_this_network[mask_final] -= diff
            seuil_pre = seuil
        new_list.append(np.array(proba_this_network))
    return new_list


def relu(x):
    return max(0.0, x)


def acc_of_ensemble_with_cross_val(dataframe, dico_pred, method, list_all_norm, temp=1):
    # On mélange dataframe
    dataframe = dataframe.sample(frac=1)
    # On sépare en nb_partie_cross_val parties
    nb_partie_cross_val = 5
    taille_one_part = int(len(dataframe) / nb_partie_cross_val)
    list_accuracy = []
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

        if method == "proba_re_calibrate":
            all_describe_function = calc_post_calib(df_train, dico_pred)
        elif method == "proba_re_calibrate_temperature":
            all_temp_opti = calc_temp_calib(df_train, dico_pred, top_k_ECE=5)
        elif method == "proba_re_calibrate_mlp":
            all_mlps = calc_mlp_calib(df_train, dico_pred)
        correct = 0
        for ind, row in df_test.iterrows():
            sequence = row["primary"]
            label = row["label"]
            all_preds = dico_pred[sequence]
            if method == "proba":
                all_preds = [softmax(class_weight) for class_weight in all_preds]
                output_transf = np.sum(all_preds, axis=0)
            elif method == "proba_median":
                all_preds = [softmax(class_weight) for class_weight in all_preds]
                output_transf = np.median(all_preds, axis=0)
            elif method == "weight_median":
                output_transf = np.median(all_preds, axis=0)
            elif method == "proba_with_relu":
                all_preds = [
                    softmax([relu(x) for x in class_weight])
                    for class_weight in all_preds
                ]
                output_transf = np.sum(all_preds, axis=0)
            elif method == "weight":
                output_transf = np.sum(all_preds, axis=0)
            elif method == "weight_with_relu":
                new_all_preds = []
                for pred in all_preds:
                    new_all_preds.append([relu(x) for x in pred])
                all_preds = new_all_preds
                output_transf = np.sum(all_preds, axis=0)
            elif method == "weight_normalize":
                all_preds = [preds / np.linalg.norm(preds) for preds in all_preds]
                output_transf = np.sum(all_preds, axis=0)
            elif method == "weight_all_normalize":
                all_preds = [
                    preds / list_all_norm[indice]
                    for indice, preds in enumerate(all_preds)
                ]
                output_transf = np.sum(all_preds, axis=0)
            elif method == "max":
                output_transf = np.max(all_preds, axis=0)
            elif method == "proba_re_calibrate":
                all_preds = [softmax(class_weight) for class_weight in all_preds]
                all_preds = recalibrate(all_preds, all_describe_function)
                output_transf = np.sum(all_preds, axis=0)
            elif method == "proba_re_calibrate_temperature":
                new_all_preds = []
                for k in range(len(all_temp_opti)):
                    new_all_preds.append(softmax(all_preds[k] / all_temp_opti[k]))
                output_transf = np.sum(new_all_preds, axis=0)
            elif method == "proba_re_calibrate_mlp":
                new_all_preds = []
                for k in range(len(all_mlps)):
                    pred_mlp = all_mlps[k].predict_proba([all_preds[k]])[0]
                    new_all_preds.append(pred_mlp)
                output_transf = np.sum(new_all_preds, axis=0)
            elif method == "proba_with_temp":
                all_preds = [
                    softmax_with_temp(np.array(class_weight), temp)
                    for class_weight in all_preds
                ]
                output_transf = np.sum(all_preds, axis=0)

            pred_class = np.argmax(output_transf)
            if pred_class == label:
                correct += 1
        acc = (correct / len(df_test)) * 100
        list_accuracy.append(acc)
    return np.mean(list_accuracy), np.std(list_accuracy)
