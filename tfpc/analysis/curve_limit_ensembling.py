import os
import numpy as np
import itertools
from analysis.test_ensembling_model import TestEnsemblingModel


class LimitEnsembling:
    def __init__(self):
        dico_res = dict()
        for k in range(1, 6):
            moyenne, var = self.get_mean_var_for_one_nb_model(k)
            dico_res[k] = [moyenne, var]
        print(dico_res)

    def get_mean_var_for_one_nb_model(self, nb_model):
        num_models = list(range(1, 6))
        combi = itertools.combinations(num_models, nb_model)
        list_acc = []
        for list_models in combi:
            acc = self.get_acc_from_these_model(list_models)
            list_acc.append(acc)
        return np.mean(list_acc), np.std(list_acc)

    def get_acc_from_these_model(self, list_model):
        FOLDER_XP = "data/models/fine_tune_models/"
        list_directory = [x[0] for x in os.walk(FOLDER_XP)]
        list_directory = [x.split("/")[-1] for x in list_directory]
        base_name = "fine_tune_for_EC_prediction_with_lr_scheduler_r"
        name_ensemble = "curve_limit"
        list_of_xp_path = []
        for k in list_model:
            if k == 1:
                path_xp = FOLDER_XP + base_name[:-2]
            else:
                path_xp = FOLDER_XP + base_name + str(k)
            list_of_xp_path.append(path_xp)

        test_ensemble = TestEnsemblingModel(list_of_xp_path, name_ensemble)
        ind_model = 0
        ind_task = 0
        dico_res = test_ensemble.eval_model(ind_model, ind_task)
        print(dico_res)
        return dico_res["classif_EC_pred_lvl_2"]["accuracy"]
