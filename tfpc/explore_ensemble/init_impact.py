import torch
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils_ensemble import list_of_result_to_dico

sys.path.append("../")
sys.path.append("../models_architectures/")
prefix = "../data/models/fine_tune_models/result_previous_may_2021/"
# Which model I want to put on the ensemble and on which dataset(train/test/valid)
LIST_PATH_MODEL = [
    "EC_prediction_impact_init_and_example_order_v1",
    "EC_prediction_impact_init_and_example_order_v2",
    "EC_prediction_impact_init_and_example_order_v3",
    "EC_prediction_impact_init_and_example_order_v4",
]

nb_model = len(LIST_PATH_MODEL)
all_weight_all_models = []
all_models = []
for path_res in LIST_PATH_MODEL:
    model_i = torch.load(
        prefix + path_res + "/transformer_embedder.pth",
        map_location=torch.device("cpu"),
    )
    all_weight = None
    for param in model_i.parameters():
        if all_weight is None:
            all_weight = param.reshape(-1)
        else:
            all_weight = torch.cat((all_weight, param.reshape(-1)))
    all_weight_all_models.append(all_weight.detach().numpy())


X = np.array(all_weight_all_models)
"""
X_embedded = TSNE(n_components=2).fit_transform(X)
plt.scatter([x[0] for x in X_embedded], [x[1] for x in X_embedded])
plt.show()
"""

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

print(np.exp(np.array(cosine_distances(X, X))))
# distance between rows of X
print(euclidean_distances(X, X))


################ Par rapport au prédiction sur la tâche d'EC prédiction

prefix = "../data/tests_data/"
# Which model I want to put on the ensemble and on which dataset(train/test/valid)
LIST_PATH_RESULT = [
    "EC_init_variation/EC_prediction_impact_init_and_example_order_v1_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    "EC_init_variation/EC_prediction_impact_init_and_example_order_v2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    "EC_init_variation/EC_prediction_impact_init_and_example_order_v3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    "EC_init_variation/EC_prediction_impact_init_and_example_order_v4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
]

nb_model = len(LIST_PATH_RESULT)
all_result = []
for path_res in LIST_PATH_RESULT:
    res_i = torch.load(prefix + path_res)
    all_result.append(res_i)

dico_pred = list_of_result_to_dico(all_result)


nb_model = len(all_result)
matrice_desaccord = np.zeros((nb_model, nb_model))
for i in range(nb_model):
    for j in range(nb_model):
        same_pred = 0
        for sequence in dico_pred.keys():
            pred_i = np.argmax(dico_pred[sequence][i])
            pred_j = np.argmax(dico_pred[sequence][j])
            if pred_i == pred_j:
                same_pred += 1
        matrice_desaccord[i][j] = 1 - (same_pred / len(dico_pred))
print(matrice_desaccord)
plt.matshow(matrice_desaccord, cmap=plt.cm.Blues)
plt.title("Matrice de desacord entre les prédictions des différents models")
plt.colorbar()
plt.show()