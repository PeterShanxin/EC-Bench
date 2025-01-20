import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from collections import Counter
from scipy import stats

fichier = open("data/residues_of_interest/dico_res_correctness.pkl", "rb")
dico_score_correctness = pkl.load(fichier)
nb_methods = len(dico_score_correctness)

print("Il y a", nb_methods, "methodes différentes")

best_method = "Raw_attention_sum_col_unit_length_norm1"
# We can take the keys of one of the dictionary entry because they are all the same
all_seqs = list(
    dico_score_correctness["Raw_attention_sum_col_unit_length_norm1"].keys()
)

all_methods = list(dico_score_correctness.keys())

dico_flatten_result = dict()
for method in all_methods:
    dico_flatten_result[method] = []

for sequence in all_seqs:
    for method in all_methods:
        correctness_prot_i = list(dico_score_correctness[method][sequence])
        dico_flatten_result[method] += correctness_prot_i


# Only combine with best method
increase_union = np.zeros((len(all_methods)))
correlation_with_best_method = np.zeros((len(all_methods)))
nb_correct = np.zeros((len(all_methods)))
res1 = np.array(dico_flatten_result[best_method])
nb_correct_method_1 = res1.sum()
for j, method2 in enumerate(all_methods):
    print(j, "/", nb_methods)
    res2 = np.array(dico_flatten_result[method2])
    nb_correct_method_2 = res2.sum()
    nb_correct_best_method = np.max([nb_correct_method_1, nb_correct_method_2])
    union_of_the_two_method = np.count_nonzero(res1 + res2)
    diff = union_of_the_two_method - nb_correct_best_method
    increase_union[j] = diff
    nb_correct[j] = res2.sum()
    corr = stats.spearmanr(res1, res2).correlation
    correlation_with_best_method[j] = corr


fig, ax = plt.subplots(1, 1)
ax.bar(list(range(len(all_methods))), increase_union)
ax.set_xticks(np.arange(len(all_methods)))
ax.set_xticklabels(all_methods, rotation="vertical")
fig.tight_layout()
plt.title(
    "How many exemple are find with a complementary method that are not find with Raw_attention_sum_col_unit_length_norm1"
)
plt.show()


# Only correlation of interests
fig, ax = plt.subplots(1, 1)
order = np.argsort(correlation_with_best_method)
sort_method_name = [all_methods[o] for o in order]
sort_method_value = correlation_with_best_method[order]
ax.bar(list(range(len(correlation_with_best_method))), sort_method_value)
ax.set_xticks(np.arange(len(all_methods)))
ax.set_xticklabels(sort_method_name, rotation="vertical")
fig.tight_layout()
plt.title("Rank correlation coefficient with the best methods")
plt.show()

# Print scores
nb_correct = np.array(nb_correct)
order = np.argsort(nb_correct)
sort_method_name = [all_methods[o] for o in order]
sort_method_value = nb_correct[order]
fig, ax = plt.subplots(1, 1)
ax.bar(list(range(len(all_methods))), sort_method_value)
ax.set_xticks(np.arange(len(all_methods)))
ax.set_xticklabels(sort_method_name, rotation="vertical")
plt.title("Scores of each methods")
plt.show()