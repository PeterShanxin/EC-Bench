import pandas as pd
import numpy as np
from tqdm import tqdm


def print_to_latex(matrice_similarity, all_name_df, col_type):
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Dataset Name} &"
    for dataset_name in all_name_df:
        latex_table += "\\textbf{" + dataset_name.replace("_", "\\_") + "} &"
    latex_table += " \\textbf{total} \\\\ \n \\midrule \n"
    for i in range(3):
        tmp_latex = all_name_df[i].replace("_", "\\_")
        for j in range(4):
            tmp_latex += "& \\num{" + str(int(matrice_similarity[i][j])) + "}"
        tmp_latex += "\\\\ \n"
        latex_table += tmp_latex

    latex_table += (
        "\\bottomrule \n   \\end{tabular} \n \\caption{Number of "
        + col_type
        + " in common between the different train set} \n \\end{table}"
    )
    print(latex_table)


df_EC40_train = pd.read_json("data/datasets/EC40/EC40_train.json")
df_ECPred40_train = pd.read_json("data/datasets/ECPred40/ECPred40_train.json")
df_SwissProt_train = pd.read_json(
    "data/datasets/OLD/SwissProt_2021_04/SwissProt_2021_04_train.json"
)

dico_colname_seq = {
    "Train EC40": "primary",
    "Train ECPred40": "sequence",
    "Train SwissProt_2021_04:": "sequence",
}
dico_colname_ec = {
    "Train EC40": "label",
    "Train ECPred40": "EC Number",
    "Train SwissProt_2021_04:": "ec_number",
}
dico_df = {
    "Train EC40": df_EC40_train,
    "Train ECPred40": df_ECPred40_train,
    "Train SwissProt_2021_04:": df_SwissProt_train,
}
all_name_df = list(dico_df.keys())

matrice_common_seq = np.zeros((3, 4))
matric_percentage_of_min = np.zeros((3, 3))
matrice_common_ec = np.zeros((3, 4))
at_level = 2
for ind1, name_d1 in tqdm(enumerate(all_name_df)):
    df1 = dico_df[name_d1]
    df1[dico_colname_ec[name_d1]] = df1[dico_colname_ec[name_d1]].apply(
        lambda x: x.replace("-", ".")
    )
    seq_d1 = set(df1[dico_colname_seq[name_d1]])
    df1 = df1[df1[dico_colname_ec[name_d1]] != "0.0.0.0"]
    df1 = df1[df1[dico_colname_ec[name_d1]] != "0.0"]
    df1[dico_colname_ec[name_d1]] = df1[dico_colname_ec[name_d1]].apply(
        lambda x: ".".join(x.split(".")[:at_level])
    )
    ec_d1 = set(df1[dico_colname_ec[name_d1]])
    # print(ec_d1)
    for ind2, name_d2 in enumerate(all_name_df):
        df2 = dico_df[name_d2]
        df2[dico_colname_ec[name_d2]] = df2[dico_colname_ec[name_d2]].apply(
            lambda x: x.replace("-", ".")
        )
        df2[dico_colname_ec[name_d2]] = df2[dico_colname_ec[name_d2]].apply(
            lambda x: ".".join(x.split(".")[:at_level])
        )
        seq_d2 = set(df2[dico_colname_seq[name_d2]])
        common_seq = seq_d1.intersection(seq_d2)
        matrice_common_seq[ind1][ind2] = len(common_seq)
        size_smaller_datasets = min(len(seq_d1), len(seq_d2))
        matric_percentage_of_min[ind1][ind2] = len(common_seq) / size_smaller_datasets
        df2 = df2[df2[dico_colname_ec[name_d2]] != "0.0.0.0"]
        df2 = df2[df2[dico_colname_ec[name_d2]] != "0.0"]
        ec_d2 = set(df2[dico_colname_ec[name_d2]])
        common_ec = ec_d1.intersection(ec_d2)
        matrice_common_ec[ind1][ind2] = len(common_ec)
    matrice_common_seq[ind1][3] = len(seq_d1)
    matrice_common_ec[ind1][3] = len(ec_d1)

print("matric_percentage_of_min:")
print(matric_percentage_of_min)
print(matrice_common_seq)
print_to_latex(matrice_common_seq, all_name_df, col_type="sequence")
print_to_latex(
    matrice_common_ec, all_name_df, col_type="ec number at level " + str(at_level)
)
