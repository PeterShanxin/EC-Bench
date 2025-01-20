from pathlib import Path

import pandas as pd

from utils.metrics_ECPred import MacroAvg
from collections import Counter


def parse_special_case(ec_number):
    if "No" in ec_number:
        return "-1.-1.-1.-1"  # We create a class for no alignement found and then no prediction of BLAST
    if "EC:" in ec_number:
        return ec_number.split(":")[1]

    return ec_number


def get_ec_to_keep_minimum_seen(SP_train, min_ec_occur):
    dataset_folder = Path("data/datasets/")
    path_df_train = dataset_folder / Path(
        "SwissProt_20" + SP_train + "/SwissProt_20" + SP_train + "_train.json"
    )
    df = pd.read_json(path_df_train)
    dico_count = Counter(df["ec_number"])
    ec_to_keep = [
        ec_number for ec_number, count in dico_count.items() if count > min_ec_occur
    ]
    return ec_to_keep


def get_seq_pred_DEEPre(SP_recent):
    pred_folder = Path("data/prediction_for_review/")
    path_DEEPre_pred = pred_folder / Path(
        "pred_"
        + "DEEPre"
        + "_apriori_enzyme_False"
        + "_Diff_SP_"
        + "16_08"
        + "_"
        + SP_recent
        + "_without_leak_at_0.4.csv"
    )
    df_DEEPre = pd.read_csv(path_DEEPre_pred)
    return list(df_DEEPre["sequence"])


def eval_on_lvl(
    SP_train,
    SP_recent,
    model_name,
    filtered_min_len,
    filtered_max_len,
):
    print(
        "Evaluation of model "
        + model_name
        + " with SwissProt release "
        + SP_train
        + " as training data"
    )
    pred_folder = Path("data/prediction_for_review/")
    dataset_folder = Path("data/datasets/")
    path_df_train = dataset_folder / Path(
        "SwissProt_20" + SP_train + "/SwissProt_20" + SP_train + "_train.json"
    )
    path_model_pred = pred_folder / Path(
        "pred_"
        + model_name
        + "_apriori_enzyme_False"
        + "_Diff_SP_"
        + SP_train
        + "_"
        + SP_recent
        + "_without_leak_at_0.4.csv"
    )
    path_test_set = dataset_folder / Path(
        "Diff_SP_"
        + SP_train
        + "_"
        + SP_recent
        + "/Diff_SP_"
        + SP_train
        + "_"
        + SP_recent
        + "_without_leak_at_0.4.json"
    )

    df_pred = pd.read_csv(path_model_pred)
    if " pred_ec" in df_pred.columns:
        df_pred = df_pred.rename(columns={" pred_ec": "pred_ec"})

    dico_pred = {row["sequence"]: row["pred_ec"] for _, row in df_pred.iterrows()}
    dico_pred = {
        sequence: parse_special_case(pred_ec) for sequence, pred_ec in dico_pred.items()
    }
    df_test_set = pd.read_json(path_test_set)

    # Keep only sequences that have ec present in the train because none of the tested tools can do zero shot predictions
    df_train = pd.read_json(path_df_train)
    list_train_class_vocab = list(df_train["ec_number"].unique())
    df_test_set = df_test_set[df_test_set["ec_number"].isin(list_train_class_vocab)]

    # For Diff_SP_16_08_23_02_without_leak_at_0.4.csv keep only sequences predicted by DEEPre because other sequences were not predicted -> Favorable to DEEPre
    if SP_train == "16_08":
        seq_pred_DEEPre = get_seq_pred_DEEPre(SP_recent)
        df_test_set = df_test_set[df_test_set["sequence"].isin(seq_pred_DEEPre)]

    # Filtered out ec that had been viewed very few times
    # ec_to_keep = get_ec_to_keep_minimum_seen(SP_train, min_ec_occur=0)  # 10 50
    # df_test_set = df_test_set[df_test_set["ec_number"].isin(ec_to_keep)]

    # Filter on sequence size depending on the tools we compare to
    df_test_set["len_seq"] = df_test_set["sequence"].apply(lambda x: len(x))
    df_test_set = df_test_set[df_test_set["len_seq"] < filtered_max_len]
    df_test_set = df_test_set[df_test_set["len_seq"] > filtered_min_len]

    metric = MacroAvg()  # Mecro F1 to be precise

    for _, row in df_test_set.iterrows():
        true_label = row["ec_number"]
        sequence = row["sequence"]
        predicted_label = dico_pred[sequence]
        metric.step(true_label, predicted_label)

    macro_f1 = {}
    # print("-" * 90)
    # print("Model name:", model_name)
    # print("Level 0 :")
    metric_value_lvl0 = metric.get_metric_at_lvl(lvl=0)
    macro_f1["lvl_0"] = metric_value_lvl0["macro avg"]["f1-score"]
    # print("macro f1-score:", metric_value_lvl0["macro avg"]["f1-score"])

    for lvl in range(1, 5):
        # print("Level", lvl, ":")
        metric_value = metric.get_metric_at_lvl(lvl)
        macro_f1["lvl_" + str(lvl)] = metric_value["macro avg"]["f1-score"]
        # print("macro f1-score:", metric_value["macro avg"]["f1-score"])
    # print("-" * 90)

    return macro_f1


def generate_latex_table(dico_all_res, SP_train):
    latex_table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{cccccc}\n\\toprule\n\\textbf{Model Name} & \\textbf{Level 0}& \\textbf{Level 1} & \\textbf{Level 2} & \\textbf{Level 3} & \\textbf{Level 4} \\\\ \n \\midrule \n"
    for model_name, macro_f1 in dico_all_res.items():
        model_name = model_name.replace("_", "\\_")
        if "EnzBert" in model_name:
            model_name = "\\textbf{" + model_name + "}"
        latex_table += (
            model_name
            + " & "
            + str(round(macro_f1["lvl_0"], 4))
            + " & "
            + str(round(macro_f1["lvl_1"], 4))
            + " & "
            + str(round(macro_f1["lvl_2"], 4))
            + " & "
            + str(round(macro_f1["lvl_3"], 4))
            + " & "
            + str(round(macro_f1["lvl_4"], 4))
            + " \\\\ \n"
        )

    latex_table += (
        "\\bottomrule \n   \\end{tabular} \n \\caption{Macro-f1 evaluated on the dataset named Diff\_SP\_"
        + SP_train.replace("_", "\_")
        + "\_23\_02} \n \\end{table}"
    )
    print(latex_table)


def compute_metric():
    dico_all_res = {}
    #### On 2016 data
    SP_train = "16_08"
    list_model_to_test = [
        "BLASTp",
        "EnzBert_SwissProt_2016_08",
        "DEEPre",
    ]
    filtered_min_len = 50
    filtered_max_len = 5000
    for model_name in list_model_to_test:
        macro_f1 = eval_on_lvl(
            SP_train=SP_train,
            SP_recent="23_02",
            model_name=model_name,
            filtered_min_len=filtered_min_len,
            filtered_max_len=filtered_max_len,
        )
        dico_all_res[model_name] = macro_f1
    generate_latex_table(dico_all_res, SP_train)

    dico_all_res = {}
    ### On 2018 data
    SP_train = "18_01"
    list_model_to_test = ["DeepEC", "BLASTp", "EnzBert_SwissProt_2018_01"]
    filtered_min_len = 10
    filtered_max_len = 1000
    for model_name in list_model_to_test:
        macro_f1 = eval_on_lvl(
            SP_train=SP_train,
            SP_recent="21_04",
            model_name=model_name,
            filtered_min_len=filtered_min_len,
            filtered_max_len=filtered_max_len,
        )
        dico_all_res[model_name] = macro_f1

    generate_latex_table(dico_all_res, SP_train)
