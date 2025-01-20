import pandas as pd


def describe_data(path_json, path_df_train):
    # Keep only sequences that have ec present in the train because none of the tested tools can do zero shot predictions
    df_train = pd.read_json(path_df_train)
    list_train_class_vocab = list(df_train["ec_number"].unique())

    print("path_json:", path_json)
    df = pd.read_json(path_json)
    df = df[df["ec_number"].isin(list_train_class_vocab)]
    print("Total:", len(df))
    dico_first_EC = {str(k): 0 for k in range(8)}

    for index, row in df.iterrows():
        ec_number = row["ec_number"]
        first_ec = ec_number.split(".")[0]
        dico_first_EC[first_ec] += 1

    print(dico_first_EC)


def verify_inclusion():
    # Keep only sequences that have ec present in the train because none of the tested tools can do zero shot predictions
    path_df_train = "data/datasets/SwissProt_2018_01/SwissProt_2018_01_train.json"
    df_train = pd.read_json(path_df_train)
    list_train_class_vocab = list(df_train["ec_number"].unique())

    Diff_SP_18_01_21_04 = pd.read_json(
        "data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json"
    )
    Diff_SP_18_01_21_04 = Diff_SP_18_01_21_04[
        Diff_SP_18_01_21_04["ec_number"].isin(list_train_class_vocab)
    ]
    Diff_SP_18_01_21_04 = Diff_SP_18_01_21_04[Diff_SP_18_01_21_04["len_seq"] < 1000]

    Diff_SP_18_01_23_02 = pd.read_json(
        "data/datasets/Diff_SP_18_01_23_02/Diff_SP_18_01_23_02_without_leak_at_0.4.json"
    )
    Diff_SP_18_01_23_02 = Diff_SP_18_01_23_02[
        Diff_SP_18_01_23_02["ec_number"].isin(list_train_class_vocab)
    ]
    Diff_SP_18_01_23_02 = Diff_SP_18_01_23_02[Diff_SP_18_01_23_02["len_seq"] < 1000]

    seq_1 = set(Diff_SP_18_01_21_04["sequence"])
    seq_2 = set(Diff_SP_18_01_23_02["sequence"])
    if seq_1 in seq_2:
        print("C'est bien une sous partie")
    else:
        in_2021_but_not_2023 = seq_1 - seq_2
        df_chelou = Diff_SP_18_01_21_04[
            Diff_SP_18_01_21_04["sequence"].isin(in_2021_but_not_2023)
        ]
        print(df_chelou)
        print(len(in_2021_but_not_2023))
        print("ERREUR, problème quelques part")


verify_inclusion()
describe_data(
    path_json="data/datasets/Diff_SP_16_08_23_02/Diff_SP_16_08_23_02_without_leak_at_0.4.json",
    path_df_train="data/datasets/SwissProt_2016_08/SwissProt_2016_08_train.json",
)
describe_data(
    path_json="data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json",
    path_df_train="data/datasets/SwissProt_2018_01/SwissProt_2018_01_train.json",
)
