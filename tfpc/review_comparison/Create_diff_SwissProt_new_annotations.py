from pathlib import Path
import pandas as pd
import os


def diff_SwissProt(dataset_name1, dataset_name2, dataset_folder=Path("data/datasets/")):
    """
    Return sequence with its annotations that is present in dataset2 but not in dataset1
    """
    tab1 = dataset_name1.split("_")
    tab2 = dataset_name2.split("_")
    output_dataset_name = (
        "Diff_SP_" + tab1[1][-2:] + "_" + tab1[2] + "_" + tab2[1][-2:] + "_" + tab2[2]
    )
    output_folder = dataset_folder / Path(output_dataset_name)
    output_json_path = output_folder / Path(output_dataset_name + ".json")
    path_d1 = dataset_folder / Path(dataset_name1) / Path(dataset_name1 + "_train.json")
    print("Try to open", path_d1)
    df1 = pd.read_json(path_d1)
    path_d2 = dataset_folder / Path(dataset_name2) / Path(dataset_name2 + "_train.json")
    print("Try to open", path_d2)
    df2 = pd.read_json(path_d2)

    seq_only_in_df2 = set(df2["sequence"]) - set(df1["sequence"])
    diff_df = df2[df2["sequence"].isin(seq_only_in_df2)]
    print(diff_df)
    print(len(diff_df))
    os.mkdir(output_folder)
    diff_df.to_json(output_json_path)
