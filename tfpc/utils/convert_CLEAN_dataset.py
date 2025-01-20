import pandas as pd

df_train = pd.read_csv("data/datasets/CLEAN_dataset/split100.csv",sep="\t")
print(df_train)
df_train.to_json("data/datasets/CLEAN_dataset/CLEAN_dataset_train.json")

df_test_valid = pd.read_csv("data/datasets/CLEAN_dataset/price.csv",sep="\t")
print(df_test_valid)
df_test_valid.to_json("data/datasets/CLEAN_dataset/CLEAN_dataset_test.json")
df_test_valid.to_json("data/datasets/CLEAN_dataset/CLEAN_dataset_valid.json")