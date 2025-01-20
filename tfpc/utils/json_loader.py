import ijson
import pandas as pd


def load_json_into_pandas_dataframe(filename):
    with open(filename, "r") as f:
        objects = ijson.items(f, "")
        dico_content = next(objects)
    df = pd.DataFrame.from_dict(dico_content)
    return df