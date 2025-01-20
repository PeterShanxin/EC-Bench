import torch
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
from collections import Counter

sys.path.append("models_architectures/")
from wrapper_prot_bert_BFD import WrapperProtBertBFD


def get_ec_weight(path_df_train, class_vocab):
    df = pd.read_json(path_df_train)
    dico_count = Counter(df["ec_number"])
    total_count = len(df)
    conter_prior = np.zeros(len(class_vocab))
    for ec_number, ind in class_vocab.items():
        conter_prior[ind] = 1 / np.log(dico_count[ec_number])
    return torch.from_numpy(conter_prior)


@torch.no_grad()
def get_EnzBert_predictions(
    model_path,
    vocab_path,
    path_class_vocab,
    path_df_test,
    apriori_enzyme,
    SP_version,  # = 18_01 or 16_08
    SP_test,
    max_seq_len,
):
    dataset_folder = Path("data/datasets/")
    path_df_train = dataset_folder / Path(
        "SwissProt_20" + SP_version + "/SwissProt_20" + SP_version + "_train.json"
    )

    folder_pred = "data/prediction_for_review/"
    dataframe = pd.read_json(path_df_test)
    list_sequences = dataframe["sequence"]

    # Load the model
    model_name = model_path.split("/")[-2]
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    vocab = torch.load(vocab_path)

    class_vocab = torch.load(path_class_vocab)

    ind_non_enzyme = class_vocab["0.0.0.0"]
    inv_class_vocab = {ind: ec_number for ec_number, ind in class_vocab.items()}

    fichier_result_proba_dist = open(
        folder_pred
        + "pred_"
        + model_name
        + "_apriori_enzyme_"
        + str(apriori_enzyme)
        + "_Diff_SP_"
        + SP_version
        + "_"
        + SP_test
        + "_without_leak_at_0.4.csv",
        "w",
    )
    fichier_result_proba_dist.write("sequence, pred_ec\n")
    # We calc the metric on all the dev set
    for sequence in tqdm(list_sequences):
        sequence = sequence[:max_seq_len]

        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        proba_pred = model(input_batch)[0]
        if apriori_enzyme:
            proba_pred[ind_non_enzyme] = -10000.0
        ind_best = torch.argmax(proba_pred).item()
        ec_prediction = inv_class_vocab[ind_best]
        fichier_result_proba_dist.write(sequence + "," + ec_prediction + "\n")

    fichier_result_proba_dist.close()
