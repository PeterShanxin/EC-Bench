import pandas as pd
from tqdm import tqdm


def verify_unicity_id_sequence(df_test):
    """
    No sequences have two ids and "inversly"
    """
    assert len(df_test["uniprot_id"]) == len(df_test["uniprot_id"].unique())
    assert len(df_test["sequence"]) == len(df_test["sequence"].unique())


def convert_DeepEC_res_to_dict(df_pred, df_test):
    dico_mapping = create_dico_df_test(df_test)
    if "Predicted class" in df_pred.columns:
        lvl0 = True
    else:
        lvl0 = False
    dico_complete_pred = {}
    dico_max_ddn_activity = {}
    for _, row in tqdm(df_pred.iterrows(), total=len(df_pred)):
        dnn_activity = row["DNN activity"]
        uniprot_id = row["Query ID"]
        associated_seq = dico_mapping[uniprot_id]
        if lvl0:
            current_pred = row["Predicted class"]
        else:
            current_pred = row["Predicted EC number"]

        if associated_seq not in dico_complete_pred.keys():
            dico_complete_pred[associated_seq] = current_pred
            dico_max_ddn_activity[associated_seq] = dnn_activity
        else:
            previous_dnn_activity = dico_max_ddn_activity[associated_seq]
            if dnn_activity > previous_dnn_activity:
                dico_complete_pred[associated_seq] = current_pred
                dico_max_ddn_activity[associated_seq] = dnn_activity
    return dico_complete_pred, dico_max_ddn_activity


def create_dico_df_test(df_test):
    dico_mappging = {}
    for _, row in df_test.iterrows():
        uniprot_id = row["uniprot_id"]
        sequence = row["sequence"]
        dico_mappging[uniprot_id] = sequence
    return dico_mappging


def post_precessing_DeepEC(
    path_test_set, prediction_folder, path_BLASTp_pred, SP_test, apriori_enzyme
):
    df_test_set = pd.read_json(path_test_set)
    verify_unicity_id_sequence(df_test_set)
    df_pred_lvl0 = pd.read_csv(
        prediction_folder + "/log_files/Enzyme_prediction.txt", sep="\t"
    )
    dico_pred_lvl0, _ = convert_DeepEC_res_to_dict(df_pred_lvl0, df_test_set)

    df_pred_lvl3 = pd.read_csv(
        prediction_folder + "/log_files/3digit_EC_prediction.txt", sep="\t"
    )
    dico_pred_lvl3, dico_max_proba_lvl3 = convert_DeepEC_res_to_dict(
        df_pred_lvl3, df_test_set
    )

    df_pred_lvl4 = pd.read_csv(
        prediction_folder + "/log_files/4digit_EC_prediction.txt", sep="\t"
    )
    dico_pred_lvl4, dico_max_proba_lvl4 = convert_DeepEC_res_to_dict(
        df_pred_lvl4, df_test_set
    )

    df_pred_blastp = pd.read_csv(path_BLASTp_pred)
    print("df_pred_blastp:", df_pred_blastp.columns)
    dico_pred_BLASTp = {
        row["sequence"]: row[" pred_ec"] for _, row in df_pred_blastp.iterrows()
    }

    threshold_value_of_their_code = 0.5

    # Replicate DeepEC pipeline
    final_sequence_order = []
    final_pred = []
    for _, row in tqdm(df_test_set.iterrows(), total=len(df_test_set)):
        sequence = row["sequence"]
        if dico_pred_lvl0[sequence] == "Non-enzyme" and not apriori_enzyme:
            prediction = "0.0.0.0"
        else:
            if dico_max_proba_lvl3[sequence] < threshold_value_of_their_code:
                prediction = dico_pred_BLASTp[sequence]
            else:
                pred_lvl3_from_lvl4 = ".".join(dico_pred_lvl4[sequence].split(".")[:3])
                lvl3_and_4_coherent = dico_pred_lvl3[sequence] == pred_lvl3_from_lvl4
                if dico_max_proba_lvl4[sequence] < threshold_value_of_their_code:
                    prediction = dico_pred_BLASTp[sequence]
                elif lvl3_and_4_coherent:
                    prediction = dico_pred_lvl4[sequence]
                else:
                    prediction = dico_pred_BLASTp[sequence]
        final_sequence_order.append(sequence)
        final_pred.append(prediction)

    df_final_pred_DeepEC = pd.DataFrame.from_dict(
        {"sequence": final_sequence_order, "pred_ec": final_pred}
    )
    df_final_pred_DeepEC.to_csv(
        "data/prediction_for_review/pred_DeepEC_apriori_enzyme_"
        + str(apriori_enzyme)
        + "_Diff_SP_18_01_"
        + SP_test
        + "_without_leak_at_0.4.csv"
    )
