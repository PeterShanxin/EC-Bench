import pandas as pd
from review_comparison.generate_pred_EnzBert import get_EnzBert_predictions
from review_comparison.post_process_deepec import post_precessing_DeepEC
from review_comparison.Sequence_KNN import SequenceKNN
from review_comparison.request_to_DEEPre import Wrapper_DEEPre_server


def generate_all_predictions():
    ### On Diff_SP_16_08_23_02 :
    # Predictions with DEEPre
    # df_test_DEEPre = pd.read_json(
    #     "data/datasets/Diff_SP_16_08_23_02/Diff_SP_16_08_23_02_without_leak_at_0.4.json"
    # )
    # DEEPre = Wrapper_DEEPre_server(
    #     df_test_DEEPre,
    #     path_output_file="data/prediction_for_review/pred_DEEPre_apriori_enzyme_False_Diff_SP_16_08_23_02_without_leak_at_0.4.csv",
    # )
    # DEEPre.make_pred_df()
    # DEEPre.make_pred_df_multiprocess()

    # Predictions with Blastp
    blast_p = SequenceKNN(
        path_train_json="data/datasets/mine/cluster-30/cluster-30_train.json",
        path_test_json="data/datasets/min/cluster-30/cluster-30_test.json",
        nb_thread=4,
        path_output_pred="data/predictions/pred_BLASTp_mine.csv",
        tool="BLASTp",
        a_priori_enzyme=False,
    )
    blast_p.launch_pipeline()

    # Predictions with DIAMOND
    # DIAMOND = SequenceKNN(
    #     path_train_json="data/datasets/SwissProt_2016_08/SwissProt_2016_08_train.json",
    #     path_test_json="data/datasets/Diff_SP_16_08_23_02/Diff_SP_16_08_23_02_without_leak_at_0.4.json",
    #     nb_thread=4,
    #     path_output_pred="data/prediction_for_review/pred_DIAMOND_apriori_enzyme_False_Diff_SP_16_08_23_02_without_leak_at_0.4.csv",
    #     tool="DIAMOND",
    #     a_priori_enzyme=False,
    # )
    # DIAMOND.launch_pipeline()

    ## Generate prediction of EnzBert
    # Prediction with EnzBert_SwissProt_2016_08 on Diff_SP_16_08_23_02_without_leak_at_0.4.json with enzyme apriori
    # get_EnzBert_predictions(
    #     model_path="data/models/fine_tune_models/EnzBert_SwissProt_2016_08/classif_EC_pred_lvl_4.pth",
    #     vocab_path="data/models/pre_trained_models/30_layer_uniparc+BFD/vocab.pkl",
    #     path_class_vocab="data/models/fine_tune_models/EnzBert_SwissProt_2016_08/classif_EC_pred_lvl_4_vocab.pth",
    #     path_df_test="data/datasets/Diff_SP_16_08_23_02/Diff_SP_16_08_23_02_without_leak_at_0.4.json",
    #     apriori_enzyme=False,
    #     SP_version="16_08",
    #     max_seq_len=5000,
    # )

    ### On Diff_SP_18_01_23_02 :
    # Predictions with Blastp
    # blast_p = SequenceKNN(
    #     path_train_json="data/datasets/SwissProt_2018_01/SwissProt_2018_01_train.json",
    #     path_test_json="data/datasets/Diff_SP_18_01_23_02/Diff_SP_18_01_23_02_without_leak_at_0.4.json",
    #     nb_thread=4,
    #     path_output_pred="data/prediction_for_review/pred_BLASTp_apriori_enzyme_False_Diff_SP_18_01_23_02_without_leak_at_0.4.csv",
    #     tool="BLASTp",
    #     a_priori_enzyme=False,
    # )
    # blast_p.launch_pipeline()

    # Predictions with DIAMOND
    # DIAMOND = SequenceKNN(
    #     path_train_json="data/datasets/SwissProt_2018_01/SwissProt_2018_01_train.json",
    #     path_test_json="data/datasets/Diff_SP_18_01_23_02/Diff_SP_18_01_23_02_without_leak_at_0.4.json",
    #     nb_thread=4,
    #     path_output_pred="data/prediction_for_review/pred_DIAMOND_apriori_enzyme_False_Diff_SP_18_01_23_02_without_leak_at_0.4.csv",
    #     tool="DIAMOND",
    #     a_priori_enzyme=False,
    # )
    # DIAMOND.launch_pipeline()

    # Post processing over DeepEC because cannot have complete prediction with their tools
    # Tool launch with the command: python3 deepec.py -i 'data/datasets/Diff_SP_18_01_23_02/Diff_SP_18_01_23_02_without_leak_at_0.4.fasta'  -o 'data/prediction_for_review/pred_DeepEC_Diff_SP_18_01_23_02_without_leak_at_0.4'
    # post_precessing_DeepEC(
    #     path_test_set="data/datasets/Diff_SP_18_01_23_02/Diff_SP_18_01_23_02_without_leak_at_0.4.json",
    #     prediction_folder="data/prediction_for_review/pred_DeepEC_Diff_SP_18_01_23_02_without_leak_at_0.4",
    #     path_BLASTp_pred="data/prediction_for_review/pred_DIAMOND_apriori_enzyme_False_Diff_SP_18_01_23_02_without_leak_at_0.4.csv",
    #     apriori_enzyme=False,
    # )

    # Prediction with EnzBert_SwissProt_2018_01 on Diff_SP_18_01_23_02_without_leak_at_0.4.json without enzyme apriori
    # get_EnzBert_predictions(
    #     model_path="data/models/fine_tune_models/EnzBert_SwissProt_2018_01/classif_EC_pred_lvl_4.pth",
    #     vocab_path="data/models/pre_trained_models/30_layer_uniparc+BFD/vocab.pkl",
    #     path_class_vocab="data/models/fine_tune_models/EnzBert_SwissProt_2018_01/classif_EC_pred_lvl_4_vocab.pth",
    #     path_df_test="data/datasets/Diff_SP_18_01_23_02/Diff_SP_18_01_23_02_without_leak_at_0.4.json",
    #     apriori_enzyme=False,
    #     SP_version="18_01",
    #     max_seq_len=1024,
    # )

    #### New Diff_SP_18_01_21_04

    # Predictions with Blastp
    # blast_p = SequenceKNN(
    #     path_train_json="data/datasets/SwissProt_2018_01/SwissProt_2018_01_train.json",
    #     path_test_json="data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json",
    #     nb_thread=4,
    #     path_output_pred="data/prediction_for_review/pred_BLASTp_apriori_enzyme_False_Diff_SP_18_01_21_04_without_leak_at_0.4.csv",
    #     tool="BLASTp",
    #     a_priori_enzyme=False,
    # )
    # blast_p.launch_pipeline()

    # Prediction with EnzBert_SwissProt_2018_01 on Diff_SP_18_01_21_04_without_leak_at_0.4.json without enzyme apriori
    # get_EnzBert_predictions(
    #     model_path="data/models/fine_tune_models/EnzBert_SwissProt_2018_01/classif_EC_pred_lvl_4.pth",
    #     vocab_path="data/models/pre_trained_models/30_layer_uniparc+BFD/vocab.pkl",
    #     path_class_vocab="data/models/fine_tune_models/EnzBert_SwissProt_2018_01/classif_EC_pred_lvl_4_vocab.pth",
    #     path_df_test="data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json",
    #     apriori_enzyme=False,
    #     SP_version="18_01",
    #     SP_test="21_04",
    #     max_seq_len=1024,
    # )

    # DIAMOND = SequenceKNN(
    #     path_train_json="data/datasets/SwissProt_2018_01/SwissProt_2018_01_train.json",
    #     path_test_json="data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json",
    #     nb_thread=4,
    #     path_output_pred="data/prediction_for_review/pred_DIAMOND_apriori_enzyme_False_Diff_SP_18_01_21_04_without_leak_at_0.4.csv",
    #     tool="DIAMOND",
    #     a_priori_enzyme=False,
    # )
    # DIAMOND.launch_pipeline()

    # Post processing over DeepEC because cannot have complete prediction with their tools
    # Tool launch with the command: python3 deepec.py -i 'data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.fasta'  -o 'data/prediction_for_review/pred_DeepEC_Diff_SP_18_01_21_04_without_leak_at_0.4'
    # post_precessing_DeepEC(
    #     path_test_set="data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json",
    #     prediction_folder="data/prediction_for_review/pred_DeepEC_Diff_SP_18_01_21_04_without_leak_at_0.4",
    #     path_BLASTp_pred="data/prediction_for_review/pred_DIAMOND_apriori_enzyme_False_Diff_SP_18_01_21_04_without_leak_at_0.4.csv",
    #     SP_test="21_04",
    #     apriori_enzyme=False,
    # )
