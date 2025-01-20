import pandas as pd
import logging
from utils_stats import output_summary_of_ensemble_strategy
from utils.json_loader import load_json_into_pandas_dataframe

"""
dict_all_ensembles = {
    "My_network_with_label_smoothing": [
        "old_before_may_2021/label_smoothing_EC_pred/fine_tune_for_EC_prediction_with_label_smoothing_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/label_smoothing_EC_pred/fine_tune_for_EC_prediction_with_label_smoothing_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/label_smoothing_EC_pred/fine_tune_for_EC_prediction_with_label_smoothing_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/label_smoothing_EC_pred/fine_tune_for_EC_prediction_with_label_smoothing_r4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/label_smoothing_EC_pred/fine_tune_for_EC_prediction_with_label_smoothing_r5_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
    "My_network_classic": [
        "old_before_may_2021/My_ensemble_EC_pred/fine_tune_for_EC_prediction_with_lr_scheduler_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/My_ensemble_EC_pred/fine_tune_for_EC_prediction_with_lr_scheduler_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/My_ensemble_EC_pred/fine_tune_for_EC_prediction_with_lr_scheduler_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/My_ensemble_EC_pred/fine_tune_for_EC_prediction_with_lr_scheduler_r4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/My_ensemble_EC_pred/fine_tune_for_EC_prediction_with_lr_scheduler_r5_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
    "ProtBert_classic": [
        "old_before_may_2021/ProtBert_EC_pred/fine_tune_ProtBert_BFD_for_EC_prediction_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/ProtBert_EC_pred/fine_tune_ProtBert_BFD_for_EC_prediction_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/ProtBert_EC_pred/fine_tune_ProtBert_BFD_for_EC_prediction_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "old_before_may_2021/ProtBert_EC_pred/fine_tune_ProtBert_BFD_for_EC_prediction_r4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
}

"""


df = load_json_into_pandas_dataframe(
    "../data/datasets/EC_prediction/EC_prediction_valid.json"
)
"""
dict_all_ensembles = {
    "ProtBert_EC40_classic": [
        "for_the_paper_ProtBert/ProtBert_EC40_classic_r1_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_ProtBert/ProtBert_EC40_classic_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_ProtBert/ProtBert_EC40_classic_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
    "ProtBert_EC40_layer_norm_proba": [
        "for_the_paper_ProtBert/ProtBert_EC40_layer_norm_proba_r1_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_ProtBert/ProtBert_EC40_layer_norm_proba_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_ProtBert/ProtBert_EC40_layer_norm_proba_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_ProtBert/ProtBert_EC40_layer_norm_proba_r4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_ProtBert/ProtBert_EC40_layer_norm_proba_r5_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
}

"""

dict_all_ensembles = {
    "MyModel_EC40_layer_norm_proba": [
        "for_the_paper_MyModel/MyModel_EC40_layer_norm_proba_r1_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_MyModel/MyModel_EC40_layer_norm_proba_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_MyModel/MyModel_EC40_layer_norm_proba_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_MyModel/MyModel_EC40_layer_norm_proba_r4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper_MyModel/MyModel_EC40_layer_norm_proba_r5_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
}


all_name_strategy = {
    "BMA_proba": ["proba"],
    "BMC_proba": ["proba"],
    "BMA_weight": ["weight"],
    "BMC_weight": ["weight"],
    "gradient_descent_opti_on_proba": ["proba", 100],  # ["proba", 1000]
    "gradient_descent_opti_on_weight": ["weight", 100],  # ["weight", 1000]
    "proba": [],
    "weight": [],
    "proba_post_calibrate_temp_tunning_ECE": ["ECE"],
    "proba_post_calibrate_temp_tunning_KS": ["KS"],
}

logging.getLogger().setLevel(logging.DEBUG)
logging.info("Début du script")
output_summary_of_ensemble_strategy(
    dict_all_ensembles=dict_all_ensembles,
    all_name_strategy=all_name_strategy,
    df=df,
    nb_partie_cross_val=5,
    save_name="result_for_differents_strategy.json",
)