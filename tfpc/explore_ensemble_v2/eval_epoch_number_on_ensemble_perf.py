import pandas as pd
import logging
from utils_stats import output_summary_of_ensemble_strategy
from utils.json_loader import load_json_into_pandas_dataframe

df = load_json_into_pandas_dataframe(
    "data/datasets/EC_prediction/EC_prediction_valid.json"
)

dict_all_ensembles = {
    "ProtBert_EC40_classic": [
        "for_the_paper/ProtBert_EC40_classic_r1_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper/ProtBert_EC40_classic_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper/ProtBert_EC40_classic_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
    "ProtBert_EC40_layer_norm_proba": [
        "for_the_paper/ProtBert_EC40_layer_norm_proba_r1_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper/ProtBert_EC40_layer_norm_proba_r2_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper/ProtBert_EC40_layer_norm_proba_r3_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper/ProtBert_EC40_layer_norm_proba_r4_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
        "for_the_paper/ProtBert_EC40_layer_norm_proba_r5_complete_output_on_valid_limit_output_False_cls_outFalse.pkl",
    ],
}


all_name_strategy = {
    "gradient_descent_opti_on_weight_1": ["weight", 1],
    "gradient_descent_opti_on_weight_5": ["weight", 5],
    "gradient_descent_opti_on_weight_10": ["weight", 10],
    "gradient_descent_opti_on_weight_50": ["weight", 50],
    "gradient_descent_opti_on_weight_100": ["weight", 100],
    "gradient_descent_opti_on_weight_500": ["weight", 500],
    "gradient_descent_opti_on_weight_1500": ["weight", 1500],
}


logging.getLogger().setLevel(logging.DEBUG)
logging.info("Début du script")
output_summary_of_ensemble_strategy(
    dict_all_ensembles=dict_all_ensembles,
    all_name_strategy=all_name_strategy,
    df=df,
    nb_partie_cross_val=10,
    save_name="result_for_differents_nb_epoch_for_gradients_strategy.json",
)