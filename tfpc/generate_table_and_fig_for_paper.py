"""
Script to generate all figs and tables from the paper
"""
import argparse
import logging
import os
from utils.utils_for_paper import (
    get_nb_prot_each_main_ec_class,
    get_all_prediction_weights,
    get_metric_on_EC40_test,
    generate_metric_file_ECPred40,
    generate_latex_result_from_prediction_on_ECPred40,
    compute_feature_importance_scores,
    compare_intepretability_methods,
)
from utils.metrics_ECPred import MacroAvg

FOLDER_XP = "data/models/fine_tune_models/"
FOLDER_SAVE_PREDICTIONS = "data/predictions/"
FOLDER_SAVE_RESIDUES_IMPORTANCE_SCORES = "data/residues_of_interest/"

logging.getLogger().setLevel(logging.INFO)
# Specification of the argument to specify when you launch the script
parser = argparse.ArgumentParser()
parser.add_argument(
    "fig_or_table",
    help="Which figure or table do you want to generate (table1,table3,table4,table5_and_figure4)",
)
args = parser.parse_args()

# Generate table 1: Number of sequence for each main classes for each datasets
if args.fig_or_table == "table1":
    get_nb_prot_each_main_ec_class(
        dataset_name="EC40",
        all_sets=["train", "valid", "test"],
        colname_ec="label",
        separator="-",
    )
    get_nb_prot_each_main_ec_class(
        dataset_name="ECPred40",
        all_sets=["train", "valid", "test"],
        colname_ec="EC Number",
        separator=".",
    )
    get_nb_prot_each_main_ec_class(
        dataset_name="SwissProt_2021_04",
        all_sets=["train"],
        colname_ec="ec_number",
        separator=".",
    )

# Generate table 3: Comparison with UDSMProt of the prediction quality at the two levels of EC40 test set.
elif args.fig_or_table == "table3":
    TEST_PREDICTION_PATH = (
        FOLDER_SAVE_PREDICTIONS
        + "all_weights_model_EnzBert_EC40_on_dataset_EC40_test.csv"
    )
    if not os.path.exists(TEST_PREDICTION_PATH):
        logging.info("Generation of the prediction files for EnzBert_EC40")
        get_all_prediction_weights(
            model_path=FOLDER_XP + "EnzBert_EC40/classif_EC_pred_lvl_2.pth",
            vocab_path="pre_trained_models/30_layer_uniparc+BFD/vocab.pkl",
            which_dataset="EC40",
            which_set="test",
            colname_sequences="primary",
            folder_save_prediction=FOLDER_SAVE_PREDICTIONS,
        )
    else:
        logging.info(
            "Skip prediction because prediction files allready present in %s",
            TEST_PREDICTION_PATH,
        )
    get_metric_on_EC40_test(
        metric_obj=MacroAvg(separator="-"),
    )

# Generate table 4: Comparison with ECPred of the prediction quality at the five levels of ECPred40 test set
elif args.fig_or_table == "table4":
    TEST_PREDICTION_PATH = (
        FOLDER_SAVE_PREDICTIONS
        + "all_weights_model_EnzBert_ECPred40_on_dataset_ECPred40_test.csv"
    )
    if not os.path.exists(TEST_PREDICTION_PATH):
        logging.info("Generation of the prediction files for EnzBert_ECPred40")
        get_all_prediction_weights(
            model_path=FOLDER_XP + "EnzBert_ECPred40/classif_EC_pred_lvl_4.pth",
            vocab_path="pre_trained_models/30_layer_uniparc+BFD/vocab.pkl",
            which_dataset="ECPred40",
            which_set="test",
            colname_sequences="sequence",
            folder_save_prediction=FOLDER_SAVE_PREDICTIONS,
        )
    else:
        logging.info(
            "Skip prediction because prediction files allready present in %s",
            TEST_PREDICTION_PATH,
        )
    generate_metric_file_ECPred40()
    # generate_latex_result_from_prediction_on_ECPred40(dir_res=FOLDER_SAVE_PREDICTIONS)

# Generate table 5: Evaluation of best interpretability method of each category with respect to the M-CSA dataset
elif args.fig_or_table == "table5_and_figure4":
    if not os.path.exists(FOLDER_SAVE_RESIDUES_IMPORTANCE_SCORES):
        os.mkdir(FOLDER_SAVE_RESIDUES_IMPORTANCE_SCORES)

    all_subdir = [x[0] for x in os.walk(FOLDER_SAVE_RESIDUES_IMPORTANCE_SCORES)]
    print(all_subdir)
    if len(all_subdir) <= 1:
        logging.info(
            "Generation of the residues importance scores with EnzBert_SwissProt_2021_04"
        )
        compute_feature_importance_scores()
    else:
        logging.info(
            "Skip computing of residues importance scores because folder %s exist",
            FOLDER_SAVE_RESIDUES_IMPORTANCE_SCORES,
        )
    compare_intepretability_methods()


# Figure 5 are present in the jupyter notebook ......


# How to launch a training?
# python3 training.py folder_with_config_inside