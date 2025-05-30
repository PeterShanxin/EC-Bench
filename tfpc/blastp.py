import pandas as pd
from review_comparison.generate_pred_EnzBert import get_EnzBert_predictions
from review_comparison.post_process_deepec import post_precessing_DeepEC
from review_comparison.Sequence_KNN import SequenceKNN
from review_comparison.request_to_DEEPre import Wrapper_DEEPre_server


def generate_all_predictions(path_train, path_test, path_output):

    # Predictions with Blastp
    blast_p = SequenceKNN(
        path_train_json=path_train,
        path_test_json=path_test,
        nb_thread=112,
        path_output_pred=path_output,
        tool="DIAMOND",
        a_priori_enzyme=True,
    )
    blast_p.launch_pipeline()
    print("blast_p done")


# call the function
generate_all_predictions(
    "data/datasets/mine_30/train_blastp.json",
    "data/datasets/price_blastp.json",
    "data/predictions/price_BLASTp_30.csv",
)
generate_all_predictions(
    "data/datasets/mine_30/train_blastp.json",
    "data/datasets/mine_30/test.json",
    "data/predictions/test_BLASTp_30.csv",
)
generate_all_predictions(
    "data/datasets/mine_100/train_blastp.json",
    "data/datasets/price_blastp.json",
    "data/predictions/price_BLASTp_100.csv",
)
generate_all_predictions(
    "data/datasets/mine_100/train_blastp.json",
    "data/datasets/mine_100/test.json",
    "data/predictions/test_BLASTp_100.csv",
)
