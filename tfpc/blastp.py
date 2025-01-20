import pandas as pd
from review_comparison.generate_pred_EnzBert import get_EnzBert_predictions
from review_comparison.post_process_deepec import post_precessing_DeepEC
from review_comparison.Sequence_KNN import SequenceKNN
from review_comparison.request_to_DEEPre import Wrapper_DEEPre_server


def generate_all_predictions():
    
    # Predictions with Blastp
    blast_p = SequenceKNN(
        path_train_json="data/datasets/mine_30_task3/mine_30_task3_train_blastp.json",
        path_test_json="data/datasets/mine/price_blastp.json",
        nb_thread=112,
        path_output_pred="data/predictions/price_BLASTp_30.csv",
        tool="DIAMOND",
        a_priori_enzyme=True,
    )
    blast_p.launch_pipeline()
    print("blast_p done")

# call the function
generate_all_predictions()
