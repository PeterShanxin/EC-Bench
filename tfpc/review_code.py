from review_comparison.compute_metric import compute_metric
from review_comparison.generate_all_predictions import generate_all_predictions
from review_comparison.parse_and_create_the_different_dataset import (
    create_the_different_dataset,
)
import logging


logging.getLogger().setLevel(logging.INFO)

# Create the different dataset: SwissProt_2016_08, SwissProt_2018_01, SwissProt_2023_02, Diff_SP_16_08_23_02 and Diff_SP_18_01_23_02
# create_the_different_dataset()

# Generate the prediction for the differents models: DEEPre, DeepEC, BLASTp and EnzBert
# generate_all_predictions()

# Evaluate the differents predictions with Macro-F1 for each EC level: (0 or 1) to 4
compute_metric()
