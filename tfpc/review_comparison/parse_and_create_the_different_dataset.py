# # Parse the different SwissProt release
import pandas as pd
from review_comparison.Create_diff_SwissProt_new_annotations import diff_SwissProt
from review_comparison.filter_out_similar_seq_in_train import DeleteToSimilarSeq
from review_comparison.swissprot_file_to_json import SwissProtExtractor
from review_comparison.utils import generate_fasta_file


def create_the_different_dataset():
    all_SwissProt_release = [
        "SwissProt_2016_08",
        "SwissProt_2018_01",
        "SwissProt_2021_04",
    ]
    for SwissProt_release in all_SwissProt_release:
        extractor = SwissProtExtractor(SwissProt_release)
        extractor.process()

    # Extract only new sequences
    diff_SwissProt(dataset_name1="SwissProt_2016_08", dataset_name2="SwissProt_2021_04")
    diff_SwissProt(dataset_name1="SwissProt_2018_01", dataset_name2="SwissProt_2021_04")

    # Filter sequences that are too similar from a training sequence
    min_seq_identity = 0.4
    filter1 = DeleteToSimilarSeq(
        name_train_dataset="SwissProt_2016_08",
        name_diff_dataset="Diff_SP_16_08_21_04",
        min_seq_identity=min_seq_identity,
        only_representative_seq=True,
        cov_mode=0,
    )
    filter1.start_workflow()

    filter2 = DeleteToSimilarSeq(
        name_train_dataset="SwissProt_2018_01",
        name_diff_dataset="Diff_SP_18_01_21_04",
        min_seq_identity=min_seq_identity,
        only_representative_seq=True,
        cov_mode=0,
    )
    filter2.start_workflow()

    # Filter for DeepEC: can only process sequence between 10 and 1000 AA in lenght
    df = pd.read_json(
        "data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json"
    )
    df["len_seq"] = df["sequence"].apply(lambda x: len(x))
    df = df[(df["len_seq"] > 10) & (df["len_seq"] < 1000)]
    df.to_json(
        "data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.json"
    )
    path_fasta_test_sequence = "data/datasets/Diff_SP_18_01_21_04/Diff_SP_18_01_21_04_without_leak_at_0.4.fasta"
    generate_fasta_file(df=df, fasta_path=path_fasta_test_sequence)

    # Filter for DEEPre: can only process sequence between 50 and 5000 AA in lenght
    df = pd.read_json(
        "data/datasets/Diff_SP_16_08_21_04/Diff_SP_16_08_21_04_without_leak_at_0.4.json"
    )
    df["len_seq"] = df["sequence"].apply(lambda x: len(x))
    df = df[(df["len_seq"] > 50) & (df["len_seq"] < 5000)]
    df.to_json(
        "data/datasets/Diff_SP_16_08_21_04/Diff_SP_16_08_21_04_without_leak_at_0.4.json"
    )
    path_fasta_test_sequence = "data/datasets/Diff_SP_16_08_21_04/Diff_SP_16_08_21_04_without_leak_at_0.4.fasta"
    generate_fasta_file(df=df, fasta_path=path_fasta_test_sequence)
