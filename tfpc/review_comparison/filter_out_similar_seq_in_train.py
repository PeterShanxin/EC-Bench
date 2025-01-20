import logging
import os
from pathlib import Path
import random
import shutil

import pandas as pd
from Bio import SeqIO
from review_comparison.utils import generate_fasta_file


class DeleteToSimilarSeq:
    def __init__(
        self,
        name_train_dataset,
        name_diff_dataset,
        min_seq_identity,
        only_representative_seq,
        cov_mode,
        colname_uniprot_id="uniprot_id",
        colname_sequence="sequence",
    ):
        self.name_train_dataset = name_train_dataset
        self.name_diff_dataset = name_diff_dataset
        self.colname_sequence = colname_sequence
        self.colname_uniprot_id = colname_uniprot_id
        self.fasta_name_all_sequences = "all_sequences.fasta"
        self.min_seq_identity = min_seq_identity
        self.cov_mode = cov_mode
        self.only_representative_seq = only_representative_seq

        dataset_folder = Path("data/datasets/")
        self.folder_train_dataset = dataset_folder / Path(name_train_dataset)
        self.folder_diff_dataset = dataset_folder / Path(name_diff_dataset)

        self.path_cluster_file_mmseq2 = self.folder_diff_dataset / "mmseq2_res"
        self.fasta_path_all_sequences = (
            self.folder_diff_dataset / self.fasta_name_all_sequences
        )
        self.path_output_mmseq2 = self.folder_diff_dataset / "output_mmseq2/"

    def start_workflow(self):
        self.get_df()
        generate_fasta_file(df=self.df_total, fasta_path=self.fasta_path_all_sequences)
        self.generate_mmseq_cluster()
        self.create_cluster_dict()
        filter_test_df = self.filter_out_too_similar_seq()
        output_path = self.folder_diff_dataset / Path(
            self.name_diff_dataset
            + "_without_leak_at_"
            + str(self.min_seq_identity)
            + ".json"
        )
        filter_test_df.to_json(output_path)

    def get_df(self):
        json_train = self.folder_train_dataset / Path(
            self.name_train_dataset + "_train.json"
        )
        self.df_train = pd.read_json(json_train)
        json_diff = self.folder_diff_dataset / Path(self.name_diff_dataset + ".json")
        self.df_diff = pd.read_json(json_diff)
        self.df_total = pd.concat(
            [self.df_train, self.df_diff], ignore_index=True, verify_integrity=True
        )
        self.verify_uniprot_id_allways_same_seq(self.df_total)
        self.df_total = self.df_total.drop_duplicates()
        logging.info(
            "There are %d duplicate uniprot_id",
            sum(self.df_total.duplicated(subset=["uniprot_id"])),
        )

    def verify_uniprot_id_allways_same_seq(self, df_total):
        dico_verif = {}
        for _, row in df_total.iterrows():
            seq = row[self.colname_sequence]
            id = row[self.colname_uniprot_id]
            if seq in dico_verif.keys():
                assert dico_verif[seq] == id
            else:
                dico_verif[seq] = id

    def generate_mmseq_cluster(self):
        # This method create a file name clusterRes_cluster.tsv with description of the close identity cluster
        logging.info("Start of mmseq2 to create similarity cluster")

        os.system(
            "mmseqs easy-cluster "
            + str(self.fasta_path_all_sequences)
            + " "
            + str(self.path_cluster_file_mmseq2)
            + " "
            + str(self.path_output_mmseq2)
            + " --min-seq-id "
            + str(self.min_seq_identity)
            + " --cov-mode "
            + str(self.cov_mode)
        )
        shutil.rmtree(self.path_output_mmseq2)
        os.remove(str(self.path_cluster_file_mmseq2) + "_all_seqs.fasta")
        os.remove(str(self.path_cluster_file_mmseq2) + "_rep_seq.fasta")

    def create_cluster_dict(self):
        logging.info("Parse result of mmseq2 to a python dictionary")
        # Create dict of clustering with key correponding to id_of_prot_ref and value list of
        # protein attach to this prot ref
        dict_cluster = dict()
        filename = str(self.path_cluster_file_mmseq2) + "_cluster.tsv"
        fichier_cluster = open(filename)
        all_lines = fichier_cluster.readlines()

        for line in all_lines:
            ref, appart = line.split("\t")
            ref = ref.strip()
            appart = appart.strip()
            if ref not in dict_cluster.keys():
                dict_cluster[ref] = [appart]
            else:
                dict_cluster[ref].append(appart)
        self.dict_cluster = dict_cluster

    def filter_out_too_similar_seq(self):
        forbiden_id = set(self.df_train[self.colname_uniprot_id])
        allowed_uniprot_id = []
        for key_cluster, content_cluster in self.dict_cluster.items():
            content_cluster = set(content_cluster)
            intersection = content_cluster.intersection(forbiden_id)
            if len(intersection) == 0:  # Allow to take this cluster
                if self.only_representative_seq:
                    allowed_uniprot_id += [key_cluster]
                else:
                    allowed_uniprot_id += content_cluster

        allowed_uniprot_id = list(set(allowed_uniprot_id))

        filter_test_df = self.df_diff[
            self.df_diff[self.colname_uniprot_id].isin(allowed_uniprot_id)
        ]
        return filter_test_df
