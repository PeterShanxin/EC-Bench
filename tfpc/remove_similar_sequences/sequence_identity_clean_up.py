import os
import pandas as pd
import logging
from utils.json_loader import load_json_into_pandas_dataframe


class SequenceIdentityCleanUp:
    """
    This class take a list of path of folder dataset and allow us to generate the same dataset folder without inter contamination.
    A contamination is when some sequence from the test set of one of the task is to similar(with a threshold) from a sequence from a train/valid set of an other task.
    This class use mmseq2 to determine the similar sequence from a threshold
    """

    def __init__(
        self, list_path_folder_datasets, sequence_identity_threshold, cov_mod=0
    ):
        self.list_path_folder_datasets = list_path_folder_datasets
        self.all_xp_names = [p.split("/")[-1] for p in list_path_folder_datasets]
        self.list_train_df = [
            load_json_into_pandas_dataframe(path_folder + "/" + xp_name + "_train.json")
            for xp_name, path_folder in zip(
                self.all_xp_names, list_path_folder_datasets
            )
        ]
        self.list_valid_df = [
            load_json_into_pandas_dataframe(path_folder + "/" + xp_name + "_valid.json")
            for xp_name, path_folder in zip(
                self.all_xp_names, list_path_folder_datasets
            )
        ]
        self.list_test_df = [
            load_json_into_pandas_dataframe(path_folder + "/" + xp_name + "_test.json")
            for xp_name, path_folder in zip(
                self.all_xp_names, list_path_folder_datasets
            )
        ]
        self.col_sequence = self.detect_and_return_col_sequence()
        self.sequence_identity_threshold = sequence_identity_threshold
        self.cov_mod = cov_mod
        self.temp_folder_name = "data/tmp_sequence_identity_cleanup"
        os.mkdir(self.temp_folder_name)

    def generate_decontaminated_dataset(self):
        self.generate_fasta_all_sequences()
        self.generate_cluster()
        self.list_id_to_exclude()
        new_list_train_df = [
            self.decontaminate(df, ind) for ind, df in enumerate(self.list_train_df)
        ]
        new_list_valid_df = [
            self.decontaminate(df, ind) for ind, df in enumerate(self.list_valid_df)
        ]
        self.save_new_data(new_list_train_df, new_list_valid_df)
        self.remove_useless_files()

    def remove_useless_files(self):
        os.system("cd ..")
        os.system("rm -r " + self.temp_folder_name)

    def decontaminate(self, df, ind_df):
        logging.debug("Pour l'experience %s :", self.all_xp_names[ind_df])
        dict_seq = self.create_dict_from_fasta()

        fichier_avoid = open(self.temp_folder_name + "/id_to_exclude_in_train.txt")
        id_to_avoid = fichier_avoid.readlines()
        id_to_avoid = [line[:-1] for line in id_to_avoid]

        id_to_delete = []
        new_df = df.reset_index(drop=True)
        for k in range(len(df)):
            element = new_df.iloc[k][self.col_sequence[ind_df]]
            if self.suppr_element(element, dict_seq, id_to_avoid):
                id_to_delete.append(k)
        logging.debug(
            "Il y a %s séquence dans ce dataset avant suppression", len(new_df)
        )
        logging.debug("je supprime %s proteines de ce fichier", len(id_to_delete))
        new_df = new_df.drop(id_to_delete)
        new_df = new_df.reset_index(drop=True)
        logging.debug(
            "Il y a %s séquence dans ce dataset après suppression", len(new_df)
        )
        return new_df

    def sequence_in_the_same_cluster(self, dict_cluster, seq, dict_seq):
        id_sequence = dict_seq[seq]
        allready_find = False
        seq_cluster = []
        value_of_clust = None
        for key, value in dict_cluster.items():
            if id_sequence in value:
                if allready_find:
                    print("ERREUR sequence présente dans plusieurs cluster")
                    exit(1)
                allready_find = True
                value_of_clust = value
        if allready_find:
            return value_of_clust
        else:
            print("ERREUR sequence non trouvé")
            return False

    def suppr_element(self, element, dict_seq, id_to_avoid):
        if dict_seq[element] in id_to_avoid:
            return True
        return False

    def create_dict_from_fasta(self):
        # Extract the correponding id for each sequence and make it a dict
        # The key is the sequence and the value is the id
        fichier_sequence = open(self.temp_folder_name + "/all_sequences.fasta")
        all_lines = fichier_sequence.readlines()
        all_keys = [id[1:].strip() for id in all_lines[::2]]
        all_seqs = [seq.strip() for seq in all_lines[1::2]]
        # On vérifie qu'il n"y a pas de séquence en double
        assert len(list(set(all_seqs))) == len(all_seqs)
        dict_seq = {all_seqs[k]: all_keys[k] for k in range(len(all_keys))}
        return dict_seq

    def list_id_to_exclude(self):
        dict_seq = self.create_dict_from_fasta()

        # Create dict of clustering with key correponding to id_of_prot_ref and value list of protein attach to this prot ref
        dict_cluster = dict()
        fichier_cluster = open(self.temp_folder_name + "/clusterRes_cluster.tsv")
        all_lines = fichier_cluster.readlines()

        for line in all_lines:
            ref, appart = line.split("\t")
            ref = ref.strip()
            appart = appart.strip()
            if ref not in dict_cluster.keys():
                dict_cluster[ref] = [appart]
            else:
                dict_cluster[ref].append(appart)

        # All test sequences
        all_sequences_test = []
        for indice_fold in range(len(self.list_path_folder_datasets)):
            all_sequences_test += self.list_test_df[indice_fold][
                self.col_sequence[indice_fold]
            ].to_list()

        # Construcut the list of all test sequence and all sequence in train that have to much sequence identity with the test
        id_seq_to_avoid = []
        for seq in all_sequences_test:
            content_cluster = self.sequence_in_the_same_cluster(
                dict_cluster, seq, dict_seq
            )
            if content_cluster is False:
                Nb_not_find += 1
                id_seq_to_avoid += [dict_seq[seq]]
            else:
                id_seq_to_avoid += content_cluster

        id_seq_to_avoid = list(set(id_seq_to_avoid))

        fichier_sortie = open(
            self.temp_folder_name + "/id_to_exclude_in_train.txt", "w"
        )
        for id_seq in id_seq_to_avoid:
            fichier_sortie.write(id_seq + "\n")

        fichier_sortie.close()

    def save_new_data(self, list_new_df_train, list_new_df_valid):
        for indice_fold, path_folder in enumerate(self.list_path_folder_datasets):
            prefix_path_folder = "/".join(path_folder.split("/")[:-1])
            new_folder_name = (
                prefix_path_folder
                + "/"
                + self.all_xp_names[indice_fold]
                + "_threshold="
                + str(self.sequence_identity_threshold)
                + "_covmod="
                + str(self.cov_mod)
                + "_without"
            )
            for xp_name in self.all_xp_names:
                new_folder_name += "_" + xp_name
            os.mkdir(new_folder_name)
            path_files = new_folder_name + "/" + self.all_xp_names[indice_fold]
            list_new_df_train[indice_fold].to_json(path_files + "_train.json")
            list_new_df_valid[indice_fold].to_json(path_files + "_valid.json")
            self.list_test_df[indice_fold].to_json(path_files + "_test.json")

    def generate_fasta_all_sequences(self):
        all_sequences = []
        for ind_dataset in range(len(self.list_train_df)):
            all_sequences += self.list_train_df[ind_dataset][
                self.col_sequence[ind_dataset]
            ].tolist()
            all_sequences += self.list_valid_df[ind_dataset][
                self.col_sequence[ind_dataset]
            ].tolist()
            all_sequences += self.list_test_df[ind_dataset][
                self.col_sequence[ind_dataset]
            ].tolist()

        # De duplicate protéine
        all_sequences = list(set(all_sequences))
        self.write_fasta_to_file(all_sequences)

    def write_fasta_to_file(self, list_of_seq):
        fichier_sortie = open(self.temp_folder_name + "/all_sequences.fasta", "w")

        for k, seq in enumerate(list_of_seq):
            fichier_sortie.write(">seq_num_" + str(k) + "\n")
            fichier_sortie.write(seq + "\n")

        fichier_sortie.close()

    def generate_cluster(self):
        os.system(
            "mmseqs easy-cluster "
            + self.temp_folder_name
            + "/all_sequences.fasta "
            + self.temp_folder_name
            + "/clusterRes tmp --min-seq-id "
            + str(self.sequence_identity_threshold)
            + " --cov-mode "
            + str(self.cov_mod)
        )
        os.system("rm -r tmp/")
        os.system("rm clusterRes_all_seqs.fasta")
        os.system("rm clusterRes_rep_seq.fasta")

    def detect_and_return_col_sequence(self):
        cols_with_seq = []
        for ind_dataset in range(len(self.list_test_df)):
            all_cols = self.list_test_df[ind_dataset].columns
            for col in all_cols:
                first_elem = self.list_test_df[ind_dataset][col][0]
                if (
                    isinstance(first_elem, str)
                    and not first_elem.isdigit()
                    and len(first_elem) > 100
                ):
                    cols_with_seq.append(col)
                    break
                else:
                    logging.debug("%s is not a col of protein sequence", col)
        logging.debug("%s are col with the sequences", cols_with_seq)
        assert len(cols_with_seq) == len(self.list_test_df)

        return cols_with_seq
