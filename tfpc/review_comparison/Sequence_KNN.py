import os
import pandas as pd
from pathlib import Path
import shutil
from Bio import SearchIO
from review_comparison.utils import generate_fasta_file


class SequenceKNN:
    def __init__(
        self,
        path_train_json,
        path_test_json,
        nb_thread,
        path_output_pred,
        tmp_folder=Path("data/tmp_blastp/"),
        tool="BLASTp",
        a_priori_enzyme=False,
    ):
        self.tool = tool
        self.df_train = pd.read_json(path_train_json)
        self.df_test = pd.read_json(path_test_json)
        self.tmp_folder = tmp_folder
        self.nb_thread = nb_thread
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)
        self.name_fasta_train = "train.fasta"
        self.name_fasta_test = "test.fasta"
        self.path_fasta_train = tmp_folder / Path(self.name_fasta_train)
        self.path_fasta_test = tmp_folder / Path(self.name_fasta_test)
        self.output_querry = "res_blastp.txt"
        self.path_output_pred = path_output_pred
        self.a_piori_enzyme = a_priori_enzyme

    def launch_pipeline(self):
        generate_fasta_file(df=self.df_train, fasta_path=self.path_fasta_train)
        self.create_blast_db()
        generate_fasta_file(df=self.df_test, fasta_path=self.name_fasta_test)
        self.querry_seq()
        # self.parse_res_and_get_pred()
        # shutil.rmtree(self.tmp_folder)

    def create_blast_db(self):
        os.chdir(self.tmp_folder)
        if self.tool == "BLASTp":
            command = f"makeblastdb -in {str(self.name_fasta_train)} -dbtype prot"
        elif self.tool == "DIAMOND":
            command = f"diamond makedb --in {str(self.name_fasta_train)} -d {str(self.name_fasta_train)+'.dmnd'}"
        else:
            raise ValueError("tool unkwown")
        print("command:", command)
        os.system(command)

    def querry_seq(self):
        if not os.path.exists(self.output_querry):
            if self.tool == "BLASTp":
                command = (
                    "blastp -query "
                    + self.name_fasta_test
                    + " -db "
                    + self.name_fasta_train
                    + " -out "
                    + self.output_querry
                    + " -outfmt 5"  # XML output
                    + " -num_threads "
                    + str(self.nb_thread)
                    # + " -mt_mode 1" # Deprecated options? not working anymore
                )

            elif self.tool == "DIAMOND":
                command = (
                    "diamond blastp -q "
                    + self.name_fasta_test
                    + " -d "
                    + self.name_fasta_train
                    + " -o "
                    + self.output_querry
                    + " --threads "
                    + str(self.nb_thread)
                    + " -b5 -c1 -k 1"
                    # + " -mt_mode 1" # Deprecated options? not working anymore
                )

            else:
                raise ValueError("tool unkwown")
            print("command:", command)
            os.system(command)
        else:
            print("File already computed")

    def create_dico_res_blastp(self):
        qresults = SearchIO.parse(self.output_querry, "blast-xml")
        dico = {}
        dico_seq_identity = {}
        for qresult in qresults:
            if len(qresult.hits) != 0:
                # print("qresult.hits:", qresult.hits[0][0])
                # print("qresult.hits:", qresult.hits[0][0].ident_num)
                # print("qresult.hits:", qresult.hits[0][0].aln_span)
                # print("qresult.hits:", qresult.hits[0][0].pos_num)
                # print("qresult.hits:", len(qresult.hits[0][0].query))
                # print("qresult.hits:", len(qresult.hits[0][0].hit))

                dico[qresult.id] = [res.id for res in qresult.hits]
                dico_seq_identity[qresult.id] = [
                    res[0].ident_num / qresult.seq_len for res in qresult.hits
                ]
        return dico, dico_seq_identity

    def create_dico_ec_train(self):
        return {
            row["id_uniprot"]: row["ec_number"]
            for index, row in self.df_train.iterrows()
        }

    def parse_res_and_get_pred(self):
        """
        Parse output of blastp file and generate prediction from it
        """
        dico_querry_to_target, dico_seq_identity = self.create_dico_res_blastp()
        dico_train_id_to_ec = self.create_dico_ec_train()
        os.chdir("../../")
        print("self.path_output_pred:", self.path_output_pred)
        with open(self.path_output_pred, "w") as output_pred:
            output_pred.write("uniprot_id, pred_ec, seq_identity\n")
            for _, row in self.df_test.iterrows():
                sequence = row["sequence"]
                uniprot_id = row["id_uniprot"]
                if uniprot_id in dico_querry_to_target.keys():
                    list_ids = dico_querry_to_target[uniprot_id]
                    list_seq_seq_identity = dico_seq_identity[uniprot_id]
                    pred = dico_train_id_to_ec[list_ids[0]]
                    seq_identity = list_seq_seq_identity[0]
                    if self.a_piori_enzyme:
                        ind = 1
                        while ind < len(list_ids):
                            # while pred == "0.0.0.0" and ind < len(list_ids):
                            pred = dico_train_id_to_ec[list_ids[ind]]
                            seq_identity = list_seq_seq_identity[ind]
                            ind += 1

                else:
                    pred = "-"
                    seq_identity = 0.0
                output_pred.write(
                    uniprot_id + "," + pred + "," + str(seq_identity) + "\n"
                )
