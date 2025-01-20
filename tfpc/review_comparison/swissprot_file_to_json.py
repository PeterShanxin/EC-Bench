import gzip
import os
from pathlib import Path
import shutil
import tarfile
import pandas as pd
from tqdm import tqdm
from Bio.SeqIO.UniprotIO import UniprotIterator
from utils.Gene_ontology_utils import Ontology
from collections import Counter
import logging


class SwissProtExtractor:
    def __init__(self, dataset_name, dataset_folder="data/datasets/"):
        self.dataset_name = dataset_name
        self.dataset_folder = Path(dataset_folder)
        self.SwissProt_folder_path = self.dataset_folder / Path(dataset_name)
        self.file_output_path = self.SwissProt_folder_path / Path(
            dataset_name + "_train.json"
        )

    def process(self):
        xml_path = self.extract_XML()
        if not os.path.exists(self.file_output_path):
            dataframe = self.parse_XML(xml_path)
            dataframe.to_json(self.file_output_path)
        else:
            dataframe = pd.read_json(self.file_output_path)
        dataframe = self.remove_duplicate(dataframe)
        dataframe.to_json(self.file_output_path)
        logging.info(dataframe)
        self.show_info_dataframe(dataframe)
        self.create_the_different_split()

    def create_the_different_split(self):
        mock_seq = "M"
        mock_id = "ENZ"
        mock_ec = "0.0.0.0"
        df_valid = pd.DataFrame.from_dict(
            {
                "sequence": [mock_seq],
                "uniprot_id": [mock_id],
                "ec_number": [mock_ec],
            }
        )
        df_valid.to_json(
            self.SwissProt_folder_path / Path(self.dataset_name + "_valid.json")
        )
        df_test = pd.DataFrame.from_dict(
            {
                "sequence": [mock_seq],
                "uniprot_id": [mock_id],
                "ec_number": [mock_ec],
            }
        )
        df_test.to_json(
            self.SwissProt_folder_path / Path(self.dataset_name + "_test.json")
        )

    def get_archive_path_and_name(self):
        list_files = os.listdir(self.SwissProt_folder_path)
        list_files = [
            filename for filename in list_files if filename.endswith(".tar.gz")
        ]
        if len(list_files) == 1:
            return self.SwissProt_folder_path / Path(list_files[0])
        else:
            raise RuntimeError("Multiple or zero tar.gz file in the folder")

    def extract_XML(self):
        xml_path = self.SwissProt_folder_path / Path("uniprot_sprot.xml")
        if os.path.exists(xml_path):
            return xml_path
        TMP_ARCHIVE = self.get_archive_path_and_name()
        # Extract the SwissProt archive
        file = tarfile.open(TMP_ARCHIVE)
        file.extractall(self.SwissProt_folder_path)
        file.close()

        # Extract the xml of SwissProt
        with gzip.open(
            self.SwissProt_folder_path / Path("uniprot_sprot.xml.gz"), "rb"
        ) as f_in:
            with open(xml_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Delete temporary folder
        os.remove(self.SwissProt_folder_path / Path("uniprot_sprot.xml.gz"))
        os.remove(self.SwissProt_folder_path / Path("uniprot_sprot_varsplic.fasta.gz"))
        os.remove(self.SwissProt_folder_path / Path("uniprot_sprot.fasta.gz"))
        os.remove(self.SwissProt_folder_path / Path("uniprot_sprot.dat.gz"))

        return xml_path

    @staticmethod
    def delete_duplicate_info_ec(list_ec):
        """
        Delete not complete ec and delete duplicate information like [3.1.2.-,3.1.2.4] -> the first one doesn't add information
        """
        list_to_remove = []
        for ec in list_ec:
            number_each_level = ec.split(".")
            for k in reversed(range(1, 4)):
                if number_each_level[k] != "-":
                    number_each_level[k] = "-"
                    derived_ec = ".".join(number_each_level)
                    list_to_remove.append(derived_ec)

        for ec_to_remove in list_to_remove:
            if ec_to_remove in list_ec:
                list_ec.remove(ec_to_remove)

        return list_ec

    @staticmethod
    def get_set_GO_descendant_and_itself_from_catalytic_activity():
        Gene_Ontology = Ontology("data/GO_files/go_04_2023.obo")
        GO_catalytic_activity = "GO:0003824"
        descendant = Gene_Ontology.get_descendents(GO_catalytic_activity)
        descendant.add(GO_catalytic_activity)
        return descendant

    def parse_XML(self, xml_path):
        GO_catalytic_activity = (
            SwissProtExtractor.get_set_GO_descendant_and_itself_from_catalytic_activity()
        )
        fichier_xml = open(xml_path, "r")
        nb_monofunc = 0
        nb_multi_fonc = 0
        nb_not_enzyme = 0
        nb_suppose_not_enzyme = 0

        list_seq = []
        list_uniprot_id = []
        list_all_ec_number = []
        for entry in tqdm(
            UniprotIterator(fichier_xml), total=575000
        ):  # Total is an estimation not the real number
            sequence = str(entry.seq)
            uniprot_id = entry.id  # .annotations["accessions"]
            if "recommendedName_ecNumber" in entry.annotations:
                list_ec_number = entry.annotations["recommendedName_ecNumber"]
                list_ec_number = SwissProtExtractor.delete_duplicate_info_ec(
                    list_ec_number
                )
                if (
                    len(list_ec_number) == 1 and "-" not in list_ec_number[0]
                ):  # Monofunctionelle and complete annotation
                    # logging.info("ecNumber:", list_ec_number)
                    ec_number = list_ec_number[0]
                    nb_monofunc += 1
                elif len(list_ec_number) > 1:
                    nb_multi_fonc += 1
                    continue
                else:  # Doesn't posses a precise annotation (level 4) but it is an enzyme
                    continue
            else:
                # logging.info(entry)
                # logging.info(entry.annotations)
                nb_suppose_not_enzyme += 1
                # Different filter to be sure it is not an enzyme
                catalytic_activity = False
                list_GO_terms = [dbref[2:] for dbref in entry.dbxrefs if "GO:" in dbref]
                for GO_term in list_GO_terms:
                    if GO_term in GO_catalytic_activity:
                        catalytic_activity = True
                if catalytic_activity:
                    continue
                elif (
                    "enzyme"
                    in ",".join(entry.annotations["recommendedName_fullName"]).lower()
                ):
                    continue
                elif (
                    "keywords" in entry.annotations
                    and "enzyme" in ",".join(entry.annotations["keywords"]).lower()
                ):
                    continue
                else:
                    ec_number = "0.0.0.0"  # Code used for not an enzyme
                    nb_not_enzyme += 1
            list_seq.append(sequence)
            list_uniprot_id.append(uniprot_id)
            list_all_ec_number.append(ec_number)
        logging.info("nb_monofunc: %d", nb_monofunc)
        logging.info("nb_multi_fonc: %d", nb_multi_fonc)
        logging.info("nb_suppose_not_enzyme: %d", nb_suppose_not_enzyme)
        logging.info("nb_not_enzyme: %d", nb_not_enzyme)

        dico_df = {
            "sequence": list_seq,
            "uniprot_id": list_uniprot_id,
            "ec_number": list_all_ec_number,
        }
        df = pd.DataFrame.from_dict(dico_df)
        os.remove(self.SwissProt_folder_path / Path("uniprot_sprot.xml"))
        return df

    def remove_duplicate(self, df):
        # Keep only one representent when same sequence and annotation but different uniprot_id
        # The representend is choosen by taking the first uniprot_id when sort by alphanumerical order
        # Remove from the dataframe when annotations are different

        conflicting_sequence = self.get_conflicting_annotations(df)
        logging.info(
            "There are %d conflicting annotation for the same sequence",
            len(conflicting_sequence),
        )

        logging.info("Len df with duplicate: %d", len(df))
        df = df.sort_values("uniprot_id")  # Sort by alphanumerical order of uniprot_id
        df = df.drop_duplicates(
            subset=["sequence"], keep="first"
        )  # Keep only the first in alphanumerical order of uniprot_id
        # Remove sequence that have conflicting annotations (At least two different annotations).
        df = df[~df["sequence"].isin(conflicting_sequence)]
        logging.info("Len df without duplicate: %d", len(df))
        return df

    def get_conflicting_annotations(self, df):
        df_duplicate = df[df.duplicated(subset=["sequence"], keep=False)]
        df_duplicate = df_duplicate.sort_values("sequence")
        logging.info(df)
        conflicting_sequence = []
        count_seq = Counter(df_duplicate["sequence"])
        logging.info("Start remove duplicate sequence")
        for sequence, count in tqdm(count_seq.items()):
            if count > 1:
                if not self.same_annotation(
                    df_duplicate[df_duplicate["sequence"] == sequence]
                ):
                    conflicting_sequence.append(sequence)
        return conflicting_sequence

    def same_annotation(self, df):
        return len(df["ec_number"].unique()) == 1

    def show_info_dataframe(self, df):
        len_df = len(df)
        nb_uniq_seq = len(df.sequence.unique())
        nb_uniq_id = len(df.uniprot_id.unique())
        logging.info(f"len_df: {len_df}")
        logging.info(f"nb_uniqu_seq: {nb_uniq_seq}")
        logging.info(f"nb_uniqu_id: {nb_uniq_id}")
