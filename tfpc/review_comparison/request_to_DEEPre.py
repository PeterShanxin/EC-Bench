import requests
import re
import time
import os
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool


# About 35 sec per pred then ~70H for all test seq
# Sometime no prediction:
# answer_content: b'Feature files not generated!!! <br /> If you are inputting file, please first check whether the input file fulfills the format requirement! <br /> If you are inputting a single sequence, that means PSI-BLAST does not find any hits against swissprot for the input sequence! <br /> Our server can run over this extreme case. If you would like to do that, please contact the developer. <br /> If you need additional help, feel free to email the developer: yu.li@kaust.edu.sa'
# answer_content: b'DEEPre could not run with the provided input.'
class Wrapper_DEEPre_server:
    def __init__(self, df_test, path_output_file):
        self.ask_pred_url = "http://www.cbrc.kaust.edu.sa/DEEPre/result.php"
        self.base_pred_res_url = (
            "http://www.cbrc.kaust.edu.sa/DEEPre/progress.php?jobid=files%2F"
        )
        self.check_time = 10  # Check time in second
        self.timeout = 1200
        self.nb_timeout_loop = self.timeout / self.check_time
        self.df_test = df_test.sample(frac=1.0)
        self.path_output_file = path_output_file

    def make_pred_df_multiprocess(self):
        output_pred = self.pre_process_pred()
        # create and configure the process pool
        with Pool(processes=5) as pool:
            # execute tasks in order
            for result in tqdm(
                pool.imap(self.process_one_input, self.df_test.iterrows()),
                total=len(self.df_test),
            ):
                predicted_ec_number, sequence = result
                if predicted_ec_number is not None:
                    output_pred.write(sequence + "," + predicted_ec_number + "\n")
                    self.seq_allready_seen.append(sequence)

    def make_pred_df(self):
        output_pred = self.pre_process_pred()
        for couple in tqdm(self.df_test.iterrows(), total=len(self.df_test)):
            predicted_ec_number, sequence = self.process_one_input(couple)
            if predicted_ec_number is not None:
                output_pred.write(sequence + "," + predicted_ec_number + "\n")
                self.seq_allready_seen.append(sequence)

    def process_one_input(self, couple):
        _, row = couple
        sequence = row["sequence"]
        if sequence not in self.seq_allready_seen:
            predicted_ec_number = self.pred(sequence)
            if predicted_ec_number == "NO_PRED":
                predicted_ec_number = "NO_PRED"
        else:
            predicted_ec_number = None
        return predicted_ec_number, sequence

    def pre_process_pred(self):
        if os.path.exists(self.path_output_file):
            print("Re open previous file")
            current_df_pred = pd.read_csv(self.path_output_file)
            print("current_df_pred:", current_df_pred)
            self.seq_allready_seen = list(current_df_pred["sequence"])
            output_pred = open(self.path_output_file, "a")
        else:
            print("Creating new file")
            output_pred = open(self.path_output_file, "w")
            self.seq_allready_seen = []
            output_pred.write("sequence, pred_ec\n")
        return output_pred

    def send_job(self, sequence):
        myobj = {
            # "file": "",
            "sequence": sequence,
            "knowledge": "No",
        }
        try:
            output_file_id_page = requests.post(self.ask_pred_url, data=myobj)
            file_id = re.findall(
                r"files/single_\d*.fasta.res", output_file_id_page.text
            )[0]
            file_id = file_id.split("/")[1]
            return file_id
        except:
            print("ERROR post request")
            return None

    def wait_for_answer(self, file_id):
        pred_url = self.base_pred_res_url + file_id
        nb = 0
        processed = False
        while not processed:
            time.sleep(self.check_time)
            r = requests.get(pred_url)
            if not "RUNNING" in str(r.content):
                processed = True
            nb += 1
            if nb > self.nb_timeout_loop:
                print("TIMEOUT")
                return ""
        return str(r.content)

    def parse_pred(self, answer_content):
        all_ec_number = re.findall(
            r"\d{1,4}\.\d{1,4}\.\d{1,4}\.\d{1,4}", answer_content
        )
        if len(all_ec_number) == 0:
            print("answer_content:", answer_content)
            ec_number = "NO_PRED"
        elif len(all_ec_number) != 1:
            raise RuntimeError("Multiple predictions")
        else:
            ec_number = all_ec_number[0]
        return ec_number

    def pred(self, sequence):
        print("sequence:", repr(sequence))
        file_id = self.send_job(sequence)
        if file_id is not None:
            answer_content = self.wait_for_answer(file_id)
            ec_number = self.parse_pred(answer_content)
            return ec_number
        else:
            return None
