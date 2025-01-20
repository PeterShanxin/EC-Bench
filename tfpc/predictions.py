import argparse
import torch
from pathlib import Path
import logging
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import sys
from interpretability.utils import (
    set_hook_to_get_attention_map,
    get_attentions_map_simple
)

sys.path.append("tfpc/models_architectures/")
from wrapper_prot_bert_BFD import WrapperProtBertBFD

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--chosen_model",
    help="Which models to use: EnzBert_SwissProt_2016_08, EnzBert_SwissProt_2018_01, EnzBert_SwissProt_2021_04, EnzBert_EC40, EnzBert_ECPred40",
    type=Path,
)

parser.add_argument(
    "--fasta_path",
    help="Fasta file with the sequences",
    type=Path,
)
parser.add_argument(
    "--output_folder_path",
    help="Path of the csv output prediction",
    default="tfpc/data/predictions.csv",
    type=Path,
)
parser.add_argument(
    "--max_seq_lenght",
    help="Limit the sequence lenght and take only the begining of the sequence. This can avoid Out Of Memory error for very long sequences.",
    default=2048,
    type=int,
)
parser.add_argument(
    "--enzyme_a_priori", help="If we know the sequences are enzyme", action="store_true"
)
parser.add_argument(
    "--output_attentions_scores", help="Compute and outputs attentions scores", action="store_true"
)
parser.add_argument(
    "--top_k", help="How many prediction per sequence", default=10, type=int
)
parser.add_argument(
    "--verbose",
    help="If the prediction results are shown in the terminal",
    action="store_true",
)
args = parser.parse_args()

if args.chosen_model == "EnzBert_EC40":
    lvl = 2
else:
    lvl = 4
path_finetune_models = Path("tfpc/data/models/fine_tune_models/")
model_path = (
    path_finetune_models
    / args.chosen_model
    / Path("classif_EC_pred_lvl_" + str(lvl) + ".pth")
)
vocab_path = Path("tfpc/data/models/pre_trained_models/30_layer_uniparc+BFD/vocab.pkl")
path_class_vocab = (
    path_finetune_models
    / args.chosen_model
    / Path("classif_EC_pred_lvl_" + str(lvl) + "_vocab.pth")
)

logging.info("Load the fasta sequences")
fasta_sequences = SeqIO.parse(open(args.fasta_path), "fasta")
dico_seq_fasta_ids = {}
set_seq_ids = set()
list_sequences_new = []
for fasta in fasta_sequences:
    seq = str(fasta.seq)
    seq = seq[: args.max_seq_lenght]
    list_sequences_new.append(seq)
    id = str(fasta.id)
    if id in set_seq_ids:
        raise RuntimeError("Fasta seqid not unique")
    set_seq_ids.add(id)
    if seq not in dico_seq_fasta_ids.keys():
        dico_seq_fasta_ids[seq] = [id]
    else:
        dico_seq_fasta_ids[seq].append(id)
list_sequences = [seq for seq in dico_seq_fasta_ids.keys()]

# Load the model
logging.info("Loading the model and its vocabulary")
model = torch.load(model_path, map_location=torch.device("cpu"))
if args.output_attentions_scores:
    # Insert hook function in the model to get back attention matrix
    (
        activation,
        nb_layer,
        nb_head,
        list_hook,
        ancienne_fonciton_lib,
    ) = set_hook_to_get_attention_map(model)

    output_scores = open(args.output_folder_path/Path("attentions_scores.csv"),"w")
    output_scores.write("seqid,attentions_scores\n")


model = model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")
vocab = torch.load(vocab_path)
class_vocab = torch.load(path_class_vocab)

if "0.0.0.0" in class_vocab:
    ind_non_enzyme = class_vocab["0.0.0.0"]
else:
    ind_non_enzyme = None
inv_class_vocab = {ind: ec_number for ec_number, ind in class_vocab.items()}
inv_class_vocab_inv = {ec_number: ind for ec_number, ind in class_vocab.items()}
# save inv_class_vocab_inv to a csv file: ec_number, ind
with open(args.output_folder_path/Path("dict_annot.csv"), "w") as f:
    for ec_number, ind in inv_class_vocab_inv.items():
        f.write(f"{ec_number},{ind}\n")

pad_indice = vocab["p"]

logging.info("Create the output prediction file")
fichier_output = open(
    args.output_folder_path/Path("predictions.csv"),
    "w",
)

fichier_output.write("seqid,pred_ec,probability,class_weight,rank\n")

logging.info("Create the output prediction file with top k")
fichier_output_top = open(
    args.output_folder_path/Path("predictions_top"+str(args.top_k)+".csv"),
    "w",
)

fichier_output_top.write("seqid,pred_ec,probability,class_weight,rank\n")

softmax = torch.nn.Softmax()
# create y_pred_all as array with size of (len(list_sequences), number of classes) storing the probability of each class for each sequence
y_pred_all = []

logging.info("Start the predictions")
with torch.no_grad():
    # We calc the metric on all the dev set
    print(len(list_sequences_new))
    for sequence in tqdm(list_sequences_new):
        input_batch = torch.tensor(
            [vocab["c"]] + [vocab[l] for l in sequence]
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        weight_pred = model(input_batch)[0]
        if args.enzyme_a_priori and ind_non_enzyme is not None:
            weight_pred[ind_non_enzyme] = -np.inf
        probability_all = softmax(weight_pred)
        # convert probability_all to a list
        probability_all = probability_all.cpu().numpy().tolist()
        # append probability_all to y_pred_all
        y_pred_all.append(probability_all)

        # ind_best = torch.argmax(weight_pred).item()
        sorted_weight = torch.argsort(weight_pred, descending=True).cpu().numpy()

        # return the first ec_prediction and if its probability is greater than 0.8 return all the ec_prediction with a probability greater than 0.5 else return the first one
        i = 0
        while i < len(sorted_weight):
            current_ind = sorted_weight[i]
            ec_prediction = inv_class_vocab[current_ind]
            class_weight = weight_pred[current_ind].item()
            probability = softmax(weight_pred)[current_ind].item()
            if i > 0 and probability < 0.5:
                break
            fasta_ids = dico_seq_fasta_ids[sequence]
            for id in fasta_ids:
                fichier_output.write(
                    id
                    + ","
                    + ec_prediction
                    + ","
                    + str(probability)
                    + ","
                    + str(class_weight)
                    + ",0\n"
                )
            if args.output_attentions_scores :
                    attentions_map = get_attentions_map_simple(
                        activation,
                        nb_layer,
                        input_batch,
                        pad_indice,
                    )
                    
                    first_agreg = np.mean(attentions_map,axis=0)
                    second_agreg = np.mean(first_agreg,axis=0)
                    list_scores = list(second_agreg)
                    output_scores.write( id
                        + ',"'
                        + str(list_scores)
                        +'"\n')
            if args.verbose:
                print("-" * 90)
                print("fasta_ids:", fasta_ids)
                print("Sequence:", sequence)
                print("Predicted ec:", ec_prediction)
                print("With a weight of:", class_weight)
                print("With a probability of:", round(probability * 100, 3), "%")    
            if probability < 0.8:
                break
            else:
                i += 1
        
        '''
        for rank in range(args.top_k):
            current_ind = sorted_weight[rank]
            ec_prediction = inv_class_vocab[current_ind]
            class_weight = weight_pred[current_ind].item()
            probability = softmax(weight_pred)[current_ind].item()

            fasta_ids = dico_seq_fasta_ids[sequence]
            for seqid in fasta_ids:
                fichier_output_top.write(
                    seqid
                    + ","
                    + ec_prediction
                    + ","
                    + str(probability)
                    + ","
                    + str(class_weight)
                    + ","
                    + str(rank)
                    + "\n"
                )

            if args.output_attentions_scores :
                attentions_map = get_attentions_map_simple(
                    activation,
                    nb_layer,
                    input_batch,
                    pad_indice,
                )
                
                first_agreg = np.mean(attentions_map,axis=0)
                second_agreg = np.mean(first_agreg,axis=0)
                list_scores = list(second_agreg)
                output_scores.write( seqid
                    + ',"'
                    + str(list_scores)
                    +'"\n')

            if args.verbose:
                print("-" * 90)
                print("fasta_ids:", fasta_ids)
                print("Sequence:", sequence)
                print("Predicted ec:", ec_prediction)
                print("With a weight of:", class_weight)
                print("With a probability of:", round(probability * 100, 3), "%")
        '''
    y_pred_all = np.array(y_pred_all)
    # save y_pred_all to a file npy
    np.save(args.output_folder_path/Path("y_pred.npy"), y_pred_all)


fichier_output.close()