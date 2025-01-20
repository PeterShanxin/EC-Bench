from lime.lime_text import LimeTextExplainer
import pandas as pd
import torch
import sys
import numpy as np
from tqdm import tqdm
from utils.json_loader import load_json_into_pandas_dataframe
import scipy as sp

sys.path.append("../models_architectures/")
from wrapper_prot_bert_BFD import WrapperProtBertBFD

sys.path.append("../")

which_dataset = "binding_site"
path_model = (
    "../data/models/fine_tune_models/ProtBert_EC40_classic_r1/classif_EC_pred_lvl_2.pth"
)
path_vocab = "pre_trained_models/30_layer_uniparc+BFD/vocab.pkl"

max_seq_len = 1024
if which_dataset == "catalytic_site":
    dataframe = load_json_into_pandas_dataframe(
        "../data/datasets/catalytic_site/catalytic_site_train.json"
    )
    list_catalytic_site = []
    for index, row in dataframe.iterrows():
        arr = np.zeros((len(row["sequence"]))).astype(int)
        tab = row["catalytic_residue_position"]
        arr[tab] = 1
        list_catalytic_site.append(list(arr))
    # Exemple row["roles"] for one line : 90:['metal ligand', 'electrostatic stabiliser'];88:['electrostatic stabiliser'];90:['metal ligand'];63:['proton donor', 'proton acceptor'];93:['metal ligand'];61:['metal ligand']

    dataframe = dataframe.assign(label=list_catalytic_site)
elif which_dataset == "binding_site":
    dataframe = load_json_into_pandas_dataframe(
        "../data/datasets/binding_site_with_roles/binding_site_with_roles_train.json"
    )
else:
    logging.error(
        "%s : Ce dataset n'est pas encore supporté par la fonction", which_dataset
    )
dataframe = dataframe.sample(frac=1, random_state=1).reset_index(drop=True)

# We load the model and the vocab
if torch.cuda.is_available():
    model = torch.load(path_model)
    model = model.cuda()
else:
    model = torch.load(path_model, map_location=torch.device("cpu"))

model = model.eval()
vocab = torch.load("../data/models/" + path_vocab)
pad_indice = vocab["p"]

sequence = "cMSEQQRFNNEVEEIKKWWSSPRWKHTKRVYSPEDIASRRGTIKVPQASSQQADKLFKLLQEHEKNHTASFTYGALDPVQVTQMAKYLDSIYVSGWQSSSTASTSNEPSPDLADYPMDTVPNKVEHLWFAQLFHDRKQNEERLSLPESERSKLPAPVDYLRPIIADADTGHGGLTAVVKLTKMFIERGAAGIHIEDQAPGTKKCGHMAGKVLVPIQEHINRLIAIRASADIFGSDLLAIARTDSEAATLITSSIDYRDHYFIAGATNKDAGHLVDVMVAAELEGKQGAALQAVEDEWNRKAGVKLFHEAFADEVNAGSYSNKAELIAEFNKKVTPLSNTPALEARALAARLLGKDIYFNWEAARVREGYYRYQGGTQCAVNRGIAYAPYADLIWMESKLPDYAQAKEFAEGVKNAVPHQWLAYNLSPSFNWTTAMSPEDQETYISRLAKLGYVWQFITLAGLHTNALISDKFAKAYSERGMKAYGGEIQQPEIDQGCEVVKHQKWSGAEYIDGILRMVTGGITSTAAMGAGVTEDQFKSKL"
input_batch = torch.tensor([vocab["c"]] + [vocab[l] for l in sequence]).unsqueeze(0)


#######################SHAP PART


def encode(sequence):
    tv = torch.tensor([vocab["c"]] + [vocab[l] for l in sequence]).unsqueeze(0)
    return tv


# define a prediction function
def prediction_function(batch_sequences):
    batch_resul = None
    max_len = max([len(s) for s in batch_sequences])
    for sequence in tqdm(batch_sequences):
        if len(sequence) < max_len:
            sequence += "p" * (max_len - len(sequence))
        input = encode(sequence)
        outputs = model(input).detach().cpu()  # .numpy()
        if batch_resul is None:
            batch_resul = outputs
        else:
            batch_resul = torch.cat((batch_resul, outputs))
    return batch_resul


def get_label_from_lime(explainer, sequence):
    input_batch = encode(sequence)
    outputs = model(input_batch).detach().cpu()
    argmax = int(torch.argmax(outputs[0]))

    exp = explainer.explain_instance(
        sequence, prediction_function, num_samples=500, labels=list(range(65))
    )  # default num_samples=5000

    map_result = exp.as_map()[argmax]
    list_of_position = [m[0] for m in map_result]
    list_of_weight = [m[1] for m in map_result]
    return list_of_position


explainer = LimeTextExplainer(char_level=True, bow=False, mask_string="m")
pred_label = get_label_from_lime(explainer, sequence)
print(pred_label)