import torch
from tqdm import tqdm
import numpy as np
from interpretability.utils import (
    get_attentions_map_simple,
    set_hook_to_get_attention_map,
    load_model_and_vocab,
)
from interpretability.generate_residues_of_interests_with_attn import (
    create_score_vector,
)
from sty import fg, bg, ef, rs
from sty import Style, RgbBg
import numpy as np
import nglview as nv
import requests
import os
import pandas as pd
from Bio.PDB import *
import sys

sys.path.append("../models_architectures/")
from wrapper_prot_bert_BFD import WrapperProtBertBFD

dict_of_amino_acid = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}
list_3_letter_code = list(dict_of_amino_acid.keys())


def normalize_for_color(force_of_highlights):
    # Min max normalization
    force_of_highlights = (force_of_highlights - np.min(force_of_highlights)) / (
        np.max(force_of_highlights) - np.min(force_of_highlights)
    )
    force_of_highlights = force_of_highlights * 255
    force_of_highlights = force_of_highlights.astype(int)
    # force_of_highlights = (force_of_highlights-np.mean(force_of_highlights))/np.std(force_of_highlights)

    return force_of_highlights


def get_color_for_viz(force_of_highlights, method):
    force_of_highlights = np.array(force_of_highlights)
    if method == "classic":
        force_of_highlights = normalize_for_color(force_of_highlights)
    elif method == "classic_with_saturation":
        force_of_highlights = normalize_for_color(force_of_highlights)
        # Have more than 255 values
        force_of_highlights = force_of_highlights * 2
        # Clip value to be correct colors
        force_of_highlights = np.clip(force_of_highlights, 0, 255)
    elif method == "binary_with_threashold":
        # Discrete version with the optimal threashold
        force_of_highlights[force_of_highlights >= 0.11828187716390] = 1
        force_of_highlights[force_of_highlights < 0.11828187716390] = 0
        force_of_highlights = normalize_for_color(force_of_highlights)
    elif method == "rank_colorize":
        force_of_highlights = np.argsort(force_of_highlights)
        force_of_highlights = normalize_for_color(force_of_highlights)
    elif method == "first_percentile_colorize":
        # Les 5% des plus fort highlights en découpant en tranche de 0.5 ?
        min_percentile = 95
        step = 0.05  # 0.5
        nb_step = int((100 - min_percentile) / step)
        list_percentile = [min_percentile + k * step for k in range(nb_step)]
        new_force_of_highlights = np.zeros(force_of_highlights.shape)
        # print("list_percentile :",list_percentile)
        for ind, percentile in enumerate(list_percentile):
            threashold = np.percentile(force_of_highlights, percentile)
            new_force_of_highlights[force_of_highlights > threashold] = ind
        force_of_highlights = normalize_for_color(new_force_of_highlights)
    return force_of_highlights


def print_color_text(
    sequence, intensite_color, method="first_percentile_colorize", composante="green"
):
    force_of_highlights = intensite_color

    force_of_highlights = get_color_for_viz(force_of_highlights, method)

    intensite_color = force_of_highlights

    sequence_highlight = fg.white + ""
    symbolcount = 0
    for letter, c in zip(sequence, intensite_color):
        c = int(c)

        if composante == "green":
            bg.new_color = Style(RgbBg(255 - c, 255, 255 - c))
        elif composante == "red":
            bg.new_color = Style(RgbBg(255, 255 - c, 255 - c))
        elif composante == "blue":
            bg.new_color = Style(RgbBg(255 - c, 255 - c, 255))

        if symbolcount % 10 == 0:
            sequence_highlight += bg(255, 255, 255) + " "
        if symbolcount % 100 == 0:
            sequence_highlight += (
                "\n\n" + bg(255, 255, 255) + str(int(symbolcount / 100)) + " "
            )

        sequence_highlight += bg.new_color + letter
        symbolcount += 1

    print(sequence_highlight)


def set_color_by_residue(self, colors, component_index=0, repr_index=0):
    self._remote_call(
        "setColorByResidue", target="Widget", args=[colors, component_index, repr_index]
    )


def unit_length_norm2_normalize(vec_score):
    vec_score = np.array(vec_score, dtype=np.float64)
    norm_vec_score = np.sqrt(np.sum(np.power(vec_score, 2)))
    norm_vec = np.divide(
        vec_score,
        norm_vec_score,
        out=np.zeros_like(vec_score),
        where=norm_vec_score != 0,
    )
    return norm_vec


@torch.no_grad()
def predict_ec_and_scores_from_sequence(
    sequence,
):
    path_model = "../data/models/fine_tune_models/EnzBert_SwissProt_2021_04/classif_EC_pred_lvl_4.pth"
    path_class_vocab = "../data/models/fine_tune_models/EnzBert_SwissProt_2021_04/classif_EC_pred_lvl_4_vocab.pth"
    path_vocab = "../data/models/pre_trained_models/30_layer_uniparc+BFD/vocab.pkl"
    method = "mean_follow_by_mean_order1"
    max_seq_len = 1024
    ############################################
    # STEP 1 : We load the model and the dataset#
    ############################################

    # We load the model and the vocab
    if torch.cuda.is_available():
        model = torch.load(path_model)
        model = model.cuda()
    else:
        model = torch.load(path_model, map_location=torch.device("cpu"))

    model = model.eval()
    vocab = torch.load(path_vocab)
    pad_indice = vocab["p"]
    class_vocab = torch.load(path_class_vocab, map_location=torch.device("cpu"))
    inverse_class_vocab = {value: key for key, value in class_vocab.items()}

    ######################################################
    # STEP 2 : We hook the model to get the attention map#
    ######################################################

    (
        activation,
        nb_layer,
        nb_head,
        list_hook,
        ancienne_fonciton_lib,
    ) = set_hook_to_get_attention_map(model)

    #####################################################################################################
    # STEP 3 : We evaluate the model on the rest of the dataset, to see how well we can get the property#
    #####################################################################################################

    # We calc the metric on all the dev set
    sequence = sequence[:max_seq_len]
    # print("len seq :", len(sequence))

    input_batch = torch.tensor([vocab["c"]] + [vocab[l] for l in sequence]).unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.cuda()

    prediction_model = model(input_batch)
    ec_class = inverse_class_vocab[int(torch.argmax(prediction_model))]

    attentions_map = get_attentions_map_simple(
        activation,
        nb_layer,
        input_batch,
        pad_indice,
    )

    list_res = create_score_vector(
        attentions_map,
        nb_layer,
        nb_head,
        None,
        method=method,
    )

    list_res = unit_length_norm2_normalize(list_res)
    ##########################################################################
    # STEP 5 : We delete the hook and get the model back to the original state#
    ##########################################################################
    # We delete the hook from the model
    for hook in list_hook:
        hook.remove()
    # On rétablie la fonction correct dans la librairie pytorch
    torch.functional.multi_head_attention_forward = ancienne_fonciton_lib
    model = torch.nn.DataParallel(model)
    model = model.train()

    return (
        ec_class,
        list_res,
        np.max(torch.nn.functional.softmax(prediction_model).numpy()),
    )


def highlight_catalytic_on_3D_structure(
    view, chain_selected, indices_catalytic_res, sequence
):
    for res in chain_selected.get_unpacked_list():
        pos_in_pdb_seq = res.get_id()[1]
        if pos_in_pdb_seq in indices_catalytic_res:
            for atom in res.get_atoms():
                coord = np.array(atom.coord)
                coord = list(coord)
                break

            one_letter_code = res.get_resname()
            view.shape.add_sphere(coord, [0, 0, 1], 0.9)  # "Catalytic_"+str(res_number)
            view.shape.add_text(
                coord, [0, 0, 1], 5, one_letter_code + str(pos_in_pdb_seq)
            )


def get_3D_structure_from_pdb_id(pdb_id):
    parser = PDBParser()
    path = "../data/pdb_files/"
    if not os.path.isfile(path):
        # print("Download the pdb file associated")
        url = "https://files.rcsb.org/download/" + pdb_id + ".pdb"
        r = requests.get(url, allow_redirects=True)
        open(path + pdb_id + ".pdb", "wb").write(r.content)
    structure = parser.get_structure(pdb_id, path + pdb_id + ".pdb")
    return structure


def convert_number_to_colors(scores, decalage=0):
    print("Je décalle les couleurs de", decalage)
    print("len scores :", len(scores))
    all_couleur = []
    for red_comp in scores[decalage:]:
        couleur = "%02x%02x%02x" % (255, 255 - red_comp, 255 - red_comp)
        couleur = couleur.upper()
        couleur = "0x" + couleur
        all_couleur.append(couleur)
    # all_couleur = all_couleur + ["0xFFFFFF"] * 255
    return all_couleur


def delete_res_outside_sequence(chain, max_len):
    for res in chain.get_unpacked_list():
        if res.get_id()[1] > max_len:
            chain.__delitem__(res.id)
        elif res.get_resname() not in list_3_letter_code:
            chain.__delitem__(res.id)


""" Get catalytic residues indices from the M-CSA database for pdb sequence or uniprot sequence
Args :
    source : pdb or uniprot
Return : 
    indices_catalytic_res : indice of the catalytic residues
    indice_catalytic_res 
    id_of_the_sequence : the pdb or uniprot id of the sequence
    chain : None for uniprot and letter of the chain for pdb
    assembly_number :  None for uniprot and one number for pdb
"""


def get_catalytic_res_from_sequence(sequence, source):
    df_infos_MCSA = pd.read_json(
        "../data/datasets/catalytic_site/M-CSA_infos_for_3D_representations_V2.json"
    )
    list_match = []
    for ind, row in df_infos_MCSA.iterrows():
        sequence_db = row["sequence_uniprot"]
        if sequence[:1024] == sequence_db[:1024]:
            list_match.append(row)

    if len(list_match) == 1:
        row = list_match[0]
        if source == "pdb":
            indices_catalytic_res = row["resid_catalytic_pdb"]
            id_of_the_sequence = row["id_pdb"]
            chain_pdb = row["chain_pdb"]
            assembly_pdb = row["assembly_pdb"]
        elif source == "uniprot":
            indices_catalytic_res = row["resid_catalytic_uniprot"]
            id_of_the_sequence = row["id_uniprot"]
            chain_pdb = None
            assembly_pdb = None
        else:
            raise RuntimeError("Source unknown")
        binding_site_uniprot = row["resid_binding_uniprot"]
    else:
        print("list_match :", list_match)
        raise RuntimeError(
            "We have no corresponding sequence in our database or we have more than one sequence that correspond so we can't decide."
        )

    return (
        indices_catalytic_res,
        id_of_the_sequence,
        chain_pdb,
        assembly_pdb,
        binding_site_uniprot,
    )


def show_3D_structure(
    sequence,
    scores,
    pdb_id,
    chain_pdb,
    assembly_pdb,
    indices_catalytic_res=None,
    show_catalytic_res=False,
    decalage_uniprot_pdb=0,
):
    colors_scores = get_color_for_viz(scores, method="first_percentile_colorize")
    structure = get_3D_structure_from_pdb_id(pdb_id)
    print("Enzyme name :", structure.header["compound"][str(assembly_pdb)]["molecule"])
    chain_selected = structure[assembly_pdb - 1][chain_pdb]
    delete_res_outside_sequence(chain_selected, len(sequence))
    view = nv.show_biopython(chain_selected)
    all_couleur = convert_number_to_colors(colors_scores, decalage=decalage_uniprot_pdb)
    set_color_by_residue(view, all_couleur)

    if show_catalytic_res:
        highlight_catalytic_on_3D_structure(
            view,
            chain_selected,
            indices_catalytic_res,
            sequence,
        )
    return view
