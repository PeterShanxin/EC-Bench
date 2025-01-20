"""
This module create ProtBertProjectGOPoincareBall
"""
# pylint: disable=W0223
from curses import nonl
import logging
import torch.nn as nn
import torch
import geoopt
import numpy as np
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from tqdm import tqdm
import time
import torch
from matplotlib import pyplot as plt
import os

os.environ[
    "CUDA_PATH"
] = "/srv/soft/spack/opt/spack/linux-debian10-x86_64/gcc-8.3.0/cuda-11.2.2-pqspbantqfbi33dvnfdsccqba6kn6mg2/"
# nvcc --help
# module availmodule load spack/cuda/11.2.2
from pykeops.torch import LazyTensor
import pickle as pkl


class TransformerGOBaseline(nn.Module):
    """
    This class define a Transformer model that project each amino acid to the poincare ball and that classify each AA to a GO term and the pool to have all the GO for one protein.
    """

    def __init__(self, transformer_embedder, dico_vocab_labels):
        super(TransformerGOBaseline, self).__init__()
        self.transformer_embedder = transformer_embedder
        self.dico_vocab_labels = dico_vocab_labels
        self.nb_GO_class = len(self.dico_vocab_labels)
        self.project_to_GO = nn.Linear(1024, self.nb_GO_class)

    def forward(self, src):
        euclidian_embed_AA = self.transformer_embedder(src)
        CLS_embedding = euclidian_embed_AA[:, 0]
        weight_each_class = self.project_to_GO(CLS_embedding)
        proba_GO_term = torch.sigmoid(weight_each_class)
        return proba_GO_term
