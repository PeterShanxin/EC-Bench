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


class TransformerProjectGOPoincareBallV2(nn.Module):
    """
    This class define a Transformer model that project each amino acid to the poincare ball and that classify each AA to a GO term and the pool to have all the GO for one protein.
    """

    def __init__(self, transformer_embedder, dico_vocab_labels):
        super(TransformerProjectGOPoincareBallV2, self).__init__()
        self.transformer_embedder = transformer_embedder
        self.dico_vocab_labels = dico_vocab_labels
        self.nb_GO_class = len(self.dico_vocab_labels)
        self.transformer_embedder.eval()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        GOonly_200dim = torch.load(
            "data/GO_embedings/GOonly_200dim.pth", map_location=self.device
        )
        dico_embedings = {
            go_name: embeding
            for go_name, embeding in zip(
                GOonly_200dim["objects"], GOonly_200dim["embeddings"]
            )
        }
        print()
        self.order_embedings = torch.cat(
            [
                dico_embedings[go_name].unsqueeze(0)
                for go_name in dico_vocab_labels.keys()
            ]
        )
        print("Shape :", self.order_embedings.shape)

        # self.reduce_seq_dim = torch.nn.MaxPool2d(
        #     (16, 1), stride=(8, 1)
        # )  # -> Indépendant pour chauqe embedding dim et donc ça donne peut etre un vecteur sans sens...
        # Reduce with one attention layer ?
        self.attn_reducer = torch.nn.MultiheadAttention(
            embed_dim=1024, num_heads=8, dropout=0.1, batch_first=True
        )
        self.vector_reducer = torch.nn.Parameter(torch.FloatTensor(10, 1024))
        print("self.vector_reducer grad:", self.vector_reducer.requires_grad)
        self.ball = create_ball(c=1.0)  # geoopt.PoincareBallExact()
        self.mobius_linear1 = MobiusLinear(
            1024, 200, ball=self.ball, nonlin=torch.nn.ReLU()
        )
        self.mobius_linear2 = MobiusLinear(
            200, 200, ball=self.ball, nonlin=torch.nn.ReLU()
        )
        self.mobius_linear3 = MobiusLinear(200, 200, ball=self.ball, nonlin=None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TODO: init self.vector_reducer parameter
        # init.uniform_(self.bias, -bound, bound)
        torch.nn.init.xavier_normal_(self.vector_reducer)

    def dist(self, u, v, eps=10e-7):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    def distance_all_AA_all_GO_term(self, all_AA_embedings):
        bs, len_max, embed_size = all_AA_embedings.shape
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")
        order_embedings = self.order_embedings.to(device)
        nb_GO_term = len(order_embedings)
        all_AA_embedings = all_AA_embedings.reshape(bs * len_max, embed_size)
        list_dist = []
        for AA_embedding in all_AA_embedings:
            dists = self.ball.dist(AA_embedding, order_embedings)
            dists = dists.unsqueeze(0)
            list_dist.append(dists)

        all_dists = torch.cat(list_dist)
        all_dists = all_dists.reshape(bs, len_max, nb_GO_term)
        return all_dists

    def forward(self, src):
        key_padding_mask = src == self.transformer_embedder.pad_index
        with torch.no_grad():
            euclidian_embed_AA = self.transformer_embedder(src)
        # euclidian_embed_AA = self.reduce_seq_dim(euclidian_embed_AA)

        batch_size = euclidian_embed_AA.shape[0]
        querry_vec = self.vector_reducer.expand((batch_size, -1, -1))
        euclidian_embed_AA = self.attn_reducer(  # querry_vec +self.attn_reducer(  ? veut-on faire une skip connection ici ?
            querry_vec,
            euclidian_embed_AA,
            euclidian_embed_AA,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[
            0
        ]

        poincare_embed_AA = self.ball.expmap0(euclidian_embed_AA)
        poincare_embed_AA = self.mobius_linear1(poincare_embed_AA)
        poincare_embed_AA = self.mobius_linear2(poincare_embed_AA)
        poincare_embed_AA = self.mobius_linear3(poincare_embed_AA)
        all_dists = self.distance_all_AA_all_GO_term(poincare_embed_AA)

        all_dists = torch.min(all_dists, axis=1).values
        proba_GO_term = torch.exp(-all_dists)

        return proba_GO_term


class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))

        if self.bias is not None:
            self.bias.zero_()


def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output


# package.nn.modules.py
def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.
    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.
    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float
    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)  # geoopt.PoincareBallExact(c)  #
    # else trust input
    return ball
