"""
This module create ProtBertProjectGOPoincareBall
"""
# pylint: disable=W0223
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


class TransformerProjectGOPoincareBall(nn.Module):
    """
    This class define a Transformer model that project each amino acid to the poincare ball and that classify each AA to a GO term and the pool to have all the GO for one protein.
    """

    def __init__(self, transformer_embedder, dico_vocab_labels):
        super(TransformerProjectGOPoincareBall, self).__init__()
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
        self.order_embedings = torch.cat(
            [
                dico_embedings[go_name].unsqueeze(0)
                for go_name in dico_vocab_labels.keys()
            ]
        )
        print("Shape :", self.order_embedings.shape)

        self.ball = create_ball(c=1.0)  # geoopt.PoincareBallExact()
        self.mobius_linear = MobiusLinear(32, 200, ball=self.ball)
        print(self.mobius_linear.weight)
        print(type(self.mobius_linear.weight))
        # TMP
        # size_vocab = len(dico_vocab_labels)
        # self.linear_proj = torch.nn.Linear(1024, size_vocab)
        # FIN TMP

        # Test version convolution
        self.output_reduce_dim = torch.nn.Conv1d(
            in_channels=1024, out_channels=32, kernel_size=7, stride=1, padding="same"
        )
        self.max_pool = torch.nn.MaxPool1d(64, stride=32)
        self.conv_output_1 = torch.nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=7, stride=4, padding=6
        )
        self.relu = nn.ReLU()
        self.conv_output_2 = torch.nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=7, stride=4, padding=6
        )
        self.conv_output_3 = torch.nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=7, stride=4, padding=6
        )

        self.dist_to_score = torch.nn.Linear(self.nb_GO_class, self.nb_GO_class)

        # min_dist_between_go_term = 4.7
        # self.K = -np.log(0.5) / np.power(min_dist_between_go_term, 2)
        # print("Self.K", self.K)
        # path_ancestor_matrix = config.path_folder_json + "/ancestor_matrix.pkl"
        # if not os.path.exists(path_ancestor_matrix):

        #     fichier = open(path_ancestor_matrix, "wb")
        #     pkl.dump(self.ancestor_matrix, fichier)
        # else:
        #     fichier = open(path_ancestor_matrix, "rb")
        #     self.ancestor_matrix = pkl.load(fichier)
        # self.path_dico_leaves_childs = "data/GO_embedings/dico_leaves_childs.pkl"
        # if not os.path.exists(self.path_dico_leaves_childs):
        #     self.dico_leaves_childs = self.create_dico_leaves_childs()
        # else:
        #     fichier = open(self.path_dico_leaves_childs, "rb")
        #     self.dico_leaves_childs = pkl.load(fichier)

        # self.dico_ancestors = self.create_dico_ancestors()
        # self.ancestor_matrix = self.create_ancestor_matrix()

    # def create_dico_ancestors(self):
    #     dico_ancestors = dict()
    #     for go_term, leaves_child in self.dico_leaves_childs.items():
    #         for leaf in leaves_child:
    #             dico_ancestors[leaf] = go_term

    #     return dico_ancestors

    # def get_ancestor(self, go_term, godag):
    #     gosubdag_r0 = GoSubDag([go_term], godag, prt=None)
    #     return gosubdag_r0.rcntobj.go2ancestors[go_term]

    # def create_ancestor_matrix(self):
    #     """
    #     Create a matrix with each row that represent a GO term i, and there is a one in the column j if j is an ancestor of i.
    #     The order of the GO_term is define by the order of the vocabulary
    #     """
    #     # dico_ancestor list go_term ancestor with go_term as key
    #     logging.info("Creating ancestor matrix")
    #     ancestor_matrix = None
    #     ind_current_go_term = 0
    #     for go_term in tqdm(self.dico_vocab_labels.keys()):
    #         ancestor = self.dico_ancestors[go_term]  # self.get_ancestor(go_term, godag)
    #         # ancestor = list(ancestor)
    #         ancestor = [anc for anc in ancestor if anc in self.dico_vocab_labels.keys()]
    #         tmp_row = torch.zeros((len(self.dico_vocab_labels)))
    #         ind_ancestor = [self.dico_vocab_labels[anc] for anc in ancestor]
    #         tmp_row[ind_ancestor] = 1
    #         tmp_row[ind_current_go_term] = 1
    #         tmp_row = tmp_row.unsqueeze(0)
    #         if ancestor_matrix is None:
    #             ancestor_matrix = tmp_row
    #         else:
    #             ancestor_matrix = torch.cat((ancestor_matrix, tmp_row))
    #         ind_current_go_term += 1
    #     ancestor_matrix = ancestor_matrix.bool()
    #     print("Shape ancestor_matrix :", ancestor_matrix.shape)
    #     print("dtype ancestor matrix :", ancestor_matrix.dtype)
    #     return ancestor_matrix

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

    def KMeans(self, x, K=10, Niter=1000):
        """Implements Lloyd's algorithm for the Euclidean metric."""
        N, D = x.shape  # Number of samples, dimension of the ambient space

        c = x[:K, :].clone()  # Simplistic initialization for the centroids

        x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(Niter):

            # E step: assign points to the closest cluster -------------------------
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Divide by the number of points per cluster:
            Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)

            Ncl = torch.clamp(Ncl, min=1)
            c /= Ncl  # in-place division to compute the average

        return cl, c

    def forward(self, src):
        with torch.no_grad():
            euclidian_embed_AA = self.transformer_embedder(src)
            # euclidian_embed_AA.requiere_grad = False
            bs = euclidian_embed_AA.shape[0]

            # 1- KMeans version
            # all_centers = []
            # for batch_num in range(bs):
            #     _, center = self.KMeans(euclidian_embed_AA[batch_num], K=10, Niter=10)
            #     all_centers.append(center.unsqueeze(0))
            # all_centers = torch.cat(all_centers)
            # Best Fmax dev 0.0025017209547832813, best smin dev 0.13288067382475555

            # 2- CLS version
            # all_centers = []
            # for batch_num in range(bs):
            #     all_centers.append(
            #         euclidian_embed_AA[batch_num][0].unsqueeze(0).unsqueeze(0)
            #     )
            # all_centers = torch.cat(all_centers)
            # Best result than Kmeans version

        # 3- Conv version
        # print("Shape1:", euclidian_embed_AA.shape)
        euclidian_embed_AA = euclidian_embed_AA.transpose(1, 2)
        # print("Shape2:", euclidian_embed_AA.shape)
        euclidian_embed_AA = self.output_reduce_dim(euclidian_embed_AA)
        # print("Shape3:", euclidian_embed_AA.shape)
        euclidian_embed_AA = self.relu(euclidian_embed_AA)
        # print("Shape4:", euclidian_embed_AA.shape)
        # euclidian_embed_AA = self.max_pool(euclidian_embed_AA)
        # print("Shape5:", euclidian_embed_AA.shape)
        euclidian_embed_AA = self.conv_output_1(euclidian_embed_AA)
        euclidian_embed_AA = self.relu(euclidian_embed_AA)
        euclidian_embed_AA = self.conv_output_2(euclidian_embed_AA)
        euclidian_embed_AA = self.relu(euclidian_embed_AA)
        euclidian_embed_AA = self.conv_output_3(euclidian_embed_AA)
        euclidian_embed_AA = self.relu(euclidian_embed_AA)
        all_centers = euclidian_embed_AA.transpose(1, 2)
        # print("Shape6:", all_centers.shape)
        # fmax on dev on task number 0 is 0.22357875503508992

        # 1- With Riemanian manifold
        poincare_embed_AA = self.ball.expmap0(all_centers)
        poincare_embed_AA = self.mobius_linear(poincare_embed_AA)
        # print("Shape7:", poincare_embed_AA.shape)
        all_dists = self.distance_all_AA_all_GO_term(poincare_embed_AA)
        # print("Shape8:", all_dists.shape)

        # min_dist_each_GO_term = torch.min(all_dists, axis=1).values
        # proba_GO_term = torch.exp(-self.K * torch.pow(min_dist_each_GO_term, 2))
        # I think it's not a good idea to directly to a softmax on dist, myabe it's better to do like in HyperIM, take all the dist for each class and use MLP + sigmoid to predict probability ?
        # all_dists = all_dists.float()
        # scores = self.dist_to_score(all_dists)
        # scores = self.relu(scores)
        proba_GO_term = torch.nn.functional.softmax(-all_dists, dim=2)
        # print("Shape9:", proba_GO_term.shape)
        proba_GO_term = torch.max(proba_GO_term, axis=1).values
        # print("Shape10:", proba_GO_term.shape)
        # proba_GO_term = proba_GO_term.double()

        # 2-Without Riemanian manifold
        # all_centers = all_centers.reshape(bs, -1)
        # proba_GO_term = self.linear_proj(all_centers)
        # proba_GO_term = torch.nn.functional.softmax(proba_GO_term)
        # proba_GO_term = proba_GO_term.type(torch.float64)

        # Correct proba with ancestor information
        # proba_GO_term = self.correct_proba_ancestor(proba_GO_term)

        # proba_GO_term = torch.clamp(proba_GO_term, min=0.01, max=0.99)

        return proba_GO_term

    # def correct_proba_ancestor(self, all_proba):
    #     # print("all_proba shape :", all_proba.shape)
    #     # print("Shape self.ancestor_matrix", self.ancestor_matrix.shape)
    #     # print("selection shape :", all_proba[:, self.ancestor_matrix[0]].shape)
    #     # print(
    #     #     "selection shape max:",
    #     #     torch.max(all_proba[:, self.ancestor_matrix[0]], axis=1).shape,
    #     # )
    #     new_all_proba = torch.stack(
    #         [
    #             torch.max(all_proba[:, self.ancestor_matrix.T[k]], axis=1).values
    #             for k in range(len(all_proba[0]))
    #         ]
    #     )
    #     new_all_proba = new_all_proba.T
    #     # print("Apres all_proba shape :", new_all_proba.shape)
    #     # bs, nb_go_class = all_proba.shape
    #     # print("all_proba shape :", all_proba.shape)
    #     # all_proba = all_proba.unsqueeze(-1)
    #     # all_proba = all_proba.expand(bs, nb_go_class, nb_go_class)
    #     # print("After repeat shape :", all_proba.shape)
    #     # print("Shape self.ancestor_matrix", self.ancestor_matrix.shape)
    #     # selected_ancestor = all_proba[:, self.ancestor_matrix]
    #     # print("Selected ancestor shape :", selected_ancestor.shape)
    #     # all_proba = torch.max(selected_ancestor, axis=1).values
    #     # print("After select and max :", all_proba.shape)
    #     return new_all_proba


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
