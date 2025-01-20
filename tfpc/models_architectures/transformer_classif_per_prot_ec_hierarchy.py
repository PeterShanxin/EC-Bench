"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import torch
import time


class TransformerClassifPerProtECHierarchy(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(
        self,
        transformer_embedder,
        num_classes,
        dropout,
        train_only_last_layer,
        dico_vocab_labels,
    ):
        super(TransformerClassifPerProtECHierarchy, self).__init__()
        print("dico_vocab_labels:", dico_vocab_labels)
        self.dico_vocab_labels = dico_vocab_labels
        self.transformer_embedder = transformer_embedder
        self.train_only_last_layer = train_only_last_layer
        if self.train_only_last_layer:
            self.transformer_embedder = self.transformer_embedder.eval()
            for param in self.transformer_embedder.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = self.transformer_embedder.d_model
        self.norm1 = LayerNorm(num_classes)

        vocab_ec_at_each_levels = [
            self.get_class_at_level(dico_vocab_labels, lvl) for lvl in range(1, 5)
        ]
        self.vocab_ec_at_each_levels = {
            ind + 1: {vocab_at_one_level[k]: k for k in range(len(vocab_at_one_level))}
            for ind, vocab_at_one_level in enumerate(vocab_ec_at_each_levels)
        }
        tensor_weights = dict()
        for lvl in range(1, 5):
            nb_ec_at_this_levels = len(self.vocab_ec_at_each_levels[lvl])
            weight_at_one_lvl = torch.rand(nb_ec_at_this_levels, 256)

            tensor_weights[str(lvl)] = torch.nn.Parameter(weight_at_one_lvl)
        self.dict_of_tensor_weights = torch.nn.ParameterDict(parameters=None)
        self.dict_of_tensor_weights.update(tensor_weights)
        print(self.dict_of_tensor_weights.keys())
        self.bias_ec_class = torch.nn.Parameter(
            torch.rand(1, len(self.vocab_ec_at_each_levels[4]))
        )

    def get_class_at_level(self, dico_vocab_labels, level):
        all_ec_class = list(dico_vocab_labels.keys())
        all_ec_class = list(
            set([self.get_one_ec_at_lvl(ec, level) for ec in all_ec_class])
        )
        return all_ec_class

    def get_one_ec_at_lvl(self, complete_ec, level):
        ec_at_level = ".".join(complete_ec.split(".")[:level])
        return ec_at_level

    def construct_ec_embeddings(self, dico_vocab_labels, vocab_ec_at_each_levels):
        embedding_ec = None
        for ec, _ in dico_vocab_labels.items():
            embedding_one_ec = None
            for level in range(1, 5):
                ec_at_this_level = self.get_one_ec_at_lvl(ec, level)
                indice_in_vocab = vocab_ec_at_each_levels[level][ec_at_this_level]
                tensor_to_concat = self.dict_of_tensor_weights[level][indice_in_vocab]
                if embedding_one_ec is None:
                    embedding_one_ec = tensor_to_concat
                else:
                    embedding_one_ec = torch.cat(
                        (
                            embedding_one_ec,
                            tensor_to_concat,
                        )
                    )

            embedding_one_ec = embedding_one_ec.unsqueeze(0)
            if embedding_ec is None:
                embedding_ec = embedding_one_ec
            else:
                embedding_ec = torch.cat((embedding_ec, embedding_one_ec))
        return embedding_ec

    def construct_ec_embeddings_V2(self, dico_vocab_labels, vocab_ec_at_each_levels):
        all_blocks = []
        for level in range(1, 5):
            list_tensor_at_this_level = []
            for ec, _ in dico_vocab_labels.items():
                ec_at_this_level = self.get_one_ec_at_lvl(ec, level)
                indice_in_vocab = vocab_ec_at_each_levels[level][ec_at_this_level]
                tensor_to_concat = self.dict_of_tensor_weights[str(level)][
                    indice_in_vocab
                ]
                tensor_to_concat = tensor_to_concat.unsqueeze(0)
                list_tensor_at_this_level.append(tensor_to_concat)

            embedding_at_this_level = torch.cat(list_tensor_at_this_level)
            all_blocks.append(embedding_at_this_level.T)

        embedding_ec = torch.cat(all_blocks)
        return embedding_ec

    def forward(self, src):
        start_time = time.time()
        ec_class_weight_matrice = self.construct_ec_embeddings_V2(
            self.dico_vocab_labels, self.vocab_ec_at_each_levels
        )
        # ec_class_weight_matrice = torch.transpose(ec_class_weight_matrice, 0, 1)
        # print("Time to compute ec matrix:", time.time() - start_time)
        # print("Start embedding class 1.14.16.4:", ec_class_weight_matrice[:25, 0])
        # print("Start embedding class 1.6.5.11:", ec_class_weight_matrice[:25, 3])
        # print("End embedding class 1.14.16.4:", ec_class_weight_matrice[1000:1024, 0])
        # print("End embedding class 1.6.5.11:", ec_class_weight_matrice[1000:1024, 3])
        # print("self.list_of_tensor_weights[1] :", self.dict_of_tensor_weights["1"])
        # print(
        #     "Grad self.list_of_tensor_weights[1] :",
        #     self.dict_of_tensor_weights["1"].grad,
        # )
        start_time = time.time()
        output = self.transformer_embedder(src)
        # print("Time to embed sequence:", time.time() - start_time)
        output = self.dropout(output[:, 0])
        # print("shape self.ec_class_weight_matrice:", self.ec_class_weight_matrice.shape)
        # print("shape output:", output.shape)
        # print("shape self.bias_ec_class:", self.bias_ec_class.shape)
        output = output @ ec_class_weight_matrice
        # print("output shape after:", output.shape)
        output += self.bias_ec_class
        # print("output shape after 2:", output.shape)
        output = self.norm1(output)
        return output
