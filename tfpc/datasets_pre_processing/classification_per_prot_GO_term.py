"""
This module define a specific dataset
"""
from cProfile import label
import logging
import torch
from datasets_pre_processing.generic_dataset import GenericDataset
from goatools.obo_parser import GODag
import pickle as pkl
import os
from tqdm import tqdm
from goatools.gosubdag.gosubdag import GoSubDag


class ClassificationPerProtGOTerm(GenericDataset):
    """
    This class is a dataset where we have one label per protein/example
    """

    def __init__(
        self,
        dataframe,
        limit_size_input_prot,
        config,
        col_name_input,
        col_name_output,
        indice_label,
    ):
        super().__init__(
            dataframe, limit_size_input_prot, config, col_name_input, col_name_output
        )
        # Vocab of the model
        self.indice_label = indice_label
        self.dico_childs = self.get_dico_childs()
        self.global_class_vocab_GO_MF_2016_08 = self.get_global_vocab()
        print("Len global vocab:", len(self.global_class_vocab_GO_MF_2016_08))
        self.ancestor_matrix = self.get_ancestor_matrix()
        print("Shape self.ancestor_matrix:", self.ancestor_matrix.shape)
        self.delete_data_no_valid_annotation()

    def delete_data_no_valid_annotation(self):
        ind_to_delete = []
        for k in range(len(self.labels)):
            all_annotation_this_prot = self.labels[k]
            all_annotation_this_prot = [
                annot
                for annot in all_annotation_this_prot
                if annot in self.global_class_vocab_GO_MF_2016_08
            ]
            all_derived_GO = self.labels_encoding_for_the_model(
                all_annotation_this_prot
            )
            if len(all_derived_GO) == 0:
                ind_to_delete.append(k)
        logging.info(
            "%d proteins have no correct annotation and will be deleted, on a total of %d proteins",
            len(ind_to_delete),
            len(self.labels),
        )
        for index in sorted(ind_to_delete, reverse=True):
            del self.labels[index]
            del self.data[index]

    def get_ancestor_matrix(self):
        path_ancestor_matrix = "data/GO_embedings/ancestor_matrix_2016_08.pkl"
        if not os.path.exists(path_ancestor_matrix):
            ancestor_matrix = self.create_ancestor_matrix_with_global_vocab(
                path_ancestor_matrix
            )
        else:
            fichier = open(path_ancestor_matrix, "rb")
            ancestor_matrix = pkl.load(fichier)
        return ancestor_matrix

    def create_ancestor_matrix_with_global_vocab(self, path_ancestor_matrix):
        """
        Create a matrix with each row that represent a GO term i, and there is a one in the column j if j is an ancestor of i.
        The order of the GO_term is define by the order of the vocabulary
        """
        # dico_ancestor list go_term ancestor with go_term as key
        logging.info("Creating ancestor matrix")
        dico_ancestors = pkl.load(
            open("data/GO_embedings/dico_ancestor_2016_08_MF.pkl", "rb")
        )
        ancestor_matrix = None
        ind_current_go_term = 0
        for go_term in tqdm(self.global_class_vocab_GO_MF_2016_08):
            ancestor = dico_ancestors[go_term]  # self.get_ancestor(go_term, godag)
            # ancestor = list(ancestor)
            # ancestor = [anc for anc in ancestor if anc in self.indice_label.keys()]
            tmp_row = torch.zeros((len(self.global_class_vocab_GO_MF_2016_08)))
            ind_ancestor = [
                self.global_class_vocab_GO_MF_2016_08[anc] for anc in ancestor
            ]
            tmp_row[ind_ancestor] = 1
            tmp_row[ind_current_go_term] = 1
            tmp_row = tmp_row.unsqueeze(0)
            if ancestor_matrix is None:
                ancestor_matrix = tmp_row
            else:
                ancestor_matrix = torch.cat((ancestor_matrix, tmp_row))
            ind_current_go_term += 1
        ancestor_matrix = ancestor_matrix.bool()
        print("Shape ancestor_matrix :", ancestor_matrix.shape)
        print("dtype ancestor matrix :", ancestor_matrix.dtype)
        fichier = open(path_ancestor_matrix, "wb")
        pkl.dump(ancestor_matrix, fichier)
        return ancestor_matrix

    @staticmethod
    def get_available_GO_embeddings():
        GOonly_200dim = torch.load(
            "data/GO_embedings/GOonly_200dim.pth", map_location=torch.device("cpu")
        )
        available_GO_embeddings = GOonly_200dim["objects"]
        return available_GO_embeddings

    @staticmethod
    def get_global_vocab():
        global_list_GO_term_MF_2016_08 = pkl.load(
            open("data/GO_embedings/global_list_GO_term_MF_2016_08.pkl", "rb")
        )
        global_class_vocab_GO_MF_2016_08 = {
            GO_term: num for num, GO_term in enumerate(global_list_GO_term_MF_2016_08)
        }
        return global_class_vocab_GO_MF_2016_08

    @staticmethod
    def get_dico_childs():
        path_dico_childs = "data/GO_embedings/dico_childs.pkl"
        if not os.path.exists(path_dico_childs):
            dico_childs = ClassificationPerProtGOTerm.create_dico_childs(
                path_dico_childs
            )
        else:
            fichier = open(path_dico_childs, "rb")
            dico_childs = pkl.load(fichier)
        return dico_childs

    @staticmethod
    def get_descendent(go_term, godag):
        gosubdag_r0 = GoSubDag([go_term], godag, prt=None)
        if go_term in gosubdag_r0.rcntobj.go2descendants.keys():
            return gosubdag_r0.rcntobj.go2descendants[go_term]
        else:  # It's a leaf
            return [go_term]

    @staticmethod
    def create_dico_childs(path_dico_childs):
        global_class_vocab_GO_MF_2016_08 = (
            ClassificationPerProtGOTerm.get_global_vocab()
        )
        godag = GODag("data/GO_embedings/go_2016_08.obo")
        dico_child = {go_term: [] for go_term in global_class_vocab_GO_MF_2016_08}
        for go_term in tqdm(global_class_vocab_GO_MF_2016_08):
            for descendent in ClassificationPerProtGOTerm.get_descendent(
                go_term, godag
            ):
                dico_child[go_term].append(descendent)

        fichier = open(path_dico_childs, "wb")
        pkl.dump(dico_child, fichier)

        return dico_child

    def delete_go_term_not_in_model_vocab(self, list_go_term):
        go_term_with_embeding = []
        for go_term in list_go_term:
            if go_term in self.indice_label:
                go_term_with_embeding.append(go_term)
        return go_term_with_embeding

    def delete_go_term_not_in_global_MF(self, list_go_term):
        go_term_with_embeding = []
        for go_term in list_go_term:
            if go_term in list(self.global_class_vocab_GO_MF_2016_08.values()):
                go_term_with_embeding.append(go_term)
        return go_term_with_embeding

    @staticmethod
    def get_GO_leaves_in_MF():
        godag = GODag("data/GO_embedings/go_2016_08.obo")
        list_leaves = []
        for go_term in tqdm(godag.values(), total=44949):
            is_leaf = not go_term.children
            in_MF = go_term.namespace == "molecular_function"
            if is_leaf and in_MF:
                list_leaves.append(go_term.id)

        return list(set(list_leaves))

    @staticmethod
    def construct_label_indice(train_label, dev_label):
        # list_leaves_in_MF = ClassificationPerProtGOTerm.get_GO_leaves_in_MF()
        global_class_vocab_GO_MF_2016_08 = (
            ClassificationPerProtGOTerm.get_global_vocab()
        )
        return global_class_vocab_GO_MF_2016_08

        available_GO_embeddings = (
            ClassificationPerProtGOTerm.get_available_GO_embeddings()
        )
        list_leaves_in_MF = [
            GO_leaf
            for GO_leaf in list_leaves_in_MF
            if GO_leaf in available_GO_embeddings
        ]
        dico_labels = {GO_term: ind for ind, GO_term in enumerate(list_leaves_in_MF)}
        return dico_labels

    def sequence_encoding(self, seq):
        return [self.vocab["c"]] + [self.vocab[amine] for amine in seq][
            : self.size_limit
        ]

    def labels_encoding_for_the_model(self, labels):
        """
        self.dico_childs_which_are_leaves: key are the GO term id and values are list of child leaves of this GO term AND imself
        """
        # We add the leaf descendant of some annotated ancestors
        new_labels = []
        for one_go_term in labels:
            new_labels += self.dico_childs[
                one_go_term
            ]  # If we want only leaf: dico_childs_which_are_leaves[one_go_term]
        # We delete GO term that is not in the model vocabulary
        # print("new_labels:", new_labels)
        new_labels = self.delete_go_term_not_in_model_vocab(new_labels)
        # print("new_labels after delete:", new_labels)
        # We encode the GO id to the corresponding indice in the model vocabulary
        return list(set([self.indice_label[l] for l in new_labels]))

    def labels_encoding_for_the_metric(self, labels):
        indice_label = [
            self.global_class_vocab_GO_MF_2016_08[go_term] for go_term in labels
        ]
        new_labels = self.append_ancestor(indice_label)
        # We delete GO term that is not in the global MF vocabulary
        new_labels = self.delete_go_term_not_in_global_MF(new_labels)

        return list(set(new_labels))

    def append_ancestor(self, all_target):
        all_ancestor = []
        for target in all_target:
            all_ancestor += self.ancestor_matrix[target].nonzero().view(-1).tolist()
        return all_ancestor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def collate(self, batch):
        data = [torch.LongTensor(self.sequence_encoding(item[0])) for item in batch]
        only_leaves_labels = [
            torch.LongTensor(self.labels_encoding_for_the_model(item[1]))
            for item in batch
        ]
        all_labels = [
            torch.LongTensor(self.labels_encoding_for_the_model(item[1]))
            for item in batch
        ]
        # p permet d'encoder le pad token
        return (
            torch.nn.utils.rnn.pad_sequence(
                data, batch_first=True, padding_value=self.vocab["p"]
            ),
            (only_leaves_labels, all_labels),
        )
