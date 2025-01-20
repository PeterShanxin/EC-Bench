from collections import deque, Counter
import math

BIOLOGICAL_PROCESS = "GO:0008150"
MOLECULAR_FUNCTION = "GO:0003674"
CELLULAR_COMPONENT = "GO:0005575"
FUNC_DICT = {
    "cc": CELLULAR_COMPONENT,
    "mf": MOLECULAR_FUNCTION,
    "bp": BIOLOGICAL_PROCESS,
}

NAMESPACES = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process",
}

EXP_CODES = set(
    [
        "EXP",
        "IDA",
        "IPI",
        "IMP",
        "IGI",
        "IEP",
        "TAS",
        "IC",
        "HTP",
        "HDA",
        "HMP",
        "HGI",
        "HEP",
    ]
)


def is_exp_code(code):
    return code in EXP_CODES


class Ontology(object):
    def __init__(self, filename, parse_relationship=True):
        self.parse_relationship = parse_relationship
        self.ont = self.load(filename)

    def delete_list_GO_terms(self, list_GO_terms):
        for go_term in list_GO_terms:
            del self.ont[go_term]
        for GO_term_id, value in self.ont.items():
            self.ont[GO_term_id]["children"] = list(
                set(value["children"]) - set(list_GO_terms)
            )

    def filter_one_namespace(self, namespace):
        other_namespaces = []
        for GO_term, value in self.ont.items():
            if value["namespace"] != namespace:
                other_namespaces.append(GO_term)
        self.delete_list_GO_terms(other_namespaces)

    def delete_obsolete_terms(self):
        # Delete obsolete terms
        all_obsolete = []
        for GO_term_id, value in self.ont.items():
            if value["is_obsolete"]:
                all_obsolete.append(GO_term_id)
        self.delete_list_GO_terms(all_obsolete)

    def delete_alternative_terms(self):
        all_alternative_id = []

        for GO_term_id, value in self.ont.items():
            if GO_term_id != value["id"]:
                all_alternative_id.append(GO_term_id)

        all_alternative_id = list(set(all_alternative_id))
        self.delete_list_GO_terms(all_alternative_id)

    # def get_networkx_graph(self):
    #     DG = nx.DiGraph()
    #     DG.add_nodes_from(list(self.ont.keys()))
    #     for GO_term, value in tqdm(self.ont.items()):
    #         all_is_a_and_part_of = value["is_a"] + value["part_of"]
    #         for parent_GO_term in all_is_a_and_part_of:
    #             DG.add_edge(GO_term, parent_GO_term)
    #     return DG

    def get_all_edges(self):
        list_edges = []
        for GO_term, value in tqdm(self.ont.items()):
            all_is_a_and_part_of = value["is_a"] + value["part_of"]
            for parent_GO_term in all_is_a_and_part_of:
                edge = (parent_GO_term, GO_term)
                list_edges.append(edge)
        return list_edges

    def get_prop_existing_edge(self):
        """
        Estimate the proportion of positive edges in the gene ontology from all possibles edges
        Can be usefull to try to estimate the performance of an edge predictors that model the Ontology
        """
        list_edges = self.get_all_edges()
        nb_positive_edges = len(list_edges)
        nb_all_possible_edges = len(self.ont) ** 2  # With a fully connected graph
        return nb_positive_edges / nb_all_possible_edges

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    # def calculate_ia(self, list_dataframe: List[pd.DataFrame], colname_prop_GO):
    #     annotations = {}
    #     for dataframe in list_dataframe:
    #         for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
    #             for GO_term in row[colname_prop_GO]:
    #                 if GO_term in annotations.keys():
    #                     annotations[GO_term].add(index)
    #                 else:
    #                     annotations[GO_term] = set([index])
    #     ia = {}
    #     for go_id in tqdm(self.ont.keys()):
    #         if go_id not in annotations.keys():
    #             """
    #             We can also just have no value for this go_id but we need one to compute the metric if the model
    #             predict all GO_term even the ones that are not present in the training set
    #             """
    #             ia[go_id] = math.inf
    #             continue
    #         term_count = len(annotations[go_id]) + 1
    #         parents = self.get_parents(go_id)
    #         if len(parents) == 0:
    #             ia[go_id] = 0.0
    #             # nb_annot_all_parents = term_count
    #         else:
    #             nb_annot_all_parents = (
    #                 self.get_nb_annotations_all_GO_id_at_the_same_time(
    #                     parents, annotations
    #                 )
    #             ) + 1
    #             ia[go_id] = math.log(nb_annot_all_parents / term_count, 2)
    #         assert ia[go_id] >= 0.0
    #     return ia

    def get_nb_annotations_all_GO_id_at_the_same_time(self, list_go_id, annotations):
        intersection = None
        for go_id in list_go_id:
            id_annot = annotations[go_id]
            if intersection is None:
                intersection = set(id_annot)
            else:
                intersection = intersection.intersection(set(id_annot))
        if intersection is None:
            raise RuntimeError("No annotations found")
        return len(intersection)

    def load(self, filename):
        ont = dict()
        list_of_informations = [
            "is_a",
            "part_of",
            "regulates",
            "alt_ids",
            "negatively_regulates",
            "positively_regulates",
            "happens_during",
            "has_part",
            "occurs_in",
            "ends_during",
        ]
        obj = None
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = dict()
                    for info in list_of_informations:
                        obj[info] = []
                    obj["is_obsolete"] = False
                    continue
                elif line == "[Typedef]":
                    if obj is not None:
                        ont[obj["id"]] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    after_double_dot = line.split(": ")
                    if after_double_dot[0] == "id":
                        obj["id"] = after_double_dot[1]
                    elif after_double_dot[0] == "alt_id":
                        obj["alt_ids"].append(after_double_dot[1])
                    elif after_double_dot[0] == "namespace":
                        obj["namespace"] = after_double_dot[1]
                    elif after_double_dot[0] == "is_a":
                        obj["is_a"].append(
                            after_double_dot[1].split(" ! ")[0].split("{")[0].strip()
                        )
                    elif (
                        after_double_dot[0] == "relationship"
                        and self.parse_relationship
                    ):
                        it = after_double_dot[1].split()
                        obj[it[0]].append(it[1])
                    elif after_double_dot[0] == "name":
                        obj["name"] = after_double_dot[1]
                    elif (
                        after_double_dot[0] == "is_obsolete"
                        and after_double_dot[1] == "true"
                    ):
                        obj["is_obsolete"] = True
            if obj is not None:
                ont[obj["id"]] = obj

        # Delete obsolete terms
        # for term_id in list(ont.keys()):
        #     for t_id in ont[term_id]["alt_ids"]:
        #         ont[t_id] = ont[term_id]
        #     if ont[term_id]["is_obsolete"]:
        #         del ont[term_id]

        # Define the children
        for term_id, val in ont.items():
            if "children" not in val:
                val["children"] = set()
            for p_id in val["is_a"]:
                if p_id in ont:
                    if "children" not in ont[p_id]:
                        ont[p_id]["children"] = set()
                    ont[p_id]["children"].add(term_id)
            for p_id in val["part_of"]:
                if p_id in ont:
                    if "children" not in ont[p_id]:
                        ont[p_id]["children"] = set()
                    ont[p_id]["children"].add(term_id)

        # Create entry for alternative indice
        # appended_terms = {}
        # for term_id, val in ont.items():
        #     for alt_GO_term in val["alt_ids"]:
        #         appended_terms[alt_GO_term] = val
        # ont.update(appended_terms)
        return ont

    def is_leaf(self, term_id):
        return len(self.ont[term_id]["children"]) == 0

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]["is_a"] + self.ont[t_id]["part_of"]:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set - set([term_id])

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]["is_a"] + self.ont[term_id]["part_of"]:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj["namespace"] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]["namespace"]

    def get_descendents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]["children"]:
                    q.append(ch_id)
        return term_set - set([term_id])
