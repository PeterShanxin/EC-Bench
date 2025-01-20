"""
This module define Model_factory
"""
# Obliger d'utiliser le sys path append car le structure de chargement de model_ENCODER
#  doit etre la meme que lors de la sauvegarde dans l'ancien projet
import sys
import logging
import torch
from models_architectures.transformer_classif_per_prot import (
    TransformerClassifPerProt,
)
from models_architectures.transformer_classif_per_prot_layer_norm import (
    TransformerClassifPerProtLayerNorm,
)
from models_architectures.Transformer_project_GO_poincare_ball import (
    TransformerProjectGOPoincareBall,
)
from models_architectures.Transformer_project_GO_poincare_ball_V2 import (
    TransformerProjectGOPoincareBallV2,
)
from models_architectures.Transformer_project_GO_baseline import TransformerGOBaseline
from models_architectures.transformer_classif_per_prot_layer_norm_concat import (
    TransformerClassifPerProtLayerNormConcat,
)
from models_architectures.transformer_classif_per_prot_cnn import (
    TransformerClassifPerProtConv,
)
from models_architectures.transformer_classif_per_prot_like_ensemble import (
    TransformerClassifPerProtLikeEnsemble,
)
from models_architectures.transformer_classif_per_prot_ec_hierarchy import (
    TransformerClassifPerProtECHierarchy,
)
from models_architectures.transformer_classif_per_prot_apriori import (
    TransformerClassifPerProtApriori,
)
from models_architectures.transformer_classif_per_prot_change_all_dropout import (
    TransformerClassifPerProtChangeDropout,
)
from models_architectures.transformer_classif_per_prot_more_expressive import (
    TransformerClassifPerProtMoreExpressive,
)
from models_architectures.transformer_classif_per_prot_more_expressive_and_layer_norm import (
    TransformerClassifPerProtMoreExpressiveAndLayerNorm,
)

from models_architectures.transformer_classif_per_amino_acid import (
    TransformerClassifPerAminoAcid,
)
from models_architectures.transformer_classif_per_prot_no_cls import (
    TransformerClassifPerProtNoCls,
)
from models_architectures.transformer_classif_contact import TransformerClassifContact
from models_architectures.transformer_lm import TransformerLM
from models_architectures.transformer_regression_per_prot import (
    TransformerRegressionPerProt,
)
from models_architectures.transformer_structure_prediction import (
    TransformerStructurePrediction,
)
from models_architectures.model_ENCODER import model_ENCODER
from models_architectures.transformer_embedder import TransformerEmbedder

sys.path.append("models_architectures/")
# from model_ENCODER import EncoderModel


class ModelFactory:
    """
    This class is a factory to get model object from their name
    """

    def __init__(self, config, dataset_manager):
        self.config = config
        self.dataset_manager = dataset_manager

    def create_all_models(self):
        all_models = []
        for index, task_param in enumerate(self.config.tasks):
            model = self.create_one_model(task_param, index)
            all_models.append(model)
        return all_models

    def create_one_model(self, task_param, index) -> torch.nn.Module:
        config = self.config
        if config.starting_model.type == "from_scratch":
            if task_param.model["name"] == "New_Transformer_LM":
                param_transformer = task_param.model["params"]
                n_token = len(config.vocab)
                logging.debug("There is %s letter in the vocab", n_token)
                starting_model = model_ENCODER(n_token, *param_transformer)
                starting_model.pad_indice = config.vocab["p"]
                starting_model = torch.nn.DataParallel(starting_model)
                starting_model = TransformerEmbedder(starting_model, config.vocab["p"])
            else:
                raise RuntimeError("type starting model inconnu")
        else:
            # On load le modele
            starting_model = torch.load(
                config.root_models + config.starting_model.path_model,
                map_location=config.device,
            )

        if task_param.model["name"] == "Transformer_classif_per_prot":
            model_classif = TransformerClassifPerProt(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
            )
            logging.debug(
                "La projection du modele est %s", model_classif.classe_embedding
            )
        elif task_param.model["name"] == "Transformer_classif_per_prot_layer_norm":
            dropout = task_param.model["params"][0]
            if len(task_param.model["params"]) > 1:
                train_only_last_layer = task_param.model["params"][1]
            else:
                train_only_last_layer = False
            model_classif = TransformerClassifPerProtLayerNorm(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                dropout,
                train_only_last_layer,
            )
        elif task_param.model["name"] == "Transformer_Project_GO_PoincareBall":
            model_classif = TransformerProjectGOPoincareBall(
                starting_model, self.dataset_manager.task_vocabs[index]
            )
        elif task_param.model["name"] == "Transformer_Project_GO_PoincareBallV2":
            model_classif = TransformerProjectGOPoincareBallV2(
                starting_model, self.dataset_manager.task_vocabs[index]
            )
        elif task_param.model["name"] == "TransformerGOBaseline":
            model_classif = TransformerGOBaseline(
                starting_model, self.dataset_manager.task_vocabs[index]
            )
        elif task_param.model["name"] == "Transformer_classif_per_prot_EC_hierarchy":
            dropout = task_param.model["params"][0]
            if len(task_param.model["params"]) > 1:
                train_only_last_layer = task_param.model["params"][1]
            else:
                train_only_last_layer = False
            model_classif = TransformerClassifPerProtECHierarchy(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                dropout,
                train_only_last_layer,
                self.dataset_manager.task_vocabs[index],
            )
        elif task_param.model["name"] == "Transformer_classif_per_prot_conv":
            model_classif = TransformerClassifPerProtConv(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
                task_param.model["params"][1],
                task_param.model["params"][2],
                task_param.model["params"][3],
            )
        elif (
            task_param.model["name"] == "Transformer_classif_per_prot_layer_norm_concat"
        ):
            model_classif = TransformerClassifPerProtLayerNormConcat(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
                task_param.model["params"][1],
                task_param.model["params"][2],
                task_param.model["params"][3],
            )

        elif task_param.model["name"] == "Transformer_classif_per_prot_with_apriori":
            model_classif = TransformerClassifPerProtApriori(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
            )
        elif task_param.model["name"] == "transformer_classif_per_prot_like_ensemble":
            model_classif = TransformerClassifPerProtLikeEnsemble(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
                task_param.model["params"][1],
            )
        elif task_param.model["name"] == "Transformer_classif_per_prot_no_cls":
            model_classif = TransformerClassifPerProtNoCls(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
            )
        elif (
            task_param.model["name"] == "Transformer_LM"
            or task_param.model["name"] == "New_Transformer_LM"
        ):
            model_classif = TransformerLM(starting_model)
        elif task_param.model["name"] == "TransformerClassifPerProtChangeDropout":
            model_classif = TransformerClassifPerProtChangeDropout(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
            )
        elif task_param.model["name"] == "Transformer_classif_per_prot_more_expressive":
            model_classif = TransformerClassifPerProtMoreExpressive(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
                task_param.model["params"][1],
            )
        elif (
            task_param.model["name"]
            == "Transformer_classif_per_prot_more_expressive_and_layer_norm"
        ):
            model_classif = TransformerClassifPerProtMoreExpressiveAndLayerNorm(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
                task_param.model["params"][1],
            )
        elif task_param.model["name"] == "Transformer_classif_per_amino_acid":
            model_classif = TransformerClassifPerAminoAcid(
                starting_model,
                self.dataset_manager.get_nb_labels(index),
                task_param.model["params"][0],
            )
        elif task_param.model["name"] == "Transformer_classif_contact":
            model_classif = TransformerClassifContact(
                starting_model, task_param.model["params"][0]
            )
        elif task_param.model["name"] == "Transformer_regression_per_prot":
            model_classif = TransformerRegressionPerProt(
                starting_model, task_param.model["params"][0]
            )
        elif task_param.model["name"] == "Transformer_structure":
            model_classif = TransformerStructurePrediction(
                starting_model, task_param.model["params"][0]
            )
        else:
            raise RuntimeError("type de model inconnu")

        model_classif = model_classif.to("cpu")
        logging.info("Le model est sur %s", next(model_classif.parameters()).device)

        return model_classif