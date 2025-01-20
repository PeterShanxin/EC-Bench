"""
This module define WeightModifier
"""
import logging
import torch
import geoopt
from utils.utils import function_not_implemented
from losses.lm_cross_entropy import LMCrossEntropy
from losses.cross_entropy_per_prot_weighted import CrossEntropyPerProtWeighted
from losses.mse_structure import MSEStructure
from losses.label_smothing import LabelSmoothLoss
from losses.cross_entropy_per_token import CrossEntropyPerToken
from losses.cross_entropy_contact_pred import CrossEntropyContactPred
from losses.l1_loss_with_min_sum_norm import L1LossMinSumNorm
from losses.l2_loss_with_min_sum_norm import L2LossMinSumNorm
from losses.l1_loss_with_softmax import L1LossSoftmax
from losses.sum_loss_just_weights import SumLossJustWeights
from losses.squared_sum_loss_just_weights import SquaredSumLossJustWeights
from losses.squared_sum_loss_just_exp_weights import SquaredSumLossJustExpWeights
from losses.sum_loss_just_exp_weights import SumLossJustExpWeights
from losses.BCE_with_ignore_index import BCEWithIgnoreIndex
from losses.BCE_with_ignore_index_with_weights import BCEWithIgnoreIndexWithWeights


class WeightModifier:
    """
    This class manage the model weight modification durring training with the correct loss
    """

    def __init__(self, config, all_models, datasets_manager):
        self.device = config.device
        self.losses = []
        self.optimizers = []
        self.lr_schedulers = []
        self.nb_step_i = []
        self.all_nb_steps_to_complete_theorical_batch = []
        for index, task_param in enumerate(config.tasks):
            loss = self.create_loss(task_param.loss, datasets_manager, index)
            # loss = loss.to(config.device)
            # loss = torch.nn.DataParallel(loss)
            self.losses.append(loss)

            optimizer = self.create_optimizer(task_param.optimizer, all_models[index])
            self.optimizers.append(optimizer)

            scheduler = (  # pylint: disable=assignment-from-none
                self.create_lr_scheduler(task_param.lr_scheduler, optimizer)
            )
            self.lr_schedulers.append(scheduler)

            nb_step = int(task_param.theorical_batch_size / task_param.batch_size)
            self.all_nb_steps_to_complete_theorical_batch.append(nb_step)
            logging.info(
                "For the task %s we log each %d batch",
                task_param.unique_task_name,
                nb_step,
            )

            self.nb_step_i.append(0)

    def create_loss(self, loss_param, datasets_manager, index):
        if loss_param["name"] == "CrossEntropy":
            if loss_param["param"] == "weighted":
                logging.info("Je charge un crossEntropy loss weighted")
                weights = datasets_manager.get_weight(index)
                weights = weights.cpu()
                loss = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                logging.info("Je charge un crossEntropy loss sans option")
                loss = torch.nn.CrossEntropyLoss()
        elif loss_param["name"] == "BCEWithIgnoreIndexWithWeights":
            logging.info("Je charge un BCEWithIgnoreIndex loss sans option")
            weight_go_class = datasets_manager.get_GO_weights(index)
            weight_go_class = weight_go_class.cpu()
            loss = BCEWithIgnoreIndexWithWeights(weight_go_class=weight_go_class)
        elif loss_param["name"] == "BCEWithIgnoreIndex":
            logging.info("Je charge un BCEWithIgnoreIndex loss sans option")
            loss = BCEWithIgnoreIndex()
        elif loss_param["name"] == "CrossEntropyWeighted":
            logging.info("Je charge un crossEntropy loss sans option")
            loss = CrossEntropyPerProtWeighted()
        elif loss_param["name"] == "L1LossNormMinSum":
            loss = L1LossMinSumNorm(datasets_manager.get_nb_labels(index))
        elif loss_param["name"] == "L2LossNormMinSum":
            loss = L2LossMinSumNorm(datasets_manager.get_nb_labels(index))
        elif loss_param["name"] == "L1LossSoftmax":
            loss = L1LossSoftmax(datasets_manager.get_nb_labels(index))
        elif loss_param["name"] == "LabelSmoothing":
            loss = LabelSmoothLoss(smoothing=loss_param["param"])
        elif loss_param["name"] == "SumLossJustWeights":
            loss = SumLossJustWeights()
        elif loss_param["name"] == "SquaredSumLossJustWeights":
            loss = SquaredSumLossJustWeights()
        elif loss_param["name"] == "SumLossJustExpWeights":
            loss = SumLossJustExpWeights()
        elif loss_param["name"] == "SquaredSumLossJustExpWeights":
            loss = SquaredSumLossJustExpWeights()
        elif loss_param["name"] == "LM_CrossEntropy":
            if loss_param["param"] == "weighted":
                logging.info("Je charge un crossEntropy loss weighted")
                weights = datasets_manager.get_weight(index)
                weights = weights.cpu()
                loss = LMCrossEntropy(weight=weights)
            else:
                logging.info("Je charge un crossEntropy loss sans option")
                loss = LMCrossEntropy()
        elif loss_param["name"] == "CrossEntropy_token_pred":
            if loss_param["param"] == "weighted":
                logging.info("Je charge un crossEntropy per token loss weighted")
                weights = datasets_manager.get_weight(index)
                weights = weights.cpu()
                loss = CrossEntropyPerToken(weight=weights)
            else:
                logging.info("Je charge un crossEntropy per token loss sans option")
                loss = ()
        elif loss_param["name"] == "CrossEntropy_contact_pred":
            loss = CrossEntropyContactPred()
        elif loss_param["name"] == "MSELoss":
            loss = torch.nn.MSELoss()
        elif loss_param["name"] == "MSE_structure":
            loss = MSEStructure()
        else:
            function_not_implemented()

        loss = loss.cpu()
        return loss

    def create_optimizer(self, optimizer_param, model):
        if optimizer_param["name"] == "Adam":
            if "weight_decay" in optimizer_param.keys():
                return torch.optim.Adam(
                    model.parameters(),
                    lr=optimizer_param["lr"],
                    weight_decay=optimizer_param["weight_decay"],
                )
            else:
                return torch.optim.Adam(model.parameters(), lr=optimizer_param["lr"])
        elif optimizer_param["name"] == "SGD":
            if hasattr(optimizer_param, "weight_decay"):
                return torch.optim.SGD(
                    model.parameters(),
                    lr=optimizer_param["lr"],
                    weight_decay=optimizer_param["weight_decay"],
                )
            else:
                return torch.optim.SGD(model.parameters(), lr=optimizer_param["lr"])
        elif optimizer_param["name"] == "RiemannianAdam":
            # Parameter:
            # -> stabilize:
            # build RADAM optimizer and specify the embeddings as parameters.
            # note that the RADAM can also optimize parameters which are not attached to a
            # manifold. then it just behaves like the usual ADAM for the Euclidean vector
            # space. we stabilize the embedding every 1 steps, which orthogonally projects
            # the embedding points onto the manifold's surface after the gradient-updates to
            # ensure that they lie precisely on the surface of the manifold. this is needed
            # because the parameters may get slightly off the manifold's surface for
            # numerical reasons. Not stabilizing may introduce small errors that accumulate
            # over time.
            # for name, param in model.named_parameters():
            #     if isinstance(param, geoopt.tensor.ManifoldParameter):
            #         print("name:", name)
            #         print(type(param))
            #         print(param)

            return geoopt.optim.RiemannianAdam(
                model.parameters(), lr=optimizer_param["lr"], stabilize=1
            )
        else:
            function_not_implemented()

    def create_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler["name"] == "no":
            return None
        elif lr_scheduler["name"] == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, *lr_scheduler["params"]
            )
        elif lr_scheduler["name"] == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer, *lr_scheduler["params"])
        else:
            raise RuntimeError("This lr scheduler name is uknown")

    def one_training_batch(self, ind_task, output, target):
        loss = self.calc_loss(ind_task, output, target)
        if torch.cuda.is_available():
            # Because we use a dataparallel wraper
            loss = loss.sum()
        loss.backward()
        self.nb_step_i[ind_task] += 1
        if (
            self.nb_step_i[ind_task]
            % self.all_nb_steps_to_complete_theorical_batch[ind_task]
            == 0
        ):
            logging.debug("Je mets à jours les poids du réseaux")
            self.optimizers[ind_task].step()
            self.optimizers[ind_task].zero_grad()
        return loss.cpu().detach().item()

    def one_eval_batch(self, ind_task, output, target):
        loss = self.calc_loss(ind_task, output, target)
        if torch.cuda.is_available():
            # Because we use a dataparallel wraper
            loss = loss.sum()
        return loss.cpu().detach().item()

    def calc_loss(self, ind_task, output, target):
        output = output.cpu()
        loss = self.losses[ind_task](output, target)
        loss = loss / self.all_nb_steps_to_complete_theorical_batch[ind_task]
        return loss

    def one_epoch_finish(self, ind_task, mean_metric_epoch):
        # On refait un optimizer step car si ce n'est pas un multiple comple on veut
        # quand même prendre en compte les derniers exemples
        # Par contre cela leurs donne un peu plus d'importance
        logging.debug(
            "On refait un optimizer step car si ce n'est pas un multiple comple on veut \
                quand même prendre en compte les derniers exemples, par contre cela leurs \
                    donne un peu plus d'importance"
        )
        self.optimizers[ind_task].step()
        self.optimizers[ind_task].zero_grad()

        if self.lr_schedulers[ind_task] is not None:
            logging.debug("Il y a un lr scheduler")
            logging.debug(str(type(self.lr_schedulers[ind_task])))
            if isinstance(
                self.lr_schedulers[ind_task], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                logging.debug("Ce lr scheduler est un ReduceLROnPlateau")
                self.lr_schedulers[ind_task].step(mean_metric_epoch)
            if isinstance(
                self.lr_schedulers[ind_task], torch.optim.lr_scheduler.StepLR
            ):
                logging.debug("Ce lr scheduler est un StepLR")
                self.lr_schedulers[ind_task].step()
            else:
                raise RuntimeError("This lr scheduler name is uknown")
