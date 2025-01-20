import logging
import csv
import torch
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from scipy.stats import entropy


class GenerateOutput:
    def __init__(self, training_analyser):
        self.training_analyser = training_analyser
        self.softmax = torch.nn.Softmax()

    @torch.no_grad()
    def calc_outputs(
        self,
        ind_task,
        mask_task=False,
        which_dataset_part="valid",
        only_mask_pos=False,
        limit_output=True,
        mode_eval=True,
        extract_cls=False,
    ):
        model = self.training_analyser.models[ind_task]
        model = torch.nn.DataParallel(model)
        if mode_eval:
            model = model.eval()
        else:
            model = model.train()

        if which_dataset_part == "train":
            print("Load on train")
            dataloader = self.training_analyser.dataset_manager.get_train_dataloader(
                ind_task
            )
        elif which_dataset_part == "test":
            print("Load on test")
            dataloader = self.training_analyser.dataset_manager.get_test_dataloader(
                ind_task
            )
        elif which_dataset_part == "valid":
            print("Load on test")
            dataloader = self.training_analyser.dataset_manager.get_dev_dataloader(
                ind_task
            )

        all_outputs = []

        vocab = self.training_analyser.config.vocab
        vocab_inv = {value: key for key, value in vocab.items()}
        print(vocab_inv)
        if extract_cls:
            all_outputs = None
            sequences = []
            activation = {}

            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output

                return hook

            the_encoder = model.module.transformer_embedder.transformer_encoder
            # Pour chaque couche on met un crochet pour recuperer les attentions
            NB_LAYER = len(the_encoder.layers)

            the_encoder.layers[NB_LAYER - 1].norm2.register_forward_hook(
                get_activation("output_before_pred")
            )
        compteur = 1
        for input_batch, target_batch in dataloader:
            if compteur % 5 == 0:
                print("Avancement :", (compteur / len(dataloader) * 100, "%"))
            output_batch = model(input_batch)
            output_proba = torch.nn.functional.softmax(output_batch, dim=1)
            if mask_task:
                true_target, mask = target_batch
            rendu = 0

            if mask_task:
                for input_prot, one_output_proba, indice_mask in zip(
                    input_batch, output_proba, mask
                ):
                    if only_mask_pos:
                        sequence = self.convert_input_to_seq(
                            input_prot.tolist(), vocab_inv
                        )
                        del_padding = input_prot != vocab["p"]
                        one_output_proba = one_output_proba[del_padding]
                        complete_input = list(sequence)
                        for indice in indice_mask:
                            complete_input[indice] = vocab_inv[int(true_target[rendu])]
                            rendu += 1
                        complete_input = "".join(complete_input)
                        entrop = entropy(one_output_proba.cpu(), base=2, axis=1)
                        maxi = torch.max(one_output_proba, dim=1)[0]
                        one_entry = [sequence, entrop, maxi, complete_input]

                    all_outputs.append(one_entry)
            elif extract_cls:
                for input_prot, one_output_proba, one_target in zip(
                    input_batch, output_batch, target_batch
                ):
                    sequence = self.convert_input_to_seq(input_prot.tolist(), vocab_inv)
                    sequences.append(sequence)

                NB_ALL_BATCH = output_batch.shape[0]
                hook_value = activation["output_before_pred"].detach().cpu().numpy()
                cls_last_encoding = hook_value[-1]
                if all_outputs is None:
                    all_outputs = cls_last_encoding
                else:
                    all_outputs = np.concatenate((all_outputs, cls_last_encoding))
            else:
                for input_prot, one_output_proba, one_target in zip(
                    input_batch, output_batch, target_batch
                ):
                    predicted_class = torch.argmax(one_output_proba)
                    predicted_class = predicted_class.cpu()
                    correct = predicted_class == one_target
                    sequence = self.convert_input_to_seq(input_prot.tolist(), vocab_inv)
                    if limit_output:
                        one_entry = [sequence, correct]
                    else:
                        one_entry = [sequence, one_output_proba.tolist(), correct]
                    all_outputs.append(one_entry)

            compteur += 1

        model = model.train()
        model = model.module

        if extract_cls:
            all_outputs = [sequences, all_outputs]

        base_path_files = "data/tests_data/" + self.training_analyser.xp_name
        with open(
            base_path_files
            + "_complete_output_on_"
            + str(which_dataset_part)
            + "_limit_output_"
            + str(limit_output)
            + "_cls_out"
            + str(extract_cls)
            + ".pkl",
            "wb",
        ) as myfile:
            torch.save(all_outputs, myfile)

    def convert_input_to_seq(self, input_prot, vocab):
        seq = ""
        for ind in input_prot:
            if vocab[ind] != "p":
                seq += vocab[ind]
            else:
                break
        return seq
