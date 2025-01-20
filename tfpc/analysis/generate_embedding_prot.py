import torch
from config_manager.config_manager import ConfigManager
from dataset_manager.training_dataset_manager import TrainingDatasetManager

# pylint: disable=no-member


class GenerateEmbeddingProt:
    def __init__(self, path_folder_xp):
        self.save_path = "data/embeddings_of_prots/"
        self.xp_name = path_folder_xp.split("/")[-1]
        base_folder = "/".join(path_folder_xp.split("/")[:-2]) + "/"
        self.path_folder_xp = path_folder_xp
        self.config = ConfigManager(self.path_folder_xp + "/config.json")
        path_model_embedder = base_folder + self.config.starting_model.path_model
        self.vocab_inv = {value: key for key, value in self.config.vocab.items()}
        self.base_model = torch.load(path_model_embedder)
        # Load dataset manager
        dico_loaded_vocab = dict()
        for k in range(len(self.config.tasks)):
            try:
                vocab = torch.load(
                    self.path_folder_xp + "/" + self.task_names[k] + "_vocab.pth",
                    map_location=self.config.device,
                )
                dico_loaded_vocab[self.task_names[k]] = vocab
            except:
                pass
        self.dataset_manager = TrainingDatasetManager(
            self.config, no_train=True, dico_loaded_vocab=dico_loaded_vocab
        )

    def calc_embedding(self, ind_task):
        model = torch.nn.DataParallel(self.base_model)
        model = model.eval()

        vector_all_prot = dict()
        dataloader = self.dataset_manager.get_test_dataloader(ind_task)
        compteur = 0
        for input_batch, _ in dataloader:
            print(compteur, "/", len(dataloader))
            embedding = model(input_batch)
            embedding_mean = torch.mean(embedding, dim=1)
            embedding_min = torch.min(embedding, dim=1)[0]
            embedding_max = torch.max(embedding, dim=1)[0]
            embedding_std = torch.std(embedding, dim=1)
            embedding_q1 = torch.quantile(embedding, 0.25, dim=1)
            embedding_q2 = torch.quantile(embedding, 0.5, dim=1)
            embedding_q3 = torch.quantile(embedding, 0.75, dim=1)
            embedding_prots = torch.cat(
                (
                    embedding_mean,
                    embedding_min,
                    embedding_max,
                    embedding_std,
                    embedding_q1,
                    embedding_q2,
                    embedding_q3,
                ),
                dim=1,
            )
            for input_prot, embed_one_prot in zip(input_batch, embedding_prots):
                sequence = self.convert_input(input_prot)
                vector_all_prot[sequence] = embed_one_prot.tolist()
            compteur += 1

        torch.save(
            vector_all_prot,
            self.save_path + self.xp_name + ".pth",
        )

    def convert_input(self, input_vec):
        seq = ""
        for indice_AA in input_vec:
            seq += self.vocab_inv[int(indice_AA)]
        seq = seq.replace("p", "")
        seq = seq.replace("c", "")
        return seq
