import torch
import sys

sys.path.append("../models_architectures/")
sys.path.append("../")

model = torch.load(
    "data/models/fine_tune_models/ProtBert_EC40_layer_norm_proba_r1/classif_EC_pred_lvl_2.pth",
    map_location=torch.device("cpu"),
)
vocab = torch.load(
    "data/models/pre_trained_models/30_layer_uniparc+BFD/vocab.pkl",
    map_location=torch.device("cpu"),
)

print(vocab)

model = model.eval()
one_weigth_selection = model.transformer_embedder.ProtBert_BFD.encoder.layer[
    0
].intermediate.dense.weight

sequence_test = "MSTAGKVIKCKAAVLWELKKPFSIEEVEVAPPKAHEVRIKMVAAGICRSD"

input_tensor = torch.tensor([vocab[s] for s in sequence_test]).unsqueeze(0)

res = model(input_tensor)

print(one_weigth_selection[0][0])
one_weigth_selection[0][0].data = torch.tensor(1200.0)
print(one_weigth_selection[0][0])
# Batch 0 et classe 0
res[0][0].backward()
print(one_weigth_selection.grad[0][0])
