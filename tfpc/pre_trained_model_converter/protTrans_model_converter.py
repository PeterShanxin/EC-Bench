import sys

sys.path.append("models_architectures/")
from wrapper_prot_bert_BFD import WrapperProtBertBFD
from transformers import AutoTokenizer, AutoModel
import os
import torch

Model_name_on_hugging_face = "Rostlab/prot_bert"
d_model = 1024
saving_directory = "data/models/pre_trained_models/"
model_directory = "ProtBert_Uniref100/"
model_name = "model_embedder.pth"
voc_name = "vocab.pkl"


os.mkdir(saving_directory + model_directory)
tokenizer = AutoTokenizer.from_pretrained(Model_name_on_hugging_face)
model = AutoModel.from_pretrained(Model_name_on_hugging_face)

dict_vocab = tokenizer.vocab
print(dict_vocab)

dict_vocab["c"] = dict_vocab.pop("[CLS]")
dict_vocab["m"] = dict_vocab.pop("[MASK]")
dict_vocab["p"] = dict_vocab.pop("[PAD]")
dict_vocab["X"] = dict_vocab.pop("[UNK]")

print(dict_vocab)
pad_index = dict_vocab["p"]
model = WrapperProtBertBFD(model, pad_index, d_model)
torch.save(model, saving_directory + model_directory + model_name)
torch.save(dict_vocab, saving_directory + model_directory + voc_name)
