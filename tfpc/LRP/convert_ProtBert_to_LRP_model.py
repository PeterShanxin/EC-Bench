import types
import torch
from LRP.layers_ours import (
    forward_hook,
    safe_divide,
    RelProp,
    IndexSelect,
)
from LRP.BERT import BertModel
import torch.nn.functional as F


def convertProtBertToLRPModel(model):
    # Relprop of TransformerClassifPerProtLayerNorm
    model.forward = types.MethodType(forward_TransformerClassifPerProtLayerNorm, model)
    model.pool = IndexSelect()
    model.relprop = types.MethodType(relprop_TransformerClassifPerProtLayerNorm, model)
    model.norm1.relprop = types.MethodType(relprop_base, model.norm1)

    class Patched(model.classe_embedding.__class__, RelProp):
        pass

    model.classe_embedding.__class__ = Patched
    model.classe_embedding.register_forward_hook(forward_hook)
    model.classe_embedding.relprop = types.MethodType(
        relprop_Linear, model.classe_embedding
    )

    model.dropout.relprop = types.MethodType(relprop_base, model.dropout)

    model.transformer_embedder.relprop = types.MethodType(
        relprop_WrapperProtBertBFD, model.transformer_embedder
    )

    new_version_BertModel = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    transfer_weights(model.transformer_embedder.ProtBert_BFD, new_version_BertModel)
    model.transformer_embedder.ProtBert_BFD = new_version_BertModel
    return model


def forward_TransformerClassifPerProtLayerNorm(self, src):
    output = self.transformer_embedder(src)
    output = self.pool(
        output, 1, torch.tensor(0, device=output.device)
    )  # equivalent to output[:, 0]
    output = output.squeeze(1)
    output = self.dropout(output)
    output = self.classe_embedding(output)
    output = self.norm1(output)
    return output


def transfer_weights(model1, model2):
    pretrained_dict = model1.state_dict()
    model_dict = model2.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model2.load_state_dict(model_dict)


def relprop_base(self, R, alpha):
    return R


def relprop_TransformerClassifPerProtLayerNorm(self, cam, **kwargs):
    cam = self.norm1.relprop(cam, **kwargs)
    cam = self.classe_embedding.relprop(cam, **kwargs)
    # print(cam.shape)
    cam = self.dropout.relprop(cam, **kwargs)
    cam = self.transformer_embedder.relprop(cam, **kwargs)
    return cam


def relprop_WrapperProtBertBFD(self, cam, **kwargs):
    cam = self.ProtBert_BFD.relprop(cam, **kwargs)
    return cam


def relprop_Linear(self, R, alpha):
    beta = alpha - 1
    pw = torch.clamp(self.weight, min=0)
    nw = torch.clamp(self.weight, max=0)
    px = torch.clamp(self.X, min=0)
    nx = torch.clamp(self.X, max=0)

    def f(w1, w2, x1, x2):
        Z1 = F.linear(x1, w1)
        Z2 = F.linear(x2, w2)
        S1 = safe_divide(R, Z1)
        S2 = safe_divide(R, Z2)
        C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
        C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

        return C1 + C2

    activator_relevances = f(pw, nw, px, nx)
    inhibitor_relevances = f(nw, pw, px, nx)

    R = alpha * activator_relevances - beta * inhibitor_relevances

    return R