"""
This module create WrapperProtBertBFD
"""
# pylint: disable=W0223
import torch.nn as nn


class WrapperProtBertBFD(nn.Module):
    """
    This class define a wrapper for pre trainned model from https://github.com/agemagician/ProtTrans
    """

    def __init__(self, ProtBert_BFD, pad_index, d_model):
        super(WrapperProtBertBFD, self).__init__()
        self.d_model = d_model
        self.ProtBert_BFD = ProtBert_BFD
        self.pad_index = pad_index

    def forward(self, src):
        output = self.ProtBert_BFD(src)
        return output[0]
