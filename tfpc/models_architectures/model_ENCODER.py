"""
This module define the EncoderModel
"""
import math
import torch.nn as nn
import torch


# Create pytorch Transformer model
class model_ENCODER(nn.Module):  # pylint: disable=invalid-name,abstract-method
    """
    This class was created a long time ago but we CANNOT change the content because
     we load the pre training model with this
    """

    def __init__(
        self, ntoken, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
    ):
        super(model_ENCODER, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pad_indice = None
        # GeLu pas disponible avec la veille version de pytorch sur le cluster
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )  # ,activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_bias = nn.Parameter(torch.zeros(ntoken))

    def forward(self, src):
        mask_pad_token = src == self.pad_indice
        src = self.encoder(src)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask_pad_token)
        output = output.transpose(0, 1)
        output = (
            torch.nn.functional.linear(output, self.encoder.weight) + self.decoder_bias
        )
        return output


class PositionalEncoding(nn.Module):  # pylint: disable=abstract-method
    """
    This class code the position of the amino acid, because transformer don't know the order instead
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()  # pylint: disable=bad-super-call
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
