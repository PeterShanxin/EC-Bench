"""
This module create TransformerClassifPerProt
"""
# pylint: disable=W0223
import torch.nn as nn
import torch


class TransformerClassifPerProtApriori(nn.Module):
    """
    This class define a Transformer model that classify an entire protein to ONE classe
    """

    def __init__(self, transformer_embedder, num_classes, dropout):
        super(TransformerClassifPerProtApriori, self).__init__()

        self.transformer_embedder = transformer_embedder

        self.d_model = transformer_embedder.d_model
        self.encoder = transformer_embedder.encoder
        self.pos_encoder = transformer_embedder.pos_encoder
        self.transformer_encoder = transformer_embedder.transformer_encoder
        self.pad_index = transformer_embedder.pad_index

        self.dropout = nn.Dropout(p=dropout)
        self.classe_embedding = nn.Linear(int(self.d_model), int(num_classes))

    def forward(self, src):
        apriori, src = src
        bs = src.shape[0]
        apriori = apriori.unsqueeze(1)

        mask_pad_token = src == self.pad_index
        if torch.cuda.is_available():
            append_mask = torch.tensor(
                [False for _ in range(bs)], device=torch.device("cuda")
            )
        else:
            append_mask = torch.tensor([False for _ in range(bs)])

        append_mask = append_mask.unsqueeze(1)

        mask_pad_token = torch.cat((append_mask, mask_pad_token), dim=1)

        output = self.encoder(src)

        output = torch.cat((apriori, output), dim=1)

        output = output.transpose(0, 1)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output, src_key_padding_mask=mask_pad_token)
        output = output.transpose(0, 1)

        output = self.dropout(output[:, 0])
        output = self.classe_embedding(output)
        return output
