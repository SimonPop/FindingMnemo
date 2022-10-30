from ipapy import UNICODE_TO_IPA
import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from typing import List
from positional_encodings.torch_encodings import PositionalEncoding1D


class SoundSiamese(pl.LightningModule):
    def __init__(
        self, embedding_dim: int = 16, dropout: float = 0.1, add_positional=False
    ):
        super().__init__()
        self.add_positional = add_positional
        self.embedding_dim = embedding_dim
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocabulary), embedding_dim=embedding_dim
        )
        # self.pos_encoder = PositionalEncoder(embedding_dim, dropout, add=add_positional, max_len=30)
        self.p_enc_1d_model = PositionalEncoding1D(self.embedding_dim)
        # self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=1)
        ## Custom
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=1, dropout=dropout
        )
        self.activation = nn.functional.relu
        self.linear1 = nn.Linear(
            in_features=embedding_dim * 2, out_features=embedding_dim * 2
        )

    def forward(self, a: List[str], b: List[str]):
        a = self.encode(a)
        b = self.encode(b)
        return torch.sigmoid(
            torch.stack([torch.dot(e_a, e_b) for e_a, e_b in zip(a, b)])
        )

    def encode(self, x: List[str]) -> List[torch.tensor]:
        x = [torch.tensor([self.vocabulary[l] for l in w]) for w in x]
        x = [self.embedding(v) for v in x]
        # x = [v.view(1, -1, self.embedding_dim) for v in x]
        # x = [torch.cat([v, self.p_enc_1d_model(v)], dim=1) for v in x]
        # x = [self.encoder(v) for v in x]
        # x = [self.self_attn(v, v, v)[0] for v in x]
        # x = [v.view(-1, self.embedding_dim) for v in x]
        # x = [self.activation(self.linear1(v)) for v in x]
        x = [torch.sum(v, dim=0) for v in x]
        return x

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "training_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _step(self, batch):
        chinese_match = batch["chinese_phonetic"]
        english_match = batch["english_phonetic"]
        y_hat = self.forward(chinese_match, english_match)
        similarity = 1 - batch["distance"].float()
        # loss = nn.functional.cross_entropy(y_hat, distance)
        loss = nn.functional.mse_loss(y_hat, similarity)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer
