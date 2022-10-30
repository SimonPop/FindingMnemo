from ipapy import UNICODE_TO_IPA
import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from typing import List
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class SoundSiamese(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        padding: int = 50,
        nhead: int = 1,
        dim_feedforward: int = 16,
    ):
        super().__init__()
        self.padding = padding
        self.embedding_dim = embedding_dim
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocabulary) + 1, embedding_dim=embedding_dim
        )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.p_enc_1d_model = PositionalEncoding1D(self.embedding_dim)
        self.p_enc_1d_model_sum = Summer(self.p_enc_1d_model)
        self.save_hyperparameters()

    def forward(self, a: List[str], b: List[str]):
        a = self.encode(a)
        b = self.encode(b)
        return self.cos(a, b)

    def encode(self, x: List[str]) -> List[torch.tensor]:
        x = [torch.tensor([self.vocabulary[l] for l in w]) for w in x]
        x = [self.pad(t) for t in x]
        x = torch.stack(x)
        x = self.embedding(x)
        x = self.p_enc_1d_model_sum(x)
        x = self.encoder(x)
        x = torch.sum(x, dim=1)
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

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def _step(self, batch):
        chinese_match = batch["chinese_phonetic"]
        english_match = batch["english_phonetic"]

        y_hat = self.forward(chinese_match, english_match)
        similarity = 1 - batch["distance"].float()
        loss = nn.functional.mse_loss(y_hat, similarity)
        return loss

    def pad(self, tensor):
        return nn.functional.pad(
            tensor, (0, self.padding - tensor.size(-1)), mode="constant", value=0
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer
