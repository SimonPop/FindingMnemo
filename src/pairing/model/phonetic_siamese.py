from ipapy import UNICODE_TO_IPA
import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from typing import List
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from src.pairing.training.config import LossType

class PhoneticSiamese(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        padding: int = 50,
        nhead: int = 1,
        dim_feedforward: int = 16,
        loss_type: LossType = LossType.Pair,
        batch_size: int = 8,
        margin: float = 0.2,
        weight_decay: float = 1e-4,
        lr: float = 1e-2
    ):
        super().__init__()

        self.loss_type = loss_type
        self.batch_size = batch_size

        self.weight_decay = weight_decay
        self.lr = lr

        self.padding = padding
        self.embedding_dim = embedding_dim
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocabulary) + 1, embedding_dim=embedding_dim
        )
        if torch.cuda.is_available():
            self.embedding = self.embedding.cuda()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.p_enc_1d_model = PositionalEncoding1D(self.embedding_dim)
        self.p_enc_1d_model_sum = Summer(self.p_enc_1d_model)

        self.triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2)

        self.save_hyperparameters()

    def forward(self, a: List[str], b: List[str]):
        a = self.encode(a)
        b = self.encode(b)
        return self.cos(a, b)

    def encode(self, x: List[str]) -> List[torch.tensor]:
        x = [torch.tensor([self.vocabulary[l] for l in w if l in self.vocabulary]) for w in x]
        x = [self.pad(t) for t in x]
        x = torch.stack(x).long()
        if torch.cuda.is_available():
            x = x.cuda(0)
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
            batch_size=self.batch_size
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
            batch_size=self.batch_size
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size
        )
        return loss

    def _step(self, batch):
        if self.loss_type == LossType.Triplet:
            return self._step_triplet(batch)
        elif self.loss_type == LossType.Pair:
            return self._step_mse(batch)
        else:
            raise ValueError('Unknown loss_type passed: {}. Please choose among ["pair", "triplet"].'.format(self.loss_type))

    def _step_mse(self, batch):
        chinese_match = batch["chinese_phonetic"]
        english_match = batch["english_phonetic"]

        y_hat = self.forward(chinese_match, english_match)
        similarity = 1 - batch["distance"].float()
        loss = nn.functional.mse_loss(y_hat, similarity)
        return loss

    def  _step_triplet(self, batch):
        anchor_match = batch["anchor_phonetic"]
        positive_match = batch["similar_phonetic"]
        negative_match = batch["distant_phonetic"]

        anchor_embedding = self.encode(anchor_match)
        positive_embedding = self.encode(positive_match)
        neegative_embedding = self.encode(negative_match)
        loss = self.triplet_loss(anchor_embedding, positive_embedding, neegative_embedding)
        return loss


    def pad(self, tensor):
        return nn.functional.pad(
            tensor, (0, self.padding - tensor.size(-1)), mode="constant", value=0
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
