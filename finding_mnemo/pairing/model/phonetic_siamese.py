from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ipapy import UNICODE_TO_IPA
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from torch import nn, optim

from finding_mnemo.pairing.training.config import LossType


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
        lr: float = 1e-2,
        lambda_triplet: float = 0.5,
        lambda_pos: float = 0.25,
        lambda_neg: float = 0.25,
    ):
        super().__init__()

        self.lambda_triplet = lambda_triplet
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

        self.loss_type = loss_type
        self.batch_size = batch_size

        self.weight_decay = weight_decay
        self.lr = lr

        self.nhead = nhead

        self.padding = padding
        self.embedding_dim = embedding_dim
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(self.vocabulary) + 1, embedding_dim=embedding_dim
        )
        if torch.cuda.is_available():
            self.embedding = self.embedding.cuda()
        if self.nhead > 0:
            self.encoder = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.p_enc_1d_model = PositionalEncoding1D(self.embedding_dim)
        self.p_enc_1d_model_sum = Summer(self.p_enc_1d_model)

        self.triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2)

        self.save_hyperparameters()

    def forward(self, a: List[str], b: List[str]):
        a_encoding = self.encode(a)
        b_encoding = self.encode(b)
        return (a_encoding - b_encoding).pow(2).sum(-1).sqrt()
        # return self.cos(a, b)

    def encode(self, x: List[str]) -> torch.Tensor:
        x_tensor = [
            torch.tensor([self.vocabulary[l] for l in w if l in self.vocabulary])
            for w in x
        ]
        x_tensor = [self.pad(t) for t in x_tensor]
        x_tensor = torch.stack(x_tensor).long()
        if torch.cuda.is_available():
            x_tensor = x_tensor.cuda(0)
        x_tensor = self.embedding(x_tensor)
        x_tensor = self.p_enc_1d_model_sum(x_tensor)
        if self.nhead > 0:
            x_tensor = self.encoder(x_tensor)
        x_tensor = torch.sum(x_tensor, dim=1)
        return x_tensor

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "training_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss

    def _step(self, batch):
        if self.loss_type == LossType.Triplet:
            return self._step_triplet(batch)
        elif self.loss_type == LossType.GenerativeTriplet:
            return self._step_triplet(batch)
        elif self.loss_type == LossType.GenerativeContrastive:
            return self._step_mse(batch)
        elif self.loss_type == LossType.Pair:
            return self._step_mse(batch)
        elif self.loss_type == LossType.Mixed:
            return self._step_mixed(batch)
        else:
            raise ValueError(
                'Unknown loss_type passed: {}. Please choose among ["pair", "triplet", "mixed"].'.format(
                    self.loss_type
                )
            )

    def _step_mse(self, batch):
        chinese_match = batch["phonetic_a"]
        english_match = batch["phonetic_b"]

        y_hat = self.forward(chinese_match, english_match)
        # similarity = 1 - batch["distance"].float()
        loss = nn.functional.mse_loss(y_hat, batch["distance"].float())
        return loss

    def _step_triplet(self, batch):
        anchor_match = batch["anchor_phonetic"]
        positive_match = batch["similar_phonetic"]
        negative_match = batch["distant_phonetic"]

        anchor_embedding = self.encode(anchor_match)
        positive_embedding = self.encode(positive_match)
        negative_embedding = self.encode(negative_match)
        loss = self.triplet_loss(
            anchor_embedding, positive_embedding, negative_embedding
        )
        return loss

    def _step_mixed(self, batch):
        anchor_match = batch["anchor_phonetic"]
        positive_match = batch["similar_phonetic"]
        negative_match = batch["distant_phonetic"]

        positive_distance = batch["similar_distance"].float()
        negative_distance = batch["distant_distance"].float()

        anchor_embedding = self.encode(anchor_match)
        positive_embedding = self.encode(positive_match)
        negative_embedding = self.encode(negative_match)

        positive_distance_hat = torch.sqrt(
            torch.sum((anchor_embedding - positive_embedding) ** 2, dim=1)
        )  # torch.cdist(anchor_embedding, positive_embedding, p=2)
        negative_distance_hat = torch.sqrt(
            torch.sum((anchor_embedding - negative_embedding) ** 2, dim=1)
        )  # torch.cdist(anchor_embedding, negative_embedding, p=2)

        loss_mse_positive = nn.functional.mse_loss(
            positive_distance_hat, positive_distance
        )
        loss_mse_negative = nn.functional.mse_loss(
            negative_distance_hat, negative_distance
        )

        loss = (
            self.lambda_triplet
            * self.triplet_loss(
                anchor_embedding, positive_embedding, negative_embedding
            )
            + self.lambda_pos * loss_mse_positive
            + self.lambda_neg * loss_mse_negative
        )
        return loss

    def pad(self, tensor):
        return nn.functional.pad(
            tensor, (0, self.padding - tensor.size(-1)), mode="constant", value=0
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
