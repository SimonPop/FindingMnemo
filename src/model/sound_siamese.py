from ipapy import UNICODE_TO_IPA
import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from typing import List

class SoundSiamese(pl.LightningModule):
    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=embedding_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)

    def forward(self, a: List[str], b: List[str]):
        a = self.encode(a)
        b = self.encode(b)
        return torch.sigmoid(torch.stack([torch.dot(e_a, e_b) for e_a, e_b in zip(a,b)]))

    def encode(self, x: List[str]) -> List[torch.tensor]:
        x = [torch.tensor([self.vocabulary[l] for l in w]) for w in x]
        x = [self.embedding(v) for v in x]
        # x = [self.encoder(v) for v in x]
        x = [torch.sum(v, dim=0) for v in x]
        return x

    def training_step(self, batch, batch_idx):
        chinese_match = batch['chinese_phonetic'] 
        english_match = batch['english_phonetic']
        y_hat = self.forward(chinese_match, english_match)
        distance = batch['distance'].float()
        loss = nn.functional.mse_loss(y_hat, distance)
        self.log("Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
