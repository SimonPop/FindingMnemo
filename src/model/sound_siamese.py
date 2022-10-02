from ipapy import UNICODE_TO_IPA
import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl

class SoundSiamese(pl.LightningModule):
    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=embedding_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)

    def forward(self, a, b):
        a = self.encode(a)
        b = self.encode(b)
        return torch.dot(a, b)

    def encode(self, a):
        a = self.embedding(torch.tensor([self.vocabulary[x] for x in a]))
        a = self.encoder(a)
        a = torch.sum(a, dim=0)
        return a

    def training_step(self, batch, batch_idx):
        chinese_match = batch['chinese_match'] 
        english_match = batch['english_match']
        y_hat = self.forward(chinese_match, english_match)
        distance = batch['distance'] 
        loss = nn.functional.mse_loss(y_hat, distance)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
