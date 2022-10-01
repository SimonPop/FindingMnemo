from ipapy import UNICODE_TO_IPA
import torch
import torch.nn.functional as F
import torch.nn as nn

class SoundSiamese():
    def __init__(self, embedding_dim: int = 16):
        self.vocabulary = {w: i for i, w in enumerate(UNICODE_TO_IPA.keys())}
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.vocabulary), embedding_dim=embedding_dim)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)

    def forward(self, a, b):
        a = self.embedding(torch.tensor([self.vocabulary[x] for x in a]))
        b = self.embedding(torch.tensor([self.vocabulary[x] for x in b]))

        a = self.encoder(a)
        b = self.encoder(b)

        a = torch.sum(a, dim=0)
        b = torch.sum(b, dim=0)

        return torch.dot(a, b)

model = SoundSiamese()
print(model.forward('ˈɑrdˌvɑrk', 'ɑɻ˧˥ tsɯ'))