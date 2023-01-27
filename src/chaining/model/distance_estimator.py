import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import gensim.downloader
from typing import List

class DistanceEstimator(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 50,
        dim_feedforward: int = 16,
        batch_size: int = 8
    ):
        super().__init__()
        glove_vectors = gensim.downloader.load(f'glove-wiki-gigaword-{embedding_dim}')
        self.key_to_index = glove_vectors.key_to_index 
        weights = torch.FloatTensor(glove_vectors.vectors)
        self.embeddings = nn.Embedding.from_pretrained(weights)
        self.final_layer = torch.nn.Linear(embedding_dim, dim_feedforward)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input_a: List[str], input_b: List[str]):
        # Convert to embedding indexes.
        a = torch.tensor([self.key_to_index[x] for x in input_a])
        b = torch.tensor([self.key_to_index[x] for x in input_b])
        # Convert to embeddings.
        a = self.embeddings(a)
        b = self.embeddings(b)
        # TODO: Add graph refining.
        # TODO: estimate distance instead of similarity.
        return self.similarity(a, b)


if __name__ == "__main__":
    estimator = DistanceEstimator()
    sim = estimator(['dog'], ['cat'])
    print(sim)