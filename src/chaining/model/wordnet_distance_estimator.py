import torch
import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
import gensim.downloader
from typing import List
from torch_geometric.nn import GraphConv
from torch_geometric.utils import k_hop_subgraph, get_num_hops

class DistanceEstimator(pl.LightningModule):
    def __init__(
        self,
        words: list,
        embedding_dim: int = 50,
        dim_feedforward: int = 16,
        batch_size: int = 8,
    ):
        super().__init__()
        glove_vectors = gensim.downloader.load(f'glove-wiki-gigaword-{embedding_dim}')
        self.key_to_index = glove_vectors.key_to_index 
        self.weights = torch.FloatTensor(glove_vectors.vectors)
        self.embeddings = nn.Embedding.from_pretrained(self.weights)
        self.final_layer = torch.nn.Linear(embedding_dim, dim_feedforward)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.embedding_dim = embedding_dim
        self.conv_1 = GraphConv(embedding_dim, embedding_dim)

    def forward(self, data, batch):
        x, y = batch['x'], batch['y']
        # Reduce the graph to only hops way from batch nodes.
        _, edge_index, _, _ = k_hop_subgraph(torch.concat((x, y)), get_num_hops(self), data.edge_index, relabel_nodes=False)
        print(edge_index)
        # Compute embeddings after propagation for batch nodes.
        embeddings = self.conv_1(self.embeddings.weight, edge_index)
        x = torch.zeros((len(x), self.embedding_dim)) # embeddings[x]
        y = torch.zeros((len(y), self.embedding_dim)) # embeddings[y]
        return (x * y).sum(dim=-1)

if __name__ == "__main__":
    pass