
# dataset = LabelledWordNet18RR(root='/tmp/WordNetRR')
# print(dataset[0])
# print(dataset[1])
# print(dataset[2])
# print(dataset[3])

# print(len(dataset))
# print(dataset.id2node)

# import nltk
# # nltk.download('wordnet')
# from nltk.corpus import wordnet as wn
# print(wn.synset('freak.n.01').definition())
# print(wn.synset('freak.n.01').examples())


# TODO: Generate neighboring sub-graph for (knn) a pair: the closest nodes from tgt and src only. (or compute everything for batch?)
# TODO: Create vocabulary <-> index of the graph to get the embeddings.

from src.chaining.dataset.wordnet import LabelledWordNet18RR
from src.chaining.model.wordnet_distance_estimator import DistanceEstimator
from src.chaining.training.config import CONFIG
from torch.utils.data import Dataset
from pathlib import Path

from torch.utils.data import DataLoader

def get_dataset() -> Dataset:
    return LabelledWordNet18RR(root='/tmp/WordNetRR')

dataset = get_dataset()

if __name__ == "__main__":
    loader = DataLoader(dataset)
    sample = next(iter(loader))
    print(sample)
    words = [dataset.id2node[str(i)] for i in range(len(dataset.id2node))]
    model = DistanceEstimator(words)
    print(model(dataset.data, sample))

    # TODO: Create an embedding dict for words contained in wordnet