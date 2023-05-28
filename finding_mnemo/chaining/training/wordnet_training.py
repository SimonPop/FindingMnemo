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

from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from finding_mnemo.chaining.dataset.wordnet import LabelledWordNet18RR
from finding_mnemo.chaining.model.wordnet_distance_estimator import DistanceEstimator
from finding_mnemo.chaining.training.config import CONFIG


def get_dataset() -> Dataset:
    return LabelledWordNet18RR(root="/tmp/WordNetRR")


dataset = get_dataset()

if __name__ == "__main__":
    loader = DataLoader(dataset)
    sample = next(iter(loader))
    print(sample)
    words = [dataset.id2node[str(i)] for i in range(len(dataset.id2node))]

    words = [x.split(".")[0].replace("_", " ") for x in words]

    import gensim.downloader

    glove_vectors = gensim.downloader.load(f"glove-wiki-gigaword-{50}")
    print(glove_vectors["addis ababa"])

    # print(words)

    # model = DistanceEstimator(words)
    # print(model(dataset.data, sample))

    # TODO: Create an embedding dict for words contained in wordnet
