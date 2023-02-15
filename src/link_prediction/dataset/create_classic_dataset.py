"""Creates a dataset to be used with classic ML (not GNN -> information about the graph itself is lost)."""
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass
class LinkPairDataset:
    vectorizer: TfidfVectorizer
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def create_dataset() -> LinkPairDataset:
    vectorizer = TfidfVectorizer()
    # 1. Load graph
    G = nx.read_graphml("graph")
    # 2. Generate features for each nodes -> TF-IDF, Categories, Degree
    summaries = nx.get_node_attributes(G, "summary")
    tfidf_vectors = vectorizer.fit_transform(summaries)
    categories = nx.get_node_attributes(G, "categories")
    degrees = G.degree
    # 3. Split nodes into train / val / test
    third = len(G.nodes)//3
    mask = np.shuffle(["train"]*third + ["val"]*third + ["test"]*(len(G.nodes) - 2*third))
    # 4. Generate pairs between nodes of each sets
    nodes = np.array(G.nodes)
    sets = {}
    for mode in ['train', 'val', 'test']:
        indexes = np.argwhere(mask == mode)
        mode_nodes = nodes[indexes]
        # TODO: sample pairs
        pairs = None
        # TODO: compute features for the pair
        pair_features = None
        #TODO: Store in dataframe
        sets[mode] = pd.DataFrame({**pairs, **pair_features}, columns=['A', 'B', 'TF-IDF', "common_categories", "degree_sum", "degree_diff"])
    return {
        "vectorizer": vectorizer,
        **sets
    }