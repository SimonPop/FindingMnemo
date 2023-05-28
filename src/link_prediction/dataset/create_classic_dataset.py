"""Creates a dataset to be used with classic ML (not GNN -> information about the graph itself is lost)."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class LinkPairDataset:
    vectorizer: TfidfVectorizer
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def create_dataset(G, size: int = 1000) -> LinkPairDataset:
    vectorizer = TfidfVectorizer()
    # 2. Generate features for each nodes -> TF-IDF, Categories, Degree
    summaries = nx.get_node_attributes(G, "summary")
    tfidf_vectors = vectorizer.fit_transform(summaries)
    categories = [set(x) for x in nx.get_node_attributes(G, "categories").values()]
    degrees = G.degree
    # 3. Split nodes into train / val / test
    enriched_nodes = np.array(list(summaries.keys()))
    third = len(enriched_nodes) // 3
    mask = np.array(
        ["train"] * third
        + ["val"] * third
        + ["test"] * (len(enriched_nodes) - 2 * third)
    )
    np.random.shuffle(mask)
    # 4. Generate pairs between nodes of each sets
    sets = {}
    for mode in ["train", "val", "test"]:
        indexes = np.argwhere(mask == mode).reshape(-1)
        subgraph_nodes = [enriched_nodes[i] for i in indexes]
        H = nx.adjacency_matrix(G.subgraph(subgraph_nodes))
        # positive_pairs = np.argwhere(H == 1)
        positive_pairs = (
            []
        )  # [(x[0], x[1]) for x in positive_pairs[np.random.randint(0, len(positive_pairs), size//2)]]
        negative_pairs = np.argwhere(H == 0)
        negative_pairs = [
            (x[0], x[1])
            for x in negative_pairs[
                np.random.randint(0, len(negative_pairs), size // 2)
            ]
        ]
        # TODO add as many positive than negative pairs.
        # pairs = np.random.randint(0, len(indexes), (2, size))
        # index_pairs = indexes[pairs]
        data = []
        for a, b in positive_pairs:
            pair_features = {
                "a": enriched_nodes[a],
                "b": enriched_nodes[b],
                "TF-IDF": (tfidf_vectors[a] * tfidf_vectors[b].T).toarray()[0][0],
                "common_neighbors": len(
                    list(nx.common_neighbors(G, enriched_nodes[a], enriched_nodes[b]))
                ),
                "common_categories": len(categories[a].intersection(categories[b])),
                "degree_sum": degrees[enriched_nodes[a]] + degrees[enriched_nodes[b]],
                "degree_diff": abs(
                    degrees[enriched_nodes[a]] - degrees[enriched_nodes[b]]
                ),
                "connected": enriched_nodes[a] in nx.neighbors(G, enriched_nodes[b])
                or enriched_nodes[b] in nx.neighbors(G, enriched_nodes[a]),
            }
            data.append(pair_features)
        for a, b in negative_pairs:
            pair_features = {
                "a": enriched_nodes[a],
                "b": enriched_nodes[b],
                "TF-IDF": (tfidf_vectors[a] * tfidf_vectors[b].T).toarray()[0][0],
                "common_neighbors": len(
                    list(nx.common_neighbors(G, enriched_nodes[a], enriched_nodes[b]))
                ),
                "common_categories": len(categories[a].intersection(categories[b])),
                "degree_sum": degrees[enriched_nodes[a]] + degrees[enriched_nodes[b]],
                "degree_diff": abs(
                    degrees[enriched_nodes[a]] - degrees[enriched_nodes[b]]
                ),
                "connected": enriched_nodes[a] in nx.neighbors(G, enriched_nodes[b])
                or enriched_nodes[b] in nx.neighbors(G, enriched_nodes[a]),
            }
            data.append(pair_features)
        sets[mode] = pd.DataFrame(data)
    return {"vectorizer": vectorizer, **sets}


if __name__ == "__main__":
    import json

    with open("graph.json", "r") as f:
        G = nx.node_link_graph(json.load(f))
    dataset = create_dataset(G, 100)
    print(dataset["train"])
