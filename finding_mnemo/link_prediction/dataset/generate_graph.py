import json

import networkx as nx
import pandas as pd
import wikipediaapi
from tqdm import tqdm


def generate_graph(core_size: int) -> nx.Graph:
    """Creates a graph of Wikipedia pages.

    Args:
        core_size (int): Size of core graph (visited nodes).

    Returns:
        nx.Graph: Created graph.
    """
    api = wikipediaapi.Wikipedia("en")
    graph = nx.Graph()

    forbidden_protocols = [
        "Category",
        "Template",
        "Wikipedia",
        "User",
        "Help",
        "Talk",
        "Portal",
        "File",
        "Module",
    ]

    english_words = pd.read_csv(
        "C:/Users/simon/Projets/FindingMnemo/src/pairing/dataset/pairing/english.csv",
        usecols=["word"],
    ).sample(core_size)["word"]
    for word in tqdm(english_words, desc="creating graph"):
        page = api.page(word)
        links = page.links
        portals = [l for l in links if l.startswith("Portal")]
        links = [
            l for l in links if all([not l.startswith(x) for x in forbidden_protocols])
        ]
        categories = list(page.categories.keys())
        graph.add_node(
            word, summary=page.summary, categories=categories, portals=portals
        )
        for link in links:
            graph.add_edge(word, link)
    return graph


if __name__ == "__main__":
    G = generate_graph(core_size=1000)
    with open("graph.json", "w") as f:
        data = nx.node_link_data(G)
        json.dump(data, f)
