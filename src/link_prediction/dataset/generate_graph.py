import networkx as nx
import pandas as pd
import wikipediaapi
from tqdm import tqdm

def generate_graph() -> nx.Graph:
    api = wikipediaapi.Wikipedia('en')
    graph = nx.Graph()
    english_words = pd.read_csv('C:/Users/simon/Projets/FindingMnemo/src/pairing/dataset/pairing/english.csv', usecols=['word'])['word']
    for word in tqdm(english_words, desc="creating graph"):
        page = api.page(word)
        links = page.links
        graph.add_node(word, summary=page.summary, categories=page.categories)
        for link in links:
            graph.add_edge(word, link)
    return graph


if __name__ == "__main__":
    graph = generate_graph()
    nx.write_graphml(graph, "graph")