import wikipediaapi
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
import random
from numpy import dot
from numpy.linalg import norm
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass

    
@dataclass
class PairRow():
    length: int 
    src: str 
    tgt: str 
    distance: float
    similarity: float
    common_cat: int
    percent_cat: float
    degree_sum: int
    degree_diff: int
    tfidf: float

class WikipediaDataset():
    """
    Dataset for learning to predict distance between two pages of Wikipedia (in terms of link hops).
    Contains pairs of pages, their distances, and some predictive features.
    """

    def __init__(self, starting_node="Koala", hop_nb = 100):
        self.wiki_wiki = wikipediaapi.Wikipedia('en')
        self.graph = nx.Graph()
        self.node2page = {} 
        self.starting_node = starting_node
        self.hop_nb = hop_nb
        self.nlp = spacy.load("en_core_web_sm")
        self.create_graph()

    def create_graph(self):
        """BFS walk to create a Graph of Wikipedia pages."""
        queue = [self.wiki_wiki.page(self.starting_node)]

        forbidden_protocols = ["Category", "Template", "Wikipedia", "User", "Help", "Talk", "Portal", "File", "Module"]

        self.node2page = {**self.node2page, queue[0].title: queue[0]}

        for hop in tqdm(range(self.hop_nb), desc="Graph accumulation."):
            if len(queue) > 0:
                page = queue.pop()
                self.node2page[page.title] = page
                for name, neighbor_page in page.links.items():
                    if all([not name.startswith(x) for x in forbidden_protocols]):
                        if name not in self.graph.nodes and neighbor_page not in queue: # graph acts as a "visited" data structure.
                            queue.append(neighbor_page)
                        self.graph.add_edge(page.title, name)

    def create_dataset(self, pair_nb = 100) -> pd.DataFrame:
        samples = np.random.choice(list(self.node2page.keys()), pair_nb*2).tolist()
        summaries = [self.node2page[x].summary for x in samples]
        categories = [self.node2page[x].categories for x in tqdm(samples)]
        docs = [x for x in self.nlp.pipe(samples)]
        pairs = np.array(range(len(samples))).reshape(2, -1) 
        tfidf = TfidfVectorizer().fit_transform(summaries)
        rows = []

        for src, tgt in tqdm(zip(*pairs)):
            row = self.compute_pair_features(src, tgt, docs, categories, samples, tfidf)
            rows.append(row)

        return pd.DataFrame(rows)

    def compute_pair_features(self, src: int, tgt: int, docs, categories, samples, tfidf) -> PairRow:
        length = nx.shortest_path_length(self.graph, samples[src], samples[tgt])
        doc_a = docs[src]
        doc_b = docs[tgt]
        degree_a = self.graph.degree[samples[src]]
        degree_b = self.graph.degree[samples[tgt]]
        similarity = doc_a.similarity(doc_b)
        common_categories = len(set(categories[src]).intersection(categories[tgt]))
        all_categories = len(set(categories[src]).union(categories[tgt])) + 1

        row = PairRow(
            length = length,
            src = samples[src],
            tgt = samples[tgt],
            distance = 1 / (1e-3 + similarity),
            similarity = similarity,
            common_cat = common_categories,
            percent_cat = common_categories / all_categories,
            degree_sum = degree_a + degree_b,
            degree_diff = abs(degree_a - degree_b),
            tfidf = (tfidf[src]*tfidf[tgt].T).toarray()[0][0],
        )

        return row      