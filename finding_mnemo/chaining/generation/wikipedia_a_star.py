from __future__ import annotations

from typing import List

import networkx as nx
import spacy
import wikipediaapi
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from finding_mnemo.chaining.dataset.wikipedia_dataset import WikipediaDataset


class Model:
    nlp: spacy.Language = spacy.load("en_core_web_sm")
    estimator: BaseEstimator
    tfidf: TfidfVectorizer = TfidfVectorizer()  # TODO: re-use from training.

    def __init__(self, estimator: BaseEstimator, tfidf: TfidfVectorizer):
        self.estimator = estimator
        self.tfidf = tfidf


# >> TODO <<
from sklearn.ensemble import RandomForestRegressor

generator = WikipediaDataset(starting_node="Pokémon", hop_nb=10)
train_set, tf = generator.create_dataset(pair_nb=10)
# TODO: TFIDF should be learned on small neighborhood of both source & target.
model = Model(RandomForestRegressor(n_estimators=10, max_depth=10), tf)


class Page:
    page: wikipediaapi.WikipediaPage
    to_target: float = None
    to_start: int = 0
    parent: Page = None
    categories = {}

    def __init__(self, page, to_start: int = 0):
        self.page = page
        self.summary = self.page.summary
        self.to_start = to_start
        self.categories = set(page.categories)
        self.degree = len(page.links)
        self.doc = model.nlp(self.summary)

    def score(self) -> float:
        return self.to_start + self.to_target

    def set_heuristic(self, target: Page):
        # common_categories = len(self.categories.intersection(target.categories))
        # all_categories = len(self.categories.union(target.categories))
        # degree_sum = self.degree + target.degree
        # degree_diff = abs(self.degree - target.degree)
        # similarity = self.doc.similarity(target.doc)
        # distance = 1 / (1e-3 + similarity)
        # percent_cat = common_categories / all_categories
        # tfidf_matrix = model.tfidf.transform((self.summary, target.summary))
        # tfidf = (tfidf_matrix[0]*tfidf_matrix[1].T).toarray()[0][0]
        self.to_target = 0  # model.estimator.predict([common_categories, degree_sum, degree_diff, similarity, distance, percent_cat, tfidf])

    def __eq__(self, __o: Page) -> bool:
        return __o.page == self.page

    def __lt__(self, other: Page) -> bool:
        return self.score() < other.score()


class AStar:
    def __init__(self):
        self.jump_limit = 10
        self.wiki_wiki = wikipediaapi.Wikipedia("en")

    def find_path(self, start: str, target: str):
        nodes = []
        visited_nodes = []

        nodes.append(Page(self.wiki_wiki.page(start)))

        target_page = Page(self.wiki_wiki.page(target))

        count = 0

        while len(nodes) > 0 and count < self.jump_limit:
            count += 1
            nodes = sorted(nodes)  # TODO: Use a cleverer data structure.
            node = nodes.pop(0)
            print(node.page.title)
            if node in visited_nodes:  # Ignore node.
                continue
            elif node == target_page:  # Over.
                return node
            else:
                visited_nodes.append(node)
                neighbors = self.get_neighbors(node)
                for neighbor in tqdm(neighbors, desc="Exploring neighbors"):
                    neighbor_node = Page(
                        self.wiki_wiki.page(neighbor), to_start=node.to_start + 1
                    )
                    if neighbor_node in visited_nodes:
                        continue
                    elif neighbor_node in nodes:
                        for n in nodes:
                            if n == neighbor_node:
                                n.parent = node
                                n.to_start = min(n.to_start, node.to_start + 1)
                    else:
                        neighbor_node.set_heuristic(target_page)
                        neighbor_node.parent = node
                        nodes.append(neighbor_node)

    def get_neighbors(self, node: Page) -> List:
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
        return [
            link
            for link in node.page.links
            if all([not link.startswith(x) for x in forbidden_protocols])
        ]


if __name__ == "__main__":
    a_start = AStar()
    # n = a_start.get_neighbors((word="pride"))
    target = a_start.find_path("pokemon", "digimon")
    # parent = target.parent
    # while not parent is None:
    #     print(parent.word)
    #     parent = parent.parent
    # TODO: handle parents
