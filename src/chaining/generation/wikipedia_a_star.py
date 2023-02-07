from __future__ import annotations

from typing import List
import networkx as nx
import wikipediaapi
import spacy
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

class Model():
    nlp: spacy.Language = spacy.load("en_core_web_sm")
    estimator: BaseEstimator
    tfidf: TfidfVectorizer

model = Model() # >> TODO <<

class Page():
    page: str
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
        common_categories = len(self.categories.intersection(target.categories))
        all_categories = len(self.categories.union(target.categories))
        degree_sum = self.degree + target.degree
        degree_diff = abs(self.degree - target.degree)
        similarity = self.doc.similarity(target.doc)
        distance = 1 / (1e-3 + similarity)
        percent_cat = common_categories / all_categories
        tfidf = (model.tfidf.transform(self.summary)*model.tfidf.transform(target.summary).T).toarray()[0][0]
        self.to_target = model.predict([common_categories, degree_sum, degree_diff, similarity, distance, percent_cat, tfidf])

    def __eq__(self, __o: Page) -> bool:
        return __o.word == self.word 
    
    def __lt__(self, other: Page) -> bool:
         return self.score() < other.score()

class AStar():
    def __init__(self):
        self.jump_limit = 10
        self.wiki_wiki = wikipediaapi.Wikipedia('en')

    def find_path(self, start: str, target: str):
        nodes = []
        visited_nodes = []

        nodes.append(Page(self.wiki_wiki.page(start)))

        count = 0

        while len(nodes) > 0 and count < self.jump_limit:
            count += 1
            nodes = sorted(nodes) #TODO: Use a cleverer data structure.
            node = nodes.pop(0)
            if node in visited_nodes: # Ignore node.
                continue
            elif node.word == target: # Over.
                return node
            else:
                visited_nodes.append(node)
                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    neighbor_node = Page(neighbor, to_start=node.to_start + 1)
                    if neighbor_node in visited_nodes:
                        continue
                    elif neighbor_node in nodes:
                        for n in nodes:
                            if n == neighbor_node:
                                n.parent = node
                                n.to_start = min(n.to_start, node.to_start + 1)
                    else:
                        neighbor_node.set_heuristic(target)
                        neighbor_node.parent = node
                        nodes.append(neighbor_node)

    def get_neighbors(self, node: Page) -> List:
        return node.page.links

if __name__ == '__main__':
    a_start = AStar()
    n = a_start.get_neighbors(Node(word="pride"))
    target = a_start.find_path("dignity", "pride")
    parent = target.parent
    while not parent is None:
        print(parent.word)
        parent = parent.parent
    # TODO: handle parents