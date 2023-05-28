from __future__ import annotations

from typing import List

import networkx as nx

from finding_mnemo.dataset.database_handler import DatabaseHandler
from finding_mnemo.generation.embedding_heuristic import EMBEDDING_HEURISTIC


class Node:
    word: str
    to_target: float = None
    to_start: int = 0
    parent: Node = None

    def __init__(self, word: str, to_start: int = 0):
        self.word = word
        self.to_start = to_start

    def score(self):
        return self.to_start + self.to_target

    def set_heuristic(self, target_word: str):
        self.to_target = EMBEDDING_HEURISTIC.distance(self.word, target_word)

    def __eq__(self, __o: Node) -> bool:
        return __o.word == self.word


class AStar:
    def __init__(self):
        self.jump_limit = 10
        self.handler = DatabaseHandler("bolt://localhost:7687", "simon", "wiktionary")
        self.temporary_graph: nx.Graph = self.handler.load_graph()

    def find_path(self, start: str, target: str):
        nodes = []
        visited_nodes = []

        nodes.append(Node(start))

        count = 0

        while len(nodes) > 0 and count < self.jump_limit:
            count += 1
            node = nodes.pop()  # TODO: get the minimum f.
            if node in visited_nodes:  # Ignore node.
                continue
            elif node.word == target:  # Over.
                return node
            else:
                visited_nodes.append(node)
                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    neighbor_node = Node(neighbor, to_start=node.to_start + 1)
                    if neighbor_node in visited_nodes:
                        continue
                    elif neighbor_node in nodes:
                        for n in nodes:
                            if n == neighbor_node:
                                n.parent = node
                                # n.to_start = min(n.to_start, node.to_start + 1)
                    elif EMBEDDING_HEURISTIC.word_available(neighbor_node.word):
                        neighbor_node.set_heuristic(target)
                        neighbor_node.parent = node
                        nodes.append(neighbor_node)

    def get_neighbors(self, node: Node) -> List[str]:
        return self.temporary_graph.neighbors(node.word)


if __name__ == "__main__":
    a_start = AStar()
    n = a_start.get_neighbors(Node(word="pride"))
    target = a_start.find_path("dignity", "pride")
    parent = target.parent
    while not parent is None:
        print(parent.word)
        parent = parent.parent
    # TODO: handle parents
