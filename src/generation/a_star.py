from __future__ import annotations

from src.generation.embedding_heuristic import EMBEDDING_HEURISTIC
from typing import List

class Node():
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

class AStar():
    def __init__(self):
        self.jump_limit = 100

    def find_path(self, start: str, target: str):
        nodes = []
        visited_nodes = []

        nodes.append(Node(start))

        while len(nodes) > 0:
            node = nodes.pop() # TODO: get the minimum f.
            if node in visited_nodes: # Ignore node.
                continue
            elif node.word == target: # Over.
                return node
            else:
                visited_nodes.append(node)
                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    neighbor_node = Node(neighbor, to_start=node.to_start + 1)
                    if neighbor_node in visited_nodes:
                        continue
                    elif neighbor_node in nodes:
                        continue
                    else:
                        neighbor_node.set_heuristic(target)
                        nodes.append(neighbor_node)

    def get_neighbors(self, node: Node) -> List[str]:
        # TODO: get neighbors from Neo4J.
        return ["dog", "cat", "table"]

if __name__ == '__main__':
    a_start = AStar()
    print(a_start.find_path("penguin", "table"))
