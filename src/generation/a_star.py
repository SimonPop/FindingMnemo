from __future__ import annotations

from .embedding_heuristic import EMBEDDING_HEURISTIC
from typing import List

class Node():
    word: str
    to_target: float = None
    to_start: int = None
    parent: Node = None

    def __init__(self, word: str):
        self.word = word

    def score(self):
        return self.to_start + self.to_target

    def set_heuristic(self, target_word: str):
        self.to_target = EMBEDDING_HEURISTIC.distance(target_word)

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
                neighbors = self.get_neighbors()
                for neighbor in neighbors:
                    neighbor_node = Node(neighbor, to_start=node.to_start + 1)
                    if neighbor_node in visited_nodes:
                        continue
                    elif neighbor_node in nodes:
                        continue
                    else:
                        neighbor_node.set_heuristic(target)
                        nodes.append(neighbor_node)


        # Check if both word exist in the graph.
        # 1. Add starting node to the heap.
        # 2. Select next node.
        # 3. Add it to visited nodes.
        # 4. Add all not visited neighbors to the heap. 

    def get_neighbors(self, node: Node) -> List[str]:
        return []