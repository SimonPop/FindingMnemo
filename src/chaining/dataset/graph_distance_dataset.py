from torch.utils.data import Dataset
import networkx as nx
from itertools import combinations
from sklearn.cluster import SpectralClustering
from torch_geometric.utils.convert import from_networkx
from dataset.database_handler import DatabaseHandler

class GraphDistanceDataset(Dataset):
    def __init__(self, mode: str = "train"):
        self.client = DatabaseHandler()
        self.mode = mode

        self.graph = self.load_graph()
        self.distance_matrix = self.compute_distances()
        self.graph = self.decouple_graph()
        self.data = from_networkx(self.graph)

        self.training_pairs = []
        self.validation_pairs = []
        self.test_pairs = []

    @staticmethod
    def load_graph() -> nx.Graph:
        """Load graph using client."""
        greeter = DatabaseHandler("bolt://localhost:7687", "simon", "wiktionary")
        #TODO: only those with defition
        query = """
        MATCH (n)-[r]->(c) RETURN *
        """

        results = greeter.driver.session().run(query)

        G = nx.MultiDiGraph()

        nodes = list(results.graph()._nodes.values())
        for node in nodes:
            G.add_node(node.id, labels=node._labels, properties=node._properties)

        rels = list(results.graph()._relationships.values())
        for rel in rels:
            G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

        return G

    def split_graph(self):
        all_pairs = list(self.create_pairs())
        third = len(all_pairs)
        self.training_pairs = all_pairs[:third]
        self.validation_pairs = all_pairs[third:2*third]
        self.test_pairs = all_pairs[2*third:]

    def compute_distances(self):
        """Compute distance between each pair of nodes.
        """
        distance_matrix = nx.floyd_warshall(self.graph)
        return distance_matrix

    def decouple_graph(self, n_component: int):
        """Decouple graph into n_component.

        Args:
            n_component (int): Number of component to decouple the graph into.
        """
        # 1. Find n clusters.
        adj_mat = nx.to_numpy_matrix(self.graph)
        sc = SpectralClustering(n_component, affinity='precomputed', n_init=100)
        sc.fit(adj_mat)
        labels = sc.labels_
        # 2. Remove all edges inter-cluster.
        for node_a, node_b in self.graph.edges():
            if labels[node_a] != labels[node_b]:
                self.graph.remove_edge(node_a, node_b)
        return self.graph

    def create_pairs(self):
        """Create pair of nodes iterator.
        """
        nodes = self.graph.nodes()
        return combinations(nodes, 2)

    def __getitem__(self, index):
        pair = self.get_pairs()[index]
        distance = self.distance_matrix[pair[0]][pair[1]]
        # FIXME: Get full graph or subgraph?
        # TODO convert graph to PyG graph.
        return {
            "label": distance,
            "graph": self.data,
            "pair": pair
        }

    def get_pairs(self):
        if self.mode == "train":
            pairs = self.training_pairs
        elif self.mode == "validation":
            pairs = self.training_pairs
        elif self.mode == "test":
            pairs = self.training_pairs
        else:
            raise ValueError(f'Unknown mode given {self.mode}. Should be train, validation or test.')
        return pairs

    def __len__(self):
        return len(self.get_pairs())


if __name__ == "__main__":
    print(GraphDistanceDataset.load_graph())