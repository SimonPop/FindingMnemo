from torch.utils.data import Dataset
import networkx as nx

class PhoneticPairDataset(Dataset):
    def __init__(self, mode: str = "train"):
        self.client = None # TODO: neo4j client?
        self.mode = mode

        self.training_pairs = []
        self.validation_pairs = []
        self.test_pairs = []

        self.distance_matrix = self.compute_distances()

    def load_graph(self) -> nx.Graph:
        """Load graph using client."""
        pass

    def split_graph(self):
        all_pairs = self.create_pairs()
        training_pairs = []
        validation_pairs = []
        test_pairs = []

    def compute_distances(self):
        """Compute distance between each pair of nodes.
        """
        distance_matrix = None
        return distance_matrix

    def decouple_gaph(self, n_component: int):
        """Decouple graph into n_component.

        Args:
            n_component (int): Number of component to decouple the graph into.
        """
        return 

    def create_pairs(self):
        """Create pair of nodes.
        """
        return []

    def __getitem__(self, index):
        pair = self.get_pairs()[index]
        distance = self.distance_matrix[pair[0]][pair[1]]
        # FIXME: Get full graph or subgraph?
        # TODO convert graph to PyG graph.
        return {
            "label": distance,
            "graph": None,
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
        return len(self.get_pairs)

