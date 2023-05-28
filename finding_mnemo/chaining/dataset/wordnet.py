import json
from itertools import chain
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils.convert import to_networkx


class LabelledWordNet18RR(InMemoryDataset):
    r"""Modified WordNet18RR dataset from PyG including word names.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = (
        "https://raw.githubusercontent.com/villmow/"
        "datasets_knowledge_embedding/master/WN18RR/text"
    )

    edge2id = {
        "_also_see": 0,
        "_derivationally_related_form": 1,
        "_has_part": 2,
        "_hypernym": 3,
        "_instance_hypernym": 4,
        "_member_meronym": 5,
        "_member_of_domain_region": 6,
        "_member_of_domain_usage": 7,
        "_similar_to": 8,
        "_synset_domain_topic_of": 9,
        "_verb_group": 10,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        n_sample: int = 1000,
        mode: str = "train",
    ):
        self.n_sample = n_sample
        self.mode = mode
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[0][:-3] + "id.json", "r") as f:
            self.id2node = json.load(f)
        with open(self.processed_paths[0][:-3] + "pairs.json", "r") as f:
            self.pairs = json.load(f)
        # TODO: Remove nodes that have no embeddings in Glove?
        # TODO: Remove nodes that aren't in any of the sampled nodes and their k hops neighborhood?
        # TODO: Subsample graph to start with?

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        x = self.pairs[self.mode]["x"][idx]
        y = self.pairs[self.mode]["y"][idx]
        label = self.pairs[self.mode]["distance"][idx]
        return {
            "x": x,
            "y": y,
            "distance": label,
            # "subgraph": subgraph,
        }

    @property
    def raw_file_names(self) -> List[str]:
        return ["train.txt", "valid.txt", "test.txt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names:
            download_url(f"{self.url}/{filename}", self.raw_dir)

    def get_node_mask(self, mode):
        if mode == "train":
            return list(range(0, self.data.num_nodes // 2))
        elif mode == "val":
            return list(
                range(
                    self.data.num_nodes // 2,
                    self.data.num_nodes // 2 + self.data.num_nodes // 4,
                )
            )
        elif mode == "test":
            return list(
                range(
                    self.data.num_nodes // 2 + self.data.num_nodes // 4,
                    self.data.num_nodes,
                )
            )
        else:
            raise ValueError("Unknown mode. Please use mode in [train, val, test].")
        # return [i for i, x in enumerate(mask) if x]

    def sample_pairs(self):
        """
        Sample pairs of nodes and compute their shortest distance.
        Compute for each mode (train / val / test) a set of pairs with their length.
        train / val / test nodes are based on the masks contained in the self.data object.
        """
        train_indexes = self.get_node_mask("train")
        val_indexes = self.get_node_mask("val")
        test_indexes = self.get_node_mask("test")

        graph = nx.Graph(to_networkx(self.data))
        components = nx.connected_components(graph)
        node2pool = {}
        for component in components:
            componentxtrain = component.intersection(train_indexes)
            componentxval = component.intersection(val_indexes)
            componentxtest = component.intersection(test_indexes)
            # Link node to a component x mask (only nodes of the same component with the same mask should be taken).
            for node in component:
                if node in train_indexes:
                    node2pool[node] = list(componentxtrain)
                elif node in val_indexes:
                    node2pool[node] = list(componentxval)
                elif node in test_indexes:
                    node2pool[node] = list(componentxtest)
                else:
                    raise KeyError("?")

        pairs = {}
        for mode, indexes in zip(
            ["train", "val", "test"], [train_indexes, val_indexes, test_indexes]
        ):
            X = np.random.choice(indexes, self.n_sample).tolist()
            Y = []
            for x in X:
                component = node2pool[
                    x
                ]  # Select the same component as X to guarantee there exists a path.
                y = int(np.random.choice(component))
                Y.append(y)
            pairs[mode] = {
                "x": X,
                "y": Y,
                "distance": [self.shortest_path(graph, x, y) for x, y in zip(X, Y)],
            }
        return pairs

    @staticmethod
    def shortest_path(graph, x, y) -> int:
        try:
            return len(nx.shortest_path(graph, x, y))
        except:
            return -1

    def process(self):
        id2node, node2id, idx = {}, {}, 0

        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path, "r") as f:
                data = f.read().split()

                src = data[::3]
                dst = data[2::3]
                edge_type = data[1::3]

                for i in chain(src, dst):
                    if i not in node2id:
                        node2id[i] = idx
                        id2node[idx] = i
                        idx += 1

                src = [node2id[i] for i in src]
                dst = [node2id[i] for i in dst]
                edge_type = [self.edge2id[i] for i in edge_type]

                srcs.append(torch.tensor(src, dtype=torch.long))
                dsts.append(torch.tensor(dst, dtype=torch.long))
                edge_types.append(torch.tensor(edge_type, dtype=torch.long))

        src = torch.cat(srcs, dim=0)
        dst = torch.cat(dsts, dim=0)
        edge_type = torch.cat(edge_types, dim=0)

        train_mask = torch.zeros(src.size(0), dtype=torch.bool)
        train_mask[: srcs[0].size(0)] = True
        val_mask = torch.zeros(src.size(0), dtype=torch.bool)
        val_mask[srcs[0].size(0) : srcs[0].size(0) + srcs[1].size(0)] = True
        test_mask = torch.zeros(src.size(0), dtype=torch.bool)
        test_mask[srcs[0].size(0) + srcs[1].size(0) :] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        perm = (num_nodes * src + dst).argsort()

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
        )

        if self.pre_transform is not None:
            data = self.pre_filter(data)
        self.data = data
        pairs = self.sample_pairs()

        torch.save(self.collate([data]), self.processed_paths[0])
        with open(self.processed_paths[0][:-3] + "id.json", "w") as f:
            json.dump(id2node, f)
        print(pairs)
        with open(self.processed_paths[0][:-3] + "pairs.json", "w") as f:
            json.dump(pairs, f)
