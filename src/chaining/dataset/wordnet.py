from itertools import chain
from typing import Callable, List, Optional

import torch
import json

from torch_geometric.data import Data, InMemoryDataset, download_url

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

    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/WN18RR/text')

    edge2id = {
        '_also_see': 0,
        '_derivationally_related_form': 1,
        '_has_part': 2,
        '_hypernym': 3,
        '_instance_hypernym': 4,
        '_member_meronym': 5,
        '_member_of_domain_region': 6,
        '_member_of_domain_usage': 7,
        '_similar_to': 8,
        '_synset_domain_topic_of': 9,
        '_verb_group': 10,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[0][:-2] + "json", "r") as f:
            self.id2node = json.load(f)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self):
        id2node, node2id, idx = {}, {}, 0

        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path, 'r') as f:
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
                text_label = [id2node[i] for i in range(len(id2node))]

                srcs.append(torch.tensor(src, dtype=torch.long))
                dsts.append(torch.tensor(dst, dtype=torch.long))
                edge_types.append(torch.tensor(edge_type, dtype=torch.long))

        src = torch.cat(srcs, dim=0)
        dst = torch.cat(dsts, dim=0)
        edge_type = torch.cat(edge_types, dim=0)

        train_mask = torch.zeros(src.size(0), dtype=torch.bool)
        train_mask[:srcs[0].size(0)] = True
        val_mask = torch.zeros(src.size(0), dtype=torch.bool)
        val_mask[srcs[0].size(0):srcs[0].size(0) + srcs[1].size(0)] = True
        test_mask = torch.zeros(src.size(0), dtype=torch.bool)
        test_mask[srcs[0].size(0) + srcs[1].size(0):] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        perm = (num_nodes * src + dst).argsort()

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask, num_nodes=num_nodes)
        
        if self.pre_transform is not None:
            data = self.pre_filter(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        with open(self.processed_paths[0][:-2] + "json", "w") as f:
            self.id2node = json.dump(id2node, f)

