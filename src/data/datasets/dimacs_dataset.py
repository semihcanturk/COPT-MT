import os
import torch
import hashlib
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx


def parse_dimacs(filepath: str) -> nx.Graph:
    g = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                n_nodes = int(parts[2])
                g.add_nodes_from(range(1, n_nodes + 1))
            elif parts[0] == 'e':
                u, v = int(parts[1]), int(parts[2])
                g.add_edge(u, v)
    return g


class DIMACSDataset(InMemoryDataset):
    def __init__(self, root, name, instance_names=None, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.instance_names = (
            [n if n.endswith('.txt') else f'{n}.txt' for n in instance_names]
            if instance_names is not None else None
        )
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        if self.instance_names and len(self.instance_names) == 1:
            folder = self.instance_names[0].replace('.txt', '')
        elif self.instance_names:
            suffix = hashlib.md5(''.join(sorted(self.instance_names)).encode()).hexdigest()[:8]
            folder = f'{self.name}_{suffix}'
        else:
            folder = self.name
        return os.path.join(self.root, 'processed', folder)

    @property
    def raw_file_names(self) -> list[str]:
        all_files = sorted([
            f for f in os.listdir(self.raw_dir)
            if f.endswith('.txt') or f.endswith('.clq') or f.endswith('.col')
        ])
        if self.instance_names is not None:
            all_files = [f for f in all_files if f in self.instance_names]
        return all_files

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def process(self):
        data_list = []
        for fname in self.raw_file_names:
            fpath = os.path.join(self.raw_dir, fname)
            g = parse_dimacs(fpath)

            if isinstance(g, nx.DiGraph):
                g = g.to_undirected()

            data = from_networkx(g)
            # Remove this — Constant() in pre_transforms_in_memory will set x
            # data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
            data.instance_name = fname

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    @property
    def num_node_features(self) -> int:
        return 1