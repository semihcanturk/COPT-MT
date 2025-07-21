import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx

from src.data.datasets.synthetic import SyntheticDataset


class BADataset(SyntheticDataset):
    def __init__(self, root, name, n_min, n_max, num_edges, num_samples, multiprocessing=False, num_workers=0, transform=None, pre_transform=None):
        self.n_min = n_min
        self.n_max = n_max
        self.num_edges = num_edges
        super().__init__(root, 'ba', name, num_samples, multiprocessing, num_workers, transform, pre_transform)


    def create_graph(self, idx):
        n = np.random.randint(self.n_min, self.n_max + 1)
        g = nx.barabasi_albert_graph(n, self.num_edges)

        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_pyg = from_networkx(g)
        return g_pyg
