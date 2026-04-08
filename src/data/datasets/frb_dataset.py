import os
import os.path as osp

import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import add_self_loops, from_networkx
import networkx as nx

from src.utils import RankedLogger
from src.utils.utils_graphgym import fun_pbar, parallelize_fn_tqdm


log = RankedLogger(__name__, rank_zero_only=True)


class FRBDataset(InMemoryDataset):
    """
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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, name, train=True, transform=None, pre_transform=None,
                 pre_filter=None):
        self.names = ['RB_Model_Train', '30-15', '40-19', '50-23', '59-26']
        if name not in self.names:
            raise ValueError(f"Unrecognized dataset name {name!r}, available "
                             f"options are: {self.names}")
        self.name = name
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # def download(self):
    #     for gname in self.idx_dict['default']:
    #         download_url(f'https://web.stanford.edu/~yyye/yyye/Gset/G{gname}', self.raw_dir)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, osp.join('frb', self.name), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join('frb', self.name), 'processed')

    @property
    def raw_file_names(self):
        fnames = list()
        if self.train:
            for i in range(2000):
                fnames.append(f'{i}.dimacs')
        else:
            for i in range(1, 6):
                fnames.append(f'frb{self.name}-{i}.dimacs')
        return fnames

    @property
    def processed_file_names(self):
        return ['data.pt']

    def build_graph(self, f):
        G = nx.Graph()

        with open(f, 'r') as file:
            for line in file:
                if line.startswith("c"):  # Comment line in DIMACS file.
                    continue
                elif line.startswith("p"):  # Problem definition, i.e. # nodes and edges.
                    _, _, num_nodes, num_edges = line.strip().split()
                    # Preset graph node labels as there might be isolated ones.
                    G.add_nodes_from(range(1, int(num_nodes) + 1))

                elif line.startswith("e"):
                    _, node1, node2 = line.strip().split()
                    G.add_edge(int(node1), int(node2))

        g_pyg = from_networkx(G)
        return g_pyg

    def process(self):
        log.info("Processing graphs...")
        path_list = [os.path.join(self.raw_dir, f) for f in self.raw_file_names]
        pbar = tqdm(total=len(list(path_list)))
        pbar.set_description(f'Graph generation')
        data_list = [fun_pbar(self.build_graph, f, pbar) for f in path_list]

        log.info("pre transform data...")
        if self.pre_transform is not None:
            pbar_pre = tqdm(total=len(data_list))
            pbar_pre.set_description(f'Graph pre-transform')
            data_list = [fun_pbar(self.pre_transform, data, pbar_pre) for data in data_list]

        log.info("Saving data...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
