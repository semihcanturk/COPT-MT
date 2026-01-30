from typing import Union

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx

from src.data.data_generation import compute_degrees, compute_eccentricity, compute_cluster_coefficient, compute_triangle_count


@functional_transform('compute_graph_stats')
class ComputeGraphStats(BaseTransform):
    def __init__(
        self,
        stats_list,
        gsn: bool = False,
    ):
        self.stats = stats_list
        self.gsn = gsn

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        g = to_networkx(data)
        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()
        # Derive adjacency matrix
        adj = torch.from_numpy(nx.to_numpy_array(g))
        norm_factor = np.sqrt(g.number_of_nodes()) if self.gsn else 1

        if 'degree' in self.stats:
            data.degree = compute_degrees(adj, log_transform=True)[0] / norm_factor
        if 'eccentricity' in self.stats:
            data.eccentricity = compute_eccentricity(g)[0] / norm_factor
        if 'cluster_coefficient' in self.stats:
            data.cluster_coefficient = compute_cluster_coefficient(g)[0] / norm_factor
        if 'triangle_count' in self.stats:
            data.triangle_count = compute_triangle_count(g)[0] / norm_factor

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(stats={self.stats}, gsn={self.gsn})'


@functional_transform('compute_complement_graph_stats')
class ComputeComplementGraphStats(ComputeGraphStats):
    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        g = to_networkx(data)
        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_c = nx.complement(g)

        # Derive adjacency matrix
        adj = torch.from_numpy(nx.to_numpy_array(g))
        adj_c = torch.from_numpy(nx.to_numpy_array(g_c))
        norm_factor = np.sqrt(g.number_of_nodes()) if self.gsn else 1

        if 'degree' in self.stats:
            data.degree = compute_degrees(adj, log_transform=True)[0] / norm_factor
            data.degree_c = compute_degrees(adj_c, log_transform=True)[0] / norm_factor
        if 'eccentricity' in self.stats:
            data.eccentricity = compute_eccentricity(g)[0] / norm_factor
            data.eccentricity_c = compute_eccentricity(g_c)[0] / norm_factor
        if 'cluster_coefficient' in self.stats:
            data.cluster_coefficient = compute_cluster_coefficient(g)[0] / norm_factor
            data.cluster_coefficient_c = compute_cluster_coefficient(g_c)[0] / norm_factor
        if 'triangle_count' in self.stats:
            data.triangle_count = compute_triangle_count(g)[0] / norm_factor
            data.triangle_count_c = compute_triangle_count(g_c)[0] / norm_factor

        return data


def compute_graph_stats(data, stat_list, gsn=False):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # Derive adjacency matrix
    adj = torch.from_numpy(nx.to_numpy_array(g))
    norm_factor = np.sqrt(g.number_of_nodes()) if gsn else 1

    if 'degree' in stat_list:
        data.degree = compute_degrees(adj, log_transform=True)[0] / norm_factor
    if 'eccentricity' in stat_list:
        data.eccentricity = compute_eccentricity(g)[0] / norm_factor
    if 'cluster_coefficient' in stat_list:
        data.cluster_coefficient = compute_cluster_coefficient(g)[0] / norm_factor
    if 'triangle_count' in stat_list:
        data.triangle_count = compute_triangle_count(g)[0] / norm_factor

    return data

