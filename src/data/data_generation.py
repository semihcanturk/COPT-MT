from typing import Union, Tuple, List, Dict, Any

import torch
import networkx as nx
import dwave_networkx as dnx
import dimod



def compute_maxcut(g):
    adj = torch.from_numpy(nx.to_numpy_array(g))
    num_nodes = adj.size(0)

    cut = dnx.maximum_cut(g, dimod.SimulatedAnnealingSampler())
    cut_size = max(len(cut), g.number_of_nodes() - len(cut))
    cut_binary = torch.zeros((num_nodes, 1), dtype=torch.int)
    cut_binary[torch.tensor(list(cut))] = 1

    return cut_size, cut_binary


def compute_degrees(
    adj: torch.Tensor,
    log_transform: bool = True
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = adj.sum(1).unsqueeze(-1)
    if log_transform:
        feat = torch.log(feat)

    return feat, base_level


def compute_eccentricity(
    graph: nx.Graph,
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.Tensor(list(nx.eccentricity(graph).values())).unsqueeze(-1)
    
    return feat, base_level


def compute_cluster_coefficient(
    graph: nx.Graph,
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.Tensor(list(nx.clustering(graph).values())).unsqueeze(-1)
    
    return feat, base_level


def compute_triangle_count(
    graph: nx.Graph,
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.Tensor(list(nx.triangles(graph).values())).unsqueeze(-1)
    
    return feat, base_level


def set_constant_feat(
    adj: torch.Tensor,
    norm: bool = True
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.ones(adj.size(0)).unsqueeze(-1)
    if norm:
        feat /= adj.size(0)

    return feat, base_level


def transfer_feat_level(
    feat: torch.tensor, in_level: str, out_level: str
) -> List[torch.Tensor]:
    
    if in_level == "node":
        if out_level == "node":
            tag = "node_"
        else:
            raise NotImplementedError()
    
    else:
        raise NotImplementedError()

    return feat, tag