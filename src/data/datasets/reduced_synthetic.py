"""Shared base + helpers for wrapper datasets that apply CO reductions
from `src.utils.reductions` to a source synthetic dataset.

Subclasses (see `reduced_rb_dataset.py`, `reduced_ba_dataset.py`) bind a
specific source dataset class and forward its construction kwargs.
"""
import os.path as osp
from typing import Callable, Dict, Type

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm

from src.utils import RankedLogger
from src.utils.reductions import (
    maxclique_to_mds,
    maxcut_to_mds,
    mis_to_mds,
    mvc_to_mds,
)


log = RankedLogger(__name__, rank_zero_only=True)


_REDUCTIONS: Dict[str, Callable[[nx.Graph], tuple]] = {
    "mis_to_mds": mis_to_mds,
    "mvc_to_mds": mvc_to_mds,
    "maxclique_to_mds": maxclique_to_mds,
    "maxcut_to_mds": maxcut_to_mds,
}


def reduce_data(src_data: Data, reduction_fn: Callable[[nx.Graph], tuple]) -> Data:
    """Apply a reduction to a single PyG Data and return the reduced Data
    with attached recovery tensors (see `_build_recovery_tensors`).
    """
    G = to_networkx(src_data, to_undirected=True)
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    G_red, meta = reduction_fn(G)
    gadget_meta = meta["gadget_metadata"]

    G_relabeled = nx.convert_node_labels_to_integers(
        G_red, label_attribute="orig_id"
    )

    N = G_relabeled.number_of_nodes()
    G_clean = nx.Graph()
    G_clean.add_nodes_from(range(N))
    G_clean.add_edges_from(G_relabeled.edges())

    data = from_networkx(G_clean)
    data.num_nodes = N

    for key, value in _build_recovery_tensors(G_relabeled, gadget_meta).items():
        setattr(data, key, value)

    return data


def _build_recovery_tensors(G_relabeled: nx.Graph, gadget_meta: Dict) -> Dict[str, torch.Tensor]:
    """Convert the gadget's metadata dict into per-graph tensors that
    PyG can collate.

    Tensor semantics:
      n_original           — scalar long, count of source-task vertices.
      original_mask        — bool [N], True for source-task vertices.
      is_isolated          — bool [N], True for vertices isolated in the
                             gadget's input graph (forced into any DS).
      mid_edge_index       — long [M], indices of mid-edge vertices.
      mid_edge_endpoints_index — long [2, M], endpoint indices in the
                             reduced graph for each mid-edge gadget vertex
                             (edge-index convention so PyG auto-offsets on
                             batching).
    """
    N = G_relabeled.number_of_nodes()
    new_id_of = {
        attrs["orig_id"]: new_id
        for new_id, attrs in G_relabeled.nodes(data=True)
    }

    original_vertices = gadget_meta["original_vertices"]
    isolated = gadget_meta["isolated_vertices"]
    mid_edge_map = gadget_meta["mid_edge_map"]

    original_mask = torch.zeros(N, dtype=torch.bool)
    for v in original_vertices:
        original_mask[new_id_of[v]] = True

    is_isolated = torch.zeros(N, dtype=torch.bool)
    for v in isolated:
        is_isolated[new_id_of[v]] = True

    if mid_edge_map:
        mid_edge_index = torch.tensor(
            [new_id_of[w] for w in mid_edge_map],
            dtype=torch.long,
        )
        mid_edge_endpoints_index = torch.tensor(
            [[new_id_of[u] for (u, _) in mid_edge_map.values()],
             [new_id_of[v] for (_, v) in mid_edge_map.values()]],
            dtype=torch.long,
        )
    else:
        mid_edge_index = torch.empty(0, dtype=torch.long)
        mid_edge_endpoints_index = torch.empty(2, 0, dtype=torch.long)

    return {
        "n_original": torch.tensor(len(original_vertices), dtype=torch.long),
        "original_mask": original_mask,
        "is_isolated": is_isolated,
        "mid_edge_index": mid_edge_index,
        "mid_edge_endpoints_index": mid_edge_endpoints_index,
    }


class ReducedSyntheticDataset(InMemoryDataset):
    """Base class for reduction-wrapper datasets. Subclasses must set
    `source_cls` and `source_format`, and forward source-dataset kwargs
    via `__init__`.
    """

    source_cls: Type[InMemoryDataset]  # set by subclass
    source_format: str                  # set by subclass

    def __init__(
        self,
        root: str,
        name: str,
        reduction: str,
        source_kwargs: Dict,
        transform=None,
        pre_transform=None,
    ):
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"Unknown reduction '{reduction}'. "
                f"Choose from {list(_REDUCTIONS)}."
            )
        self.name = name
        self.reduction = reduction
        self._source_root = root
        self._source_kwargs = source_kwargs

        super().__init__(
            osp.join(root, f"{self.source_format}-reduced", reduction),
            transform,
            pre_transform,
        )
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        log.info(
            f"Building reduced dataset (source={self.source_format}, "
            f"reduction='{self.reduction}', name='{self.name}')..."
        )
        src = self.source_cls(
            root=self._source_root,
            name=self.name,
            pre_transform=None,
            transform=None,
            **self._source_kwargs,
        )

        fn = _REDUCTIONS[self.reduction]
        data_list = [
            reduce_data(src.get(i), fn)
            for i in tqdm(range(len(src)), desc=f"Reducing ({self.reduction})")
        ]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        log.info("Saving reduced data...")
        self.save(data_list, self.processed_paths[0])
