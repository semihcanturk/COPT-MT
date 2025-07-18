import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.resolver import activation_resolver

from src.models.head.inductive_node import MinMaxNorm
from src.models.network.mlp import MLP


class GNNGraphHead(torch.nn.Module):
    r"""A GNN prediction head for graph-level prediction tasks.
    A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
    used to transform the pooled graph-level embeddings using an MLP.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, act='relu', dropout: float = 0.2, graph_pooling: str = 'add'):
        super().__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                 l2_norm=l2_norm, dropout=dropout, act=act)

        # Set pooling_fun based on graph_pooling
        if graph_pooling == 'add':
            self.pooling_fun = global_add_pool
        elif graph_pooling == 'max':
            self.pooling_fun = global_max_pool
        elif graph_pooling == 'mean':
            self.pooling_fun = global_mean_pool
        else:
            raise ValueError(f"Unknown graph_pooling: {graph_pooling}")

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


class COPTGraphHead(GNNGraphHead):
    """
    GNN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, graph_pooling: str = 'add', last_act: str = None, minmax: bool = False,
                 *args, **kwargs):
        super(COPTGraphHead, self).__init__(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                            l2_norm=l2_norm, graph_pooling=graph_pooling)

        self.last_act = None if last_act is None else activation_resolver(last_act)
        self.last_norm = None if not minmax is None else MinMaxNorm()

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch.x = self.pooling_fn(batch.x, batch.batch)
        batch.x = self.layer_post_mp(batch.x)
        batch = batch if self.last_act is None else self.last_act(batch)
        batch = batch if self.last_norm is None else self.last_norm(batch)
        return self._apply_index(batch)
