import torch
from torch_geometric.nn.resolver import activation_resolver

from src.models.act.example import MinMaxNorm
from src.models.network.mlp import MLP


class GNNNodeHead(torch.nn.Module):
    r"""A GNN prediction head for node-level prediction tasks.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool, l2_norm: bool, *args, **kwargs):
        super().__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                 l2_norm=l2_norm, dropout=0.2, act='relu')

    def _apply_index(self, batch):
        x = batch.x
        y = batch.y if 'y' in batch else None

        if 'split' not in batch:
            return x, y

        mask = batch[f'{batch.split}_mask']
        return x[mask], y[mask] if y is not None else None

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


class COPTNodeHead(GNNNodeHead):
    """
    GNN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, graph_pooling: str = 'add', last_act: str = None, minmax: bool = False,
                 *args, **kwargs):
        super(COPTNodeHead, self).__init__(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
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
