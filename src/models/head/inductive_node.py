import torch
import torch.nn as nn
from torch_geometric.nn.resolver import activation_resolver

from src.models.act.example import MinMaxNorm
from src.models.network.mlp import MLP


def _apply_index(batch, virtual_node=False):
    pred, true = batch.x, batch.y
    if virtual_node:
        # Remove virtual node
        idx = torch.concat([
            torch.where(batch.batch == i)[0][:-1]
            for i in range(batch.batch.max().item() + 1)
        ])
        pred, true = pred[idx], true[idx]
    return pred, true


class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, act='relu', dropout: float = 0.2):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                 l2_norm=l2_norm, dropout=dropout, act=act)

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, true = _apply_index(batch)
        return pred, true


class COPTInductiveNodeHead(GNNInductiveNodeHead):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, act='relu', dropout: float = 0.2, last_act: str = None, minmax: bool = False,
                 *args, **kwargs):
        super().__init__(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm, l2_norm, act, dropout)
        self.last_act = None if last_act is None else activation_resolver(last_act)
        self.last_norm = None if not minmax is None else MinMaxNorm()

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        batch = batch if self.last_act is None else self.last_act(batch)
        batch = batch if self.last_norm is None else self.last_norm(batch)
        return batch


class COPTInductiveNodeMultiHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks (one MLP per task).

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, act='relu', dropout: float = 0.2, last_act: str = None, minmax: bool = False,
                 *args, **kwargs):
        super().__init__()

        self.layer_post_mp = nn.ModuleList([MLP(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                                l2_norm=l2_norm, dropout=dropout, act=act) for _ in range(dim_out)])

    def forward(self, batch):
        batch.x = torch.hstack([m(batch.x) for m in self.layer_post_mp])
        batch = batch if self.last_act is None else self.last_act(batch)
        batch = batch if self.last_norm is None else self.last_norm(batch)
        return batch
