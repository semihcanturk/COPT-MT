from typing import Optional, Union

import torch
import torch.nn.functional as F

from src.models.layer import GeneralLayer


class GNNStackStage(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        conv: torch.nn.Module,
        stage_type: str = 'skipsum',
        final_l2_norm: bool = True,
        batch_norm: bool = True,
        l2_norm: bool = True,
        dropout: float = 0.2,
        act: Optional[Union[str, torch.nn.Module]] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.final_l2_norm = final_l2_norm

        for i in range(num_layers):
            if stage_type == 'skipconcat':
                if i == 0:
                    d_in = in_dim
                else:
                    d_in = in_dim + i * out_dim
            else:
                d_in = in_dim if i == 0 else out_dim
            layer = GeneralLayer(conv, d_in, out_dim, batch_norm, l2_norm, dropout, act)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if self.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif self.stage_type == 'skipconcat' and i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)

        if self.final_l2_norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)

        return batch


class GNNConcatStage(torch.nn.Module):
    r"""Stacks a number of GNN layers.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        num_layers (int): The number of layers.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_layers: int,
                 conv: torch.nn.Module,
                 stage_type: str = 'skipsum',
                 final_l2_norm: bool = True,
                 batch_norm: bool = True,
                 l2_norm: bool = True,
                 dropout: float = 0.2,
                 act: Optional[Union[str, torch.nn.Module]] = 'relu',
                 **kwargs
                 ):

        super().__init__()
        self.num_layers = num_layers
        self.x_dims = list()
        self.stage_type = stage_type

        for i in range(num_layers):
            if stage_type == 'skipconcat':
                d_in = in_dim if i == 0 else in_dim + i * out_dim
            else:
                d_in = in_dim if i == 0 else out_dim
            self.x_dims.append(d_in)
            layer = GeneralLayer(conv, d_in, out_dim, batch_norm, l2_norm, dropout, act, **kwargs)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        x_list = []
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            x_list.append(batch.x)
            if self.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (self.stage_type == 'skipconcat'
                  and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        batch.x_list = x_list
        return batch