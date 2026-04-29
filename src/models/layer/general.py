import copy
import functools
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.resolver import activation_resolver


class GeneralLayer(torch.nn.Module):
    def __init__(
            self,
            layer: Union[str, torch.nn.Module],
            in_dim: int,
            out_dim: int,
            batch_norm: bool,
            l2_norm: bool,
            dropout: float,
            act: Optional[Union[str, torch.nn.Module]],
            ffn: bool = False,
            **kwargs,
    ):
        super().__init__()
        if isinstance(layer, str):
            assert (layer.lower() == 'linear')
            self.lin = True
        else:
            self.lin = False

        if self.lin:
            self.layer = Linear(in_dim, out_dim, **kwargs)
        else:
            if isinstance(layer.layer, functools.partial):
                self.layer = LayerWrapper(layer.layer(in_channels=in_dim, out_channels=out_dim), edge_attr=True)
            else:
                self.layer = copy.deepcopy(layer)

        self.batch_norm = batch_norm
        self.l2_norm = l2_norm

        post_layers = []
        if batch_norm:
            post_layers.append(
                torch.nn.BatchNorm1d(out_dim, eps=1e-5, momentum=0.1))
        if dropout > 0:
            post_layers.append(torch.nn.Dropout(p=dropout, inplace=False))
        if act is not None:
            if isinstance(act, torch.nn.Module):
                post_layers.append(act)
            else:
                post_layers.append(activation_resolver(act))
        self.post_layer = nn.Sequential(*post_layers)
        self.ffn = False if self.lin else ffn

        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(out_dim)
            self.ff_linear1 = nn.Linear(out_dim, out_dim * 2)
            self.ff_linear2 = nn.Linear(out_dim * 2, out_dim)
            self.act_fn_ff = activation_resolver(act)
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        # return x
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        batch = self.layer(batch)
        x = batch if isinstance(batch, torch.Tensor) else batch.x

        x = self.post_layer(x)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)

        if self.ffn:
            if self.batch_norm:
                x = self.norm1_local(x)
            x = x + self._ff_block(x)
            if self.batch_norm:
                x = self.norm2(x)

        if isinstance(batch, torch.Tensor):
            batch = x
        else:
            batch.x = x

        return batch


class GeneralMultiLayer(torch.nn.Module):
    def __init__(
            self,
            layer: Union[str, torch.nn.Module],
            in_dim: int,
            out_dim: int,
            hid_dim: Optional[int],
            num_layers: int,
            batch_norm: bool,
            l2_norm: bool,
            dropout: float,
            act: Optional[Union[str, torch.nn.Module]],
            final_act: bool,
            **kwargs,
    ) -> None:
        super().__init__()
        hid_dim = hid_dim or out_dim

        for i in range(num_layers):
            d_in = in_dim if i == 0 else hid_dim
            d_out = out_dim if i == num_layers - 1 else hid_dim
            layer_obj = GeneralLayer(
                layer=layer,
                in_dim=d_in,
                out_dim=d_out,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=None if i == num_layers - 1 and not final_act else act,
                **kwargs,
            )
            self.add_module(f'Layer_{i}', layer_obj)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.model = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class LayerWrapper(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, edge_attr: bool = False, complement=False):
        super().__init__()
        self.layer = layer
        self.edge_attr = edge_attr
        self.complement = complement

    def forward(self, batch):
        if self.edge_attr and hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.x = self.layer(batch.x, batch.edge_index, batch.edge_attr)
        else:
            edge_index = batch.edge_index_c if self.complement else batch.edge_index
            batch.x = self.layer(batch.x, edge_index)
        return batch