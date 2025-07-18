import logging
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
import torch_geometric.nn.conv as conv_modules

from src.models.head.inductive_node import minmax_norm_pyg
from src.models.layer.linear import Linear


class FeatureEncoder(torch.nn.Module):
    r"""Encodes node and edge features, given the specified input dimension and
    the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input feature dimension.
    """
    def __init__(self, dim_in: int, dim_hid: int,
                 node_encoder: Optional[torch.nn.Module] = None, node_encoder_bn: bool = False,
                 edge_encoder: Optional[torch.nn.Module] = None, edge_encoder_bn: bool = False):
        super().__init__()
        self.dim_in = dim_in
        if node_encoder:
            # Encode integer node features via `torch.nn.Embedding`:
            self.node_encoder = node_encoder
            if node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(dim_hid)
            # Update `dim_in` to reflect the new dimension fo the node features
            self.dim_in = dim_hid
        if edge_encoder:
            # Encode integer edge features via `torch.nn.Embedding`:
            self.edge_encoder = edge_encoder
            if edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(dim_hid)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GeneralLayer(torch.nn.Module):
    def __init__(
        self,
        layer: Union[str, torch.nn.Module],
        in_dim: int,
        out_dim: int,
        batch_norm: bool,
        l2_norm: bool,
        dropout: float,
        act: Optional[str],
        ffn: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(layer, str):
            assert(layer.lower() == 'linear')
            self.lin = True
        else:
            self.lin = False
        self.layer = Linear(in_dim, out_dim, **kwargs) if self.lin else layer
        self.batch_norm = batch_norm
        self.l2_norm = l2_norm
        
        post_layers = []
        if batch_norm:
            post_layers.append(
                torch.nn.BatchNorm1d(out_dim, eps=1e-5, momentum=0.1))
        if dropout > 0:
            post_layers.append(torch.nn.Dropout(p=dropout, inplace=False))
        if act is not None:
            post_layers.append(activation_resolver(act))
        self.post_layer = nn.Sequential(*post_layers)
        self.ffn = False if self.lin else ffn

        if self.ffn:
            # Feed Forward block.
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(out_dim)
            self.ff_linear1 = nn.Linear(out_dim, out_dim*2)
            self.ff_linear2 = nn.Linear(out_dim*2, out_dim)
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
        act: str,
        final_act: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        hid_dim = hid_dim or out_dim

        for i in range(num_layers):
            d_in = in_dim if i == 0 else hid_dim
            d_out = out_dim if i == num_layers - 1 else hid_dim
            layer = GeneralLayer(
                layer=layer,
                in_dim=d_in,
                out_dim=d_out,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=None if i == num_layers - 1 and not final_act else act,
                **kwargs,
            )
            self.add_module(f'Layer_{i}', layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


class BatchNorm1dNode(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_dim: Optional[int],
        num_layers: int,
        batch_norm: bool = True,
        l2_norm: bool = True,
        dropout: float = 0.2,
        act: str = 'relu',
        **kwargs,
    ):
        super().__init__()
        hid_dim = hid_dim or in_dim

        layers = []
        if num_layers > 1:
            layer = GeneralMultiLayer(
                'linear',
                in_dim,
                hid_dim,
                hid_dim,
                num_layers - 1,
                batch_norm,
                l2_norm,
                dropout,
                act,
                final_act=True,
                **kwargs,
            )
            layers.append(layer)
        layers.append(Linear(hid_dim, out_dim, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


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
        act: Optional[str] = 'relu',
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
            layer = GeneralLayer(conv, d_in, out_dim,
                                 batch_norm, l2_norm, dropout, act)
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


class IdentityHead(torch.nn.Module):
    def forward(self, batch):
        return batch.x, batch.y


class GNN(torch.nn.Module):
    r"""The Graph Positional and Structural Encoder (GPSE) model from the
    `"Graph Positional and Structural Encoder"
    <https://arxiv.org/abs/2307.07107>`_ paper.

    The GPSE model consists of a (1) deep GNN that consists of stacked
    message passing layers, and a (2) prediction head to predict pre-computed
    positional and structural encodings (PSE).
    When used on downstream datasets, these prediction heads are removed and
    the final fully-connected layer outputs are used as learned PSE embeddings.

    GPSE also provides a static method :meth:`from_pretrained` to load
    pre-trained GPSE models trained on a variety of molecular datasets.

    .. code-block:: python

        from torch_geometric.nn import GPSE, GPSENodeEncoder,
        from torch_geometric.transforms import AddGPSE
        from torch_geometric.nn.models.gpse import precompute_GPSE

        gpse_model = GPSE.from_pretrained('molpcba')

        # Option 1: Precompute GPSE encodings in-place for a given dataset
        dataset = ZINC(path, subset=True, split='train')
        precompute_gpse(gpse_model, dataset)

        # Option 2: Use the GPSE model with AddGPSE as a pre_transform to save
        # the encodings
        dataset = ZINC(path, subset=True, split='train',
                       pre_transform=AddGPSE(gpse_model, vn=True,
                       rand_type='NormalSE'))

    Both approaches append the generated encodings to the :obj:`pestat_GPSE`
    attribute of :class:`~torch_geometric.data.Data` objects. To use the GPSE
    encodings for a downstream task, one may need to add these encodings to the
    :obj:`x` attribute of the :class:`~torch_geometric.data.Data` objects. To
    do so, one can use the :class:`GPSENodeEncoder` provided to map these
    encodings to a desired dimension before appending them to :obj:`x`.

    Let's say we have a graph dataset with 64 original node features, and we
    have generated  GPSE encodings of dimension 32, i.e.
    :obj:`data.pestat_GPSE` = 32. Additionally, we want to use a GNN with an
    inner dimension of 128. To do so, we can map the 32-dimensional GPSE
    encodings to a higher dimension of 64, and then append them to the :obj:`x`
    attribute of the :class:`~torch_geometric.data.Data` objects to obtain a
    128-dimensional node feature representation.
    :class:`~torch_geometric.nn.GPSENodeEncoder` handles both this mapping and
    concatenation to :obj:`x`, the outputs of which can be used as input to a
    GNN:

    .. code-block:: python

        encoder = GPSENodeEncoder(dim_emb=128, dim_pe_in=32, dim_pe_out=64,
                                  expand_x=False)
        gnn = GNN(dim_in=128, dim_out=128, num_layers=4)

        for batch in loader:
            batch = encoder(batch)
            batch = gnn(batch)
            # Do something with the batch, which now includes 128-dimensional
            # node representations


    Args:
        dim_in (int, optional): Input dimension. (default: :obj:`20`)
        dim_out (int, optional): Output dimension. (default: :obj:`51`)
        dim_inner (int, optional): Width of the encoder layers.
            (default: :obj:`512`)
        layer_type (str, optional): Type of graph convolutional layer for
            message-passing. (default: :obj:`resgatedgcnconv`)
        layers_pre_mp (int, optional): Number of MLP layers before
            message-passing. (default: :obj:`1`)
        layers_mp (int, optional): Number of layers for message-passing.
            (default: :obj:`20`)
        layers_post_mp (int, optional): Number of MLP layers after
            message-passing. (default: :obj:`2`)
        num_node_targets (int, optional): Number of individual PSEs used as
            node-level targets in pretraining :class:`GPSE`.
            (default: :obj:`51`)
        num_graph_targets (int, optional): Number of graph-level targets used
            in pretraining :class:`GPSE`. (default: :obj:`11`)
        stage_type (str, optional): The type of staging to apply. Possible
            values are: :obj:`skipsum`, :obj:`skipconcat`. Any other value will
            default to no skip connections. (default: :obj:`skipsum`)
        batch_norm (bool, optional): Whether to apply batch normalization in the
            layer. (default: :obj:`True`)
        final_l2_norm (bool, optional): Whether to apply L2 normalization to the
            outputs. (default: :obj:`True`)
        l2_norm (bool, optional): Whether to apply L2 normalization after
        the layer. (default: :obj:`True`)
        dropout (float, optional): Dropout ratio at layer output.
            (default: :obj:`0.2`)
        has_act (bool, optional): Whether has activation after the layer.
            (default: :obj:`True`)
        final_act (bool, optional): Whether to apply activation after the layer
            stack. (default: :obj:`True`)
        act (str, optional): Activation to apply to layer output if
            :obj:`has_act` is :obj:`True`. (default: :obj:`relu`)
        virtual_node (bool, optional): Whether a virtual node is added to
            graphs in :class:`GPSE` computation. (default: :obj:`True`)
        graph_pooling (str, optional): Type of graph pooling applied before
            post_mp. Options are :obj:`add`, :obj:`max`, :obj:`mean`.
            (default: :obj:`add`)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_inner: int,
        head_type: str = 'node',
        encoder: Optional[torch.nn.Module] = None,
        conv: Optional[torch.nn.Module] = None,
        layers_pre_mp: int = 1,
        layers_mp: int = 5,
        layers_post_mp: int = 2,
        stage_type: str = 'skipsum',
        batch_norm: bool = True,
        l2_norm: bool = True,
        final_l2_norm: bool = True,
        dropout: float = 0.2,
        has_act: bool = True,
        final_act: bool = True,
        act: str = 'relu',
        edge_decoding: str = 'concat',
        graph_pooling: str = 'add',
    ):
        super().__init__()
        self.encoder = encoder
        dim_in = self.encoder.dim_in

        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(
                layer='linear',
                in_dim=dim_in,
                out_dim=dim_inner,
                hid_dim=dim_inner,
                num_layers=layers_pre_mp,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=act,
                final_act=final_act,
            )
            dim_in = dim_inner
        if layers_mp > 0:
            self.mp = GNNStackStage(
                in_dim=dim_in,
                out_dim=dim_inner,
                num_layers=layers_mp,
                conv=conv,
                stage_type=stage_type,
                final_l2_norm=final_l2_norm,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=act if has_act else None,
            )

        # Assign the appropriate head based on head_type
        if head_type == 'inductive_node':
            GNNHead = GNNInductiveNodeHead
        elif head_type == 'node':
            GNNHead = GNNNodeHead
        elif head_type == 'edge':
            GNNHead = GNNEdgeHead
        elif head_type == 'graph':
            GNNHead = GNNGraphHead
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.post_mp = GNNHead(
            dim_in,
            dim_out,
            dim_inner,
            layers_post_mp,
            batch_norm=batch_norm,
            l2_norm=l2_norm,
            graph_pooling=graph_pooling,
            edge_decoding=edge_decoding,
        )

        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.graphgym.init import init_weights
        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


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


class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool, l2_norm: bool,
                 last_act: str = None, minmax: bool = False, *args, **kwargs):
        super().__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                 l2_norm=l2_norm, dropout=0.2, act='relu')

        self.last_act = None if last_act is None else activation_resolver(last_act)
        self.last_norm = None if not minmax is None else minmax_norm_pyg

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        batch = batch if self.last_act is None else self.last_act(batch)
        batch = batch if self.last_norm is None else self.last_norm(batch)
        return batch


class GNNEdgeHead(torch.nn.Module):
    r"""A GNN prediction head for edge-level/link-level prediction tasks.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool, l2_norm: bool, edge_decoding='concat', *args, **kwargs):
        super().__init__()
        # Module to decode edges from node embeddings:
        self.edge_decoding = edge_decoding
        if edge_decoding == 'concat':
            self.layer_post_mp = MLP(dim_in * 2, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                     l2_norm=l2_norm, dropout=0.2, act='relu')
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(f"Binary edge decoding "
                                 f"'{edge_decoding}' is used for "
                                 f"multi-class classification")
            self.layer_post_mp = MLP(dim_in, dim_in, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                     l2_norm=l2_norm, dropout=0.2, act='relu')
            if edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif edge_decoding == 'cosine_similarity':
                self.decode_module = torch.nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError(f"Unknown edge decoding "
                                 f"'{edge_decoding}'")

    def _apply_index(self, batch):
        index = f'{batch.split}_edge_index'
        label = f'{batch.split}_edge_label'
        return batch.x[batch[index]], batch[label]

    def forward(self, batch):
        if self.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, label


class GNNGraphHead(torch.nn.Module):
    r"""A GNN prediction head for graph-level prediction tasks.
    A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
    used to transform the pooled graph-level embeddings using an MLP.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool, l2_norm: bool, graph_pooling: str, *args, **kwargs):
        super().__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, dim_hid, layers_post_mp, batch_norm=batch_norm,
                                 l2_norm=l2_norm, dropout=0.2, act='relu')

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
