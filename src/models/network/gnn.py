from typing import Optional, Union

import torch

from src.models.head import GNNEdgeHead, COPTGraphHead, COPTNodeHead, COPTInductiveNodeHead
from src.models.layer import GeneralMultiLayer
from src.models.stage import GNNStackStage, GNNConcatStage


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
                self.edge_encoder_bn = BatchNorm1dEdge(dim_hid)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
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
        act: Optional[Union[str, torch.nn.Module]] = 'relu',
        last_act: Optional[Union[str, torch.nn.Module]] = 'sigmoid',
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
                final_act=True,
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
            self.GNNHead = COPTInductiveNodeHead
        elif head_type == 'node':
            self.GNNHead = COPTNodeHead
        elif head_type == 'edge':
            self.GNNHead = GNNEdgeHead
        elif head_type == 'graph':
            self.GNNHead = COPTGraphHead
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.post_mp = self.GNNHead(
            dim_in,
            dim_out,
            dim_inner,
            layers_post_mp,
            batch_norm=batch_norm,
            l2_norm=l2_norm,
            act=act,
            dropout=dropout,
            last_act=last_act,
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


class HybridGNN(GNN):
    def __init__(self,
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
                 act: Optional[Union[str, torch.nn.Module]] = 'relu',
                 last_act: Optional[Union[str, torch.nn.Module]] = 'sigmoid',
                 edge_decoding: str = 'concat',
                 graph_pooling: str = 'add',
                 hybrid_stage: str = 'concat',
    ):
        super().__init__(dim_in, dim_out, dim_inner, head_type, encoder, conv, layers_pre_mp, layers_mp, layers_post_mp,
                         stage_type, batch_norm, l2_norm, final_l2_norm, dropout, has_act, act, last_act,
                         edge_decoding, graph_pooling)
        dim_in = dim_inner if layers_pre_mp > 0 else self.encoder.dim_in

        if layers_mp > 0:
            self.mp = GNNConcatStage(
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

        # TODO: decide what to do. maybe sum x_dims
        self.stage = hybrid_stage
        if self.stage == 'sum':
            post_mp_dim_in = self.mp.x_dims[0]
        elif self.stage == 'concat':
            post_mp_dim_in = sum(self.mp.x_dims)
        else:
            raise ValueError('Stage {} is not supported.'.format(self.stage))
        self.post_mp = self.GNNHead(dim_in=post_mp_dim_in,
                                    dim_out=dim_out,
                                    dim_hid=dim_inner,
                                    layers_post_mp=layers_post_mp,
                                    batch_norm=batch_norm,
                                    l2_norm=l2_norm,
                                    act=act,
                                    dropout=dropout,
                                    last_act=last_act,
                                    graph_pooling=graph_pooling,
                                    edge_decoding=edge_decoding,
                                    )

        self.reset_parameters()

    def forward(self, batch):
        if hasattr(self, 'encoder') and self.encoder is not None:
            batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        if hasattr(self, 'mp'):
            batch = self.mp(batch)

        # TODO
        if self.stage == 'sum':
            x_list = torch.stack(batch.x_list, dim=-1)
            x_list = torch.sum(x_list, dim=-1)
        elif self.stage == 'concat':
            x_list = torch.cat(batch.x_list, dim=-1)
        batch.x = x_list
        batch = self.post_mp(batch)

        return batch
