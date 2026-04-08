import copy
from typing import Optional, Union

import torch

from src.models.head import GNNEdgeHead, COPTGraphHead, COPTNodeHead, COPTInductiveNodeHead
from src.models.layer.general import GeneralMultiLayer
from src.models.layer.gps_layer import GPSLayer


class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former (GraphGPS model).

    This model combines local message-passing GNN layers with global attention
    transformer layers, following the GraphGPS architecture.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        dim_inner (int): Hidden dimension for the model.
        head_type (str): Type of head to use. Options: 'node', 'inductive_node', 'edge', 'graph'.
        encoder (Optional[torch.nn.Module]): Feature encoder for node/edge features.
        layers_pre_mp (int): Number of layers before message-passing. Default: 1.
        layers_mp (int): Number of GPS layers. Default: 3.
        layers_post_mp (int): Number of layers after message-passing. Default: 2.
        layer_type (str): Layer type specification in format 'local_gnn+global_model'.
            Examples: 'GatedGCN+Transformer', 'GINE+Performer'. Default: 'GatedGCN+Transformer'.
        num_heads (int): Number of attention heads. Default: 4.
        pna_degrees (Optional[list]): Degrees for PNA aggregator. Required if using PNA.
        equivstable_pe (bool): Whether to use equivariant stable positional encoding. Default: False.
        dropout (float): Dropout rate. Default: 0.0.
        attn_dropout (float): Attention dropout rate. Default: 0.0.
        layer_norm (bool): Whether to use layer normalization. Default: False.
        batch_norm (bool): Whether to use batch normalization. Default: True.
        bigbird_cfg (Optional[dict]): Configuration for BigBird if using BigBird global model.
        log_attn_weights (bool): Whether to log attention weights. Default: False.
        batch_norm_pre_mp (bool): Whether to use batch norm in pre_mp. Default: True.
        l2_norm_pre_mp (bool): Whether to use L2 norm in pre_mp. Default: False.
        dropout_pre_mp (float): Dropout rate in pre_mp. Default: 0.2.
        act (Union[str, torch.nn.Module]): Activation function. Default: 'relu'.
        last_act (Optional[Union[str, torch.nn.Module]]): Last activation function. Default: None.
        graph_pooling (str): Graph pooling method. Options: 'add', 'max', 'mean'. Default: 'add'.
        edge_decoding (str): Edge decoding method. Default: 'concat'.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_inner: int,
        head_type: str = 'inductive_node',
        encoder: Optional[torch.nn.Module] = None,
        layers_pre_mp: int = 1,
        layers_mp: int = 3,
        layers_post_mp: int = 2,
        local_model: Optional[torch.nn.Module] = None,
        self_attn: Optional[torch.nn.Module] = None,
        num_heads: int = 4,
        pna_degrees: Optional[list] = None,
        equivstable_pe: bool = False,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        bigbird_cfg: Optional[dict] = None,
        log_attn_weights: bool = False,
        batch_norm_pre_mp: bool = True,
        l2_norm_pre_mp: bool = False,
        dropout_pre_mp: float = 0.2,
        act: Union[str, torch.nn.Module] = 'relu',
        last_act: Optional[Union[str, torch.nn.Module]] = 'sigmoid',
        graph_pooling: str = 'add',
        edge_decoding: str = 'concat',
    ):
        super().__init__()
        self.encoder = encoder
        dim_in = self.encoder.dim_in if encoder is not None else dim_in

        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(
                layer='linear',
                in_dim=dim_in,
                out_dim=dim_inner,
                hid_dim=dim_inner,
                num_layers=layers_pre_mp,
                batch_norm=batch_norm_pre_mp,
                l2_norm=l2_norm_pre_mp,
                dropout=dropout_pre_mp,
                act=act,
                final_act=True,
            )
            dim_in = dim_inner

        assert dim_inner == dim_in, \
            "The inner and hidden dims must match."

        # Convert act to string for GPSLayer (if using string-based construction)
        if isinstance(act, torch.nn.Module):
            act_class_name = act.__class__.__name__
            if act_class_name == 'LeakyReLU':
                act_str = 'leaky_relu'
            elif act_class_name == 'ReLU':
                act_str = 'relu'
            elif act_class_name == 'ELU':
                act_str = 'elu'
            elif act_class_name == 'GELU':
                act_str = 'gelu'
            else:
                act_str = 'relu'
        else:
            act_str = act

        layers = []
        for _ in range(layers_mp):
            layer_local_model = copy.deepcopy(local_model) if local_model is not None else None
            layer_self_attn = copy.deepcopy(self_attn) if self_attn is not None else None

            layers.append(GPSLayer(
                dim_h=dim_inner,
                local_model=layer_local_model,
                self_attn=layer_self_attn,
                equivstable_pe=equivstable_pe,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                log_attn_weights=log_attn_weights,
                act=act_str,
            ))
        self.layers = torch.nn.Sequential(*layers)

        # Assign the appropriate head based on head_type
        if head_type == 'inductive_node':
            GNNHead = COPTInductiveNodeHead
        elif head_type == 'node':
            GNNHead = COPTNodeHead
        elif head_type == 'edge':
            GNNHead = GNNEdgeHead
        elif head_type == 'graph':
            GNNHead = COPTGraphHead
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.post_mp = GNNHead(
            dim_in=dim_inner,
            dim_out=dim_out,
            dim_hid=dim_inner,
            layers_post_mp=layers_post_mp,
            batch_norm=batch_norm,
            l2_norm=False,
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
