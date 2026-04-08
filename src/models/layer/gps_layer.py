import torch
import torch.nn as nn

import torch_geometric.nn as pygnn
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch

# from src.models.layer.gine_conv_layer import GINEConvESLapPE


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    
    This layer accepts pre-constructed local_model and self_attn modules via Hydra.
    For string-based construction, use GPSLayerFromString instead.
    """

    def __init__(self, dim_h,
                 local_model=None, self_attn=None,
                 equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 log_attn_weights=False, act='relu'):
        super().__init__()

        if local_model is None and self_attn is None:
            raise ValueError("At least one of local_model or self_attn must be provided")

        self.dim_h = dim_h
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = activation_resolver(act)
        self.log_attn_weights = log_attn_weights

        # Use pre-constructed modules
        self.local_model = local_model
        self.self_attn = self_attn

        if log_attn_weights:
            # Check if it's a MultiheadAttention for log_attn_weights support
            if self_attn is None or not isinstance(self_attn, torch.nn.MultiheadAttention):
                raise NotImplementedError(
                    "Logging of attention weights is only supported for "
                    "MultiheadAttention (Transformer) global attention model."
                )

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            # Standard PyG conv interface
            if self.equivstable_pe and hasattr(batch, 'pe_EquivStableLapPE'):
                h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                           batch.pe_EquivStableLapPE)
            elif batch.edge_attr is not None:
                h_local = self.local_model(h, batch.edge_index, edge_attr=batch.edge_attr)
            elif batch.edge_weight is not None:
                h_local = self.local_model(h, batch.edge_index, edge_weight=batch.edge_weight)
            else:
                h_local = self.local_model(h, batch.edge_index)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            # Detect model type by checking instance type
            if isinstance(self.self_attn, torch.nn.MultiheadAttention):
                # Transformer (MultiheadAttention)
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            else:
                # Fallback: try standard call (assume it returns (output, mask) or just output)
                try:
                    result = self.self_attn(h_dense, mask=mask)
                    if isinstance(result, tuple):
                        h_attn = result[0][mask] if len(result) > 0 else result[mask]
                    else:
                        h_attn = result[mask]
                except (TypeError, KeyError):
                    # Try without mask
                    result = self.self_attn(h_dense)
                    if isinstance(result, tuple):
                        h_attn = result[0][mask] if len(result) > 0 else result[mask]
                    else:
                        h_attn = result[mask]

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        local_type = type(self.local_model).__name__ if self.local_model else 'None'
        global_type = type(self.self_attn).__name__ if self.self_attn else 'None'
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_model={local_type}, ' \
            f'global_model={global_type}'
        return s