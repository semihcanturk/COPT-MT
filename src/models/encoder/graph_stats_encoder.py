from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder("GraphStats")
class GraphStatsEncoder(torch.nn.Module):

    def __init__(self, dim_in, dim_pe, dim_emb, stat_list, model_type='linear', n_layers=1, batch_norm=False, expand_x=True, pass_as_var=False):
        super().__init__()

        stats_dim = len(stat_list)
        self.stat_list = stat_list
        self.pass_as_var = pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if batch_norm:
            self.raw_norm = nn.BatchNorm1d(stats_dim)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(stats_dim, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(stats_dim, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(stats_dim, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        stats = list()
        for stat in self.stat_list:
            stats.append(getattr(batch, stat))

        pos_enc = torch.cat(stats, dim=1).float()
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch
