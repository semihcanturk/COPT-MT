from typing import Optional

import torch
import torch.nn as nn

from src.models.layer import GeneralMultiLayer, Linear


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_dim: Optional[int],
        num_layers: int,
        batch_norm: bool = True,
        l2_norm: bool = False,
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
        else:
            hid_dim = in_dim
        layers.append(Linear(hid_dim, out_dim, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch
