from typing import Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('set_y')
class SetY(BaseTransform):
    def __init__(self, name, task, label=False):
        self.name = name
        self.task = task
        self.label = label

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        if not self.label:
            data.y = torch.ones(data.num_nodes, 1)
        elif self.task == 'maxcut':
            data.y = data.cut_binary
        elif self.task == 'maxclique':
            if self.name not in ['IMDB-BINARY', 'COLLAB', 'ego-twitter']:
                data.y = data.mc_size
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, task={self.task}, label={self.label})'