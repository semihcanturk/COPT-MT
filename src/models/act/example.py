import torch
import torch.nn as nn
from torch_geometric.utils import unbatch


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class LinearAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        batch.x = torch.sigmoid(batch.x)
        return batch


class MinMaxNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x_list = unbatch(batch.x, batch.batch)

        p_list = []
        for x in x_list:
            x_min = x.min()
            x_max = x.max()

            p_list.append((x - x_min) / (x_max - x_min + 1e-6))

        batch.x = torch.cat(p_list, dim=0)
        return batch
