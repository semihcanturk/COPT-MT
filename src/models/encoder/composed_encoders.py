import torch
from torch_geometric.graphgym.config import cfg


class Concat2NodeEncoder(torch.nn.Module):
    """Encoder that concatenates two node encoders.
    """

    def __init__(self, encoder1, encoder2, **kwargs):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def forward(self, batch):
        batch = self.encoder1(batch)
        batch = self.encoder2(batch)
        return batch


class Concat3NodeEncoder(torch.nn.Module):
    """Encoder that concatenates three node encoders.
    """
    enc1_cls = None
    enc2_cls = None
    enc2_name = None
    enc3_cls = None
    enc3_name = None

    def __init__(self, dim_emb):
        super().__init__()
        # PE dims can only be gathered once the cfg is loaded.
        enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
        enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
        self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
        self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
        self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)

    def forward(self, batch):
        batch = self.encoder1(batch)
        batch = self.encoder2(batch)
        batch = self.encoder3(batch)
        return batch
