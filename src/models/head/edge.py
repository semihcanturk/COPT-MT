import torch

from src.models.network.mlp import MLP


class GNNEdgeHead(torch.nn.Module):
    r"""A GNN prediction head for edge-level/link-level prediction tasks.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int, dim_hid: int, layers_post_mp: int, batch_norm: bool = True,
                 l2_norm: bool = False, edge_decoding='concat', *args, **kwargs):
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
