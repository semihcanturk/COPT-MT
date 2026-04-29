"""Wrapper dataset that applies a CO reduction from `src.utils.reductions`
to every graph in a `BADataset`.

Graph order is preserved: index `i` in the reduced dataset corresponds to
index `i` in the source dataset, so the same `split_seed` / `split_id`
in `SyntheticDataModule` produce matching train/val/test partitions.
"""
from src.data.datasets.ba_dataset import BADataset
from src.data.datasets.reduced_synthetic import ReducedSyntheticDataset


class ReducedBADataset(ReducedSyntheticDataset):
    source_cls = BADataset
    source_format = "ba"

    def __init__(
        self,
        root: str,
        name: str,
        n_min: int,
        n_max: int,
        num_edges: int,
        num_samples: int,
        reduction: str,
        multiprocessing: bool = False,
        num_workers: int = 0,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(
            root=root,
            name=name,
            reduction=reduction,
            source_kwargs=dict(
                n_min=n_min,
                n_max=n_max,
                num_edges=num_edges,
                num_samples=num_samples,
                multiprocessing=multiprocessing,
                num_workers=num_workers,
            ),
            transform=transform,
            pre_transform=pre_transform,
        )
