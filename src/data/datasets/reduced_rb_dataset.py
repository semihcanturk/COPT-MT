"""Wrapper dataset that applies a CO reduction from `src.utils.reductions`
to every graph in an `RBDataset`.

Graph order is preserved: index `i` in the reduced dataset corresponds to
index `i` in the source dataset, so the same `split_seed` / `split_id`
in `SyntheticDataModule` produce matching train/val/test partitions.
"""
from src.data.datasets.rb_dataset import RBDataset
from src.data.datasets.reduced_synthetic import ReducedSyntheticDataset


class ReducedRBDataset(ReducedSyntheticDataset):
    source_cls = RBDataset
    source_format = "rb"

    def __init__(
        self,
        root: str,
        name: str,
        n,
        na,
        k,
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
                n=n,
                na=na,
                k=k,
                num_samples=num_samples,
                multiprocessing=multiprocessing,
                num_workers=num_workers,
            ),
            transform=transform,
            pre_transform=pre_transform,
        )
