from functools import partial
from typing import Any, Dict, Optional, Tuple, List, Union

import networkx as nx
import torch
import numpy as np
import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import NormalizeFeatures, Compose, Constant
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Dataset

from src.data.datasets.synthetic import SyntheticDataset
from src.data.datasets.ba_dataset import BADataset
from src.data.datasets.er_dataset import ERDataset
from src.data.datasets.bp_dataset import BPDataset
from src.data.datasets.pc_dataset import PCDataset
from src.data.datasets.rb_dataset import RBDataset
from src.transforms.graph_stats import ComputeGraphStats, ComputeComplementGraphStats
from src.transforms.transforms import pre_transform_in_memory


def compute_maxcut(g):
    adj = torch.from_numpy(nx.to_numpy_array(g))
    num_nodes = adj.size(0)

    cut = dnx.maximum_cut(g, dimod.SimulatedAnnealingSampler())
    cut_size = max(len(cut), g.number_of_nodes() - len(cut))
    cut_binary = torch.zeros((num_nodes, 1), dtype=torch.int)
    cut_binary[torch.tensor(list(cut))] = 1

    return cut_size, cut_binary


def set_maxcut(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # Derive adjacency matrix
    cut_size, cut_binary = compute_maxcut(g)

    data.cut_size = cut_size
    data.cut_binary = cut_binary
    return data


def set_maxclique(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # target = {"mc_size": max(len(clique) for clique in nx.find_cliques(g))}
    data.mc_size = max(len(clique) for clique in nx.find_cliques(g))
    return data


def set_plantedclique(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # target = {"mc_size": max(len(clique) for clique in nx.find_cliques(g))}
    data.mc_size = max(len(clique) for clique in nx.find_cliques(g))
    return data


def set_y(data, dataset_name, task, label=False):
    if not label:
        data.y = torch.ones(data.num_nodes, 1)
    elif task == 'maxcut':
        data.y = data.cut_binary
    elif task == 'maxclique':
        if dataset_name not in ['IMDB-BINARY', 'COLLAB', 'ego-twitter']:
            data.y = data.mc_size
    return data


class SyntheticSubset(Dataset):
    """A custom Dataset class for SyntheticDataset subsets.

    This class wraps a subset of a SyntheticDataset and provides the necessary functionality
    for PyTorch Geometric's DataLoader.
    """

    def __init__(self, dataset, indices):
        """Initialize a SyntheticDatasetSubset.

        Args:
            dataset: The original SyntheticDataset.
            indices: The indices to include in the subset.
        """
        super().__init__()
        self.dataset = dataset
        self._indices = indices  # Store as private attribute

        # Store private attributes for properties
        self._num_classes = dataset.num_classes
        self._num_node_features = dataset.num_node_features

    def indices(self):
        """Return the indices of the dataset.

        This method is required by PyTorch Geometric's Dataset class.
        """
        return self._indices

    @property
    def num_node_features(self):
        """Return the number of node features in the dataset."""
        return self._num_node_features

    def len(self):
        """Return the number of samples in the subset."""
        return len(self._indices)

    def get(self, idx):
        """Get a sample from the subset.

        Args:
            idx: The index in the indices list.

        Returns:
            The sample at the given index.
        """
        if idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range for indices list of length {len(self._indices)}")
        return self.dataset[self._indices[idx]]

    def __getitem__(self, idx):
        """Get a sample from the subset.

        This method is required by PyTorch's DataLoader.

        Args:
            idx: The index of the sample in the subset.

        Returns:
            The sample at the given index.
        """
        return self.get(idx)


class SyntheticDataModule(LightningDataModule):
    """LightningDataModule for SyntheticDataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        format: str = "rb",
        name: str = "small",
        task: str = "maxcut",
        batch_size: int = 64,
        splits: Union[str, Tuple[float, float, float]] = (0.7, 0.1, 0.2),
        split_seed: int = 42,
        split_id: int = 0,
        labels: bool = False,
        graph_stats: Optional[list[str]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        transforms: Optional[Union[Dict[str, Any], BaseTransform]] = None,
        **dataset_kwargs,
    ) -> None:
        """Initialize a `TUDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param dataset_name: The name of the dataset. Defaults to `"MUTAG"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param splits: The train, validation and test split. Defaults to `(0.7, 0.1, 0.2)`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param transforms: PyTorch Geometric transforms to apply to the data. Can be a single transform,
            a list of transforms, or a dictionary of transform configurations. Defaults to `None`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Initialize transforms
        self._init_pretransforms()
        self._init_transforms(transforms)

        self.data_train: Optional[SyntheticDataset] = None
        self.data_val: Optional[SyntheticDataset] = None
        self.data_test: Optional[SyntheticDataset] = None

        self.batch_size_per_device = batch_size

        self.dataset_cls = {
            'ba': BADataset,
            'er': ERDataset,
            'bp': BPDataset,
            'pc': PCDataset,
            'rb': RBDataset,
        }[self.hparams.format]
        self.dataset_kwargs = dataset_kwargs

    def _init_pretransforms(self):
        if not self.hparams.labels or self.hparams.task == 'plantedclique':
            pre_tf_list = []
        else:
            pre_tf_list = [set_maxcut, set_maxclique]
        tf_list = [Constant(), partial(set_y, dataset_name=self.hparams.name, task=self.hparams.task, label=self.hparams.labels)]

        if self.hparams.graph_stats:
            pre_tf_list.append(ComputeComplementGraphStats(self.hparams.graph_stats, gsn=False))

        self.pre_transforms = Compose(pre_tf_list)
        self.pre_transforms_in_memory = Compose(tf_list)

    def _init_transforms(self, transforms_config: Optional[Union[Dict[str, Any], BaseTransform]] = None) -> None:
        """Initialize the transforms for the dataset.

        :param transforms_config: PyTorch Geometric transforms to apply to the data. Can be a single transform,
            a list of transforms, or a dictionary of transform configurations. Defaults to `None`.
        """
        transform_list = []

        # Always add NormalizeFeatures as the default transform
        # transform_list.append(NormalizeFeatures())

        # Process additional transforms if provided
        if transforms_config is not None:
            if isinstance(transforms_config, BaseTransform):
                # If a single transform is provided, add it to the list
                transform_list.append(transforms_config)
            elif isinstance(transforms_config, list):
                # If a list of transforms is provided, extend the list
                transform_list.extend(transforms_config)
            elif isinstance(transforms_config, Union[dict, DictConfig]):
                # If a dictionary of transform configurations is provided, instantiate each transform
                for transform_config in transforms_config.values():
                    if isinstance(transform_config, Union[dict, DictConfig]) and "_target_" in transform_config:
                        # Instantiate the transform using Hydra
                        transform = hydra.utils.instantiate(transform_config)
                        transform_list.append(transform)
                    elif isinstance(transform_config, BaseTransform):
                        # If a transform instance is provided directly
                        transform_list.append(transform_config)

        # Create a Compose transform if there are multiple transforms
        if len(transform_list) > 1:
            self.transforms = Compose(transform_list)
        elif len(transform_list) == 1:
            self.transforms = transform_list[0]
        else:
            self.transforms = None

    @property
    def num_node_features(self) -> int:
        """Get the number of node features.

        :return: The number of node features in the dataset.
        """
        if self.data_train is not None:
            return self.data_train.num_node_features
        else:
            return 1

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Download the data
        self.dataset_cls(
            root=self.hparams.data_dir,
            name=self.hparams.name,
            pre_transform=self.pre_transforms,
            transform=self.transforms,
            **self.dataset_kwargs
        )

    def split_train_val_test(self, dataset):
        # Get the total number of samples
        num_samples = len(dataset)

        # Create indices for the entire dataset
        indices = list(range(num_samples))

        # Set random seed for reproducibility
        np.random.seed(self.hparams.split_seed)
        np.random.shuffle(indices)

        # Calculate split sizes
        train_size = int(num_samples * self.hparams.splits[0])
        val_size = int(num_samples * self.hparams.splits[1])
        test_size = num_samples - train_size - val_size

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create dataset subsets using the custom SyntheticDatasetSubset class
        self.data_train = SyntheticSubset(dataset, train_indices)
        self.data_val = SyntheticSubset(dataset, val_indices)
        self.data_test = SyntheticSubset(dataset, test_indices)

    def split_k_fold(self, dataset, k):
        # separate test set
        num_samples = len(dataset)
        indices = list(range(num_samples))
        train_val_indexes, test_indexes = train_test_split(indices, test_size=float(1 / (k + 1)), random_state=self.hparams.split_seed)
        self.data_test = dataset[test_indexes]
        # choose fold to train on
        kf = KFold(n_splits=k, shuffle=True, random_state=self.hparams.split_seed)
        dataset = dataset[train_val_indexes]
        all_splits = [k for k in kf.split(dataset)]
        train_indexes, val_indexes = all_splits[self.hparams.split_id]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
        self.data_train, self.data_val = dataset[train_indexes], dataset[val_indexes]

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = self.dataset_cls(
                root=self.hparams.data_dir,
                name=self.hparams.name,
                pre_transform=self.pre_transforms,
                transform=self.transforms,
                **self.dataset_kwargs
            )

            pre_transform_in_memory(dataset, self.pre_transforms_in_memory, show_progress=True)

            if isinstance(self.hparams.splits, str):
                k, _ = self.hparams.splits.split('-')
                self.split_k_fold(dataset, int(k))
            else:
                self.split_train_val_test(dataset)


    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = SyntheticDataModule()
