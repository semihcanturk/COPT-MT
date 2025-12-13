# COPT-MT

## Installation

```bash
uv sync
uv pip install yacs einops dwave-networkx wandb ogb performer_pytorch python-sat
uv pip install torch_scatter torch_sparse --no-build-isolation
```

This project is built on the `lightning-hydra-template` (see `README_LHT.md` for template documentation), extended to support graph learning tasks using PyTorch Geometric (PyG). The following features have been added:

## TODO: Update sections below
## Data Modules

### GNNBenchmarkDataset

The `GNNBenchmarkDataModule` provides access to the PyG GNNBenchmarkDataset, which contains a variety of graph datasets for benchmarking GNNs:

- PATTERN: A node classification dataset with 2 classes
- CLUSTER: A node classification dataset with 6 classes
- MNIST: A graph classification dataset with 10 classes
- CIFAR10: A graph classification dataset with 10 classes

To use this data module, specify it in your configuration:

```yaml
defaults:
  - data: gnn_benchmark.yaml
```

You can customize the dataset by modifying the configuration:

```yaml
data:
  dataset_name: "PATTERN"  # Options: "PATTERN", "CLUSTER", "MNIST", "CIFAR10"
  batch_size: 32
```

### TUDataset

The `TUDataModule` provides access to the PyG TUDataset, which is a collection of graph datasets from the TU Dortmund University:

- MUTAG: A graph classification dataset with 2 classes
- PROTEINS: A graph classification dataset with 2 classes
- ENZYMES: A graph classification dataset with 6 classes
- DD: A graph classification dataset with 2 classes
- NCI1: A graph classification dataset with 2 classes
- And many more...

To use this data module, specify it in your configuration:

```yaml
defaults:
  - data: tu_dataset.yaml
```

You can customize the dataset by modifying the configuration:

```yaml
data:
  dataset_name: "MUTAG"  # Options: "MUTAG", "PROTEINS", "ENZYMES", "DD", "NCI1", etc.
  batch_size: 32
  train_val_test_split: [0.7, 0.1, 0.2]
```

## Models

### GCN (Graph Convolutional Network)

The `GCNNet` is a Graph Convolutional Network that can be used for both graph-level and node-level tasks. It consists of multiple GCN layers followed by global pooling for graph-level tasks.

To use this model, specify it in your configuration:

```yaml
defaults:
  - model: gcn.yaml
```

You can customize the model by modifying the configuration:

```yaml
model:
  task: "graph"  # Options: "graph", "node"
  num_classes: 2  # Number of classes for classification
  net:
    in_channels: 7  # Number of input features
    hidden_channels: 64
    out_channels: 2  # Should match num_classes
    num_layers: 2
    dropout: 0.5
```

## Example Experiments

### Graph Classification

To run a graph classification experiment using the TUDataset and GCN model:

```bash
python src/train.py experiment=gcn_graph_classification
```

This will train a GCN model on the MUTAG dataset for graph classification.

### Node Classification

To run a node classification experiment using the GNNBenchmarkDataset and GCN model:

```bash
python src/train.py experiment=gcn_node_classification
```

This will train a GCN model on the CLUSTER dataset for node classification.

## Graph Transforms

Both `GNNBenchmarkDataModule` and `TUDataModule` support PyTorch Geometric transforms, which can be used to preprocess the graph data or add additional features like positional encodings.

By default, the `NormalizeFeatures` transform is always applied. You can add additional transforms by modifying the configuration:

```yaml
data:
  transforms:
    transform_name:
      _target_: torch_geometric.transforms.TransformName
      param1: value1
      param2: value2
```

### Example: Adding Positional Encodings

Positional encodings can improve the performance of GNNs by providing information about the structure of the graph. Here's an example of how to add Laplacian eigenvector-based positional encodings:

```yaml
data:
  transforms:
    laplacian_pe:
      _target_: torch_geometric.transforms.AddLaplacianEigenvectorPE
      k: 5  # Number of Laplacian eigenvectors to use
```

This will add 5 Laplacian eigenvectors as additional node features, which can help the model learn about the graph structure.

Example configuration files are provided:
- `configs/data/gnn_benchmark_with_pe.yaml`
- `configs/data/tu_dataset_with_pe.yaml`

To use these configurations:

```bash
python src/train.py data=gnn_benchmark_with_pe
```

or

```bash
python src/train.py data=tu_dataset_with_pe
```

## Requirements

Make sure you have the required PyTorch Geometric packages installed:

```bash
pip install torch-geometric torch-scatter torch-sparse
```

These packages are included in the project's requirements.txt file.

## Example of multitask learning and finetuning

E.g. multitask on maxcut and mis:

```bash
python src/train.py experiment=multitask model.net.tasks=[maxcut, mis]
```

One can weight the different loss functions in multitask learning using model.weights, e.g. 

```bash
python src/train.py experiment=multitask model.net.tasks=[maxcut, mis] model.weights.maxcut=0.8 model.weights.mis=0.2
```

Running a task with finetuning off will upload checkpoints in logs/train/checkpoints/"task_names"/ (best.ckpt or last.ckpt)
e.g. logs/train/checkpoints/mis/, logs/train/checkpoints/color10/, logs/train/checkpoints/mis_mds/

For finetuning, we need model.net.finetuning.strategy = 'finetuning', 'linear_probing' or 'pre_post'

E.g. Finetune mis on maxcut:

```bash
python src/train.py experiment=multitask model.net.finetuning.strategy='finetuning' model.net.finetuning.new_tasks=[mis] model.net.finetuning.path=logs/train/checkpoints/maxcut/last.ckpt
```

To pick the number of colors in graph coloring, use model.net.dims_out.color = #colors

E.g. 10 colors

```bash
python src/train.py experiment=multitask model.net.tasks=[color] model.net.dim_out.color=10
```


