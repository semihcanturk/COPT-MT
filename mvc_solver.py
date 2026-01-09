from dimod.reference.samplers import ExactSolver
import dwave_networkx as dnx
from src.data.synthetic_datamodule import SyntheticDataModule
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
from tqdm import tqdm

def main():
    # Configuration for rb_small based on configs/data/rb_small.yaml
    # We use task='maxclique' as it's the default in the config, 
    # but we are computing MVC manually.
    dm = SyntheticDataModule(
        data_dir="data/",
        format="rb",
        name="small",
        task="maxclique",
        batch_size=32,
        splits="5-fold",
        split_seed=42,
        num_workers=0,
        n=[200, 300],
        na=[20, 25],
        k=[5, 12],
        num_samples=6000
    )
    
    # Setup the data module to get the test set
    dm.setup(stage="test")
    test_set = dm.data_test
    
    print(f"Number of graphs in test set: {len(test_set)}")
    
    # ExactSolver from dimod is too slow for 200+ nodes (2^200 is too much).
    # Since we need the TRUE minimum vertex cover, and for these graphs we can 
    # compute it via Maximum Independent Set (MIS), we use NetworkX's 
    # max_weight_clique on the complement graph which is much more efficient 
    # for this dataset's structure while still being exact.
    
    mvc_sizes = []
    mis_sizes = []
    for i in tqdm(range(len(test_set)), desc="Computing MVC"):
        data = test_set[i]
        # Convert PyG data to NetworkX graph
        G = to_networkx(data, to_undirected=True)
        
        # MVC(G) = |V| - MIS(G)
        # MIS(G) = Maximum Clique in Complement(G)
        complement_G = nx.complement(G)
        _, mis_size = nx.clique.max_weight_clique(complement_G, weight=None)
        
        mvc_size = G.number_of_nodes() - mis_size
        mvc_sizes.append(mvc_size)
        mis_sizes.append(mis_size)
    
    mean_mvc = np.mean(mvc_sizes)
    mean_mis = np.mean(mis_sizes)
    print(f"\nMean Minimum Vertex Cover size: {mean_mvc}")
    print(f"\nMean MIS size: {mean_mis}")
    return mean_mvc, mean_mis

if __name__ == "__main__":
    main()
