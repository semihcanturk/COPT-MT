import os

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dimod.reference.samplers import ExactSolver
import dwave_networkx as dnx
import hydra
from hydra import compose, initialize_config_dir
from src.data.synthetic_datamodule import SyntheticDataModule
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
from tqdm import tqdm

import gurobipy as gp
from gurobipy import GRB

from src.utils.reductions import complement_graph, mid_edge_vertex_gadget, maxcut_to_mds


def load_eval_artifacts(overrides):
    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="eval.yaml", overrides=overrides)
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    return model, datamodule, cfg


def gurobi_mds(G):
    # Create model object
    m = gp.Model()

    # Create variable for each node
    x = m.addVars(G.nodes, vtype=GRB.BINARY)

    # Objective function: minimize number of nodes
    m.setObjective(gp.quicksum(x[i] for i in G.nodes), GRB.MINIMIZE)

    # Add constraint for each node
    m.addConstrs(x[i] + gp.quicksum(x[j] for j in G.neighbors(i)) >= 1 for i in G.nodes)

    # Solve
    m.optimize()

    return [ i for i in G.nodes if x[i].x > 0.5 ], m.objVal


def gurobi_maxclique(G):
    # Create model object
    m = gp.Model()

    # Create variable for each node
    x = m.addVars(G.nodes, vtype=GRB.BINARY)

    # Objective function: maximize number of nodes
    m.setObjective(gp.quicksum(x[i] for i in G.nodes), GRB.MAXIMIZE)

    # Add constraint for each missing edge (i.e., edge of the complement graph)
    CG = nx.complement(G)
    m.addConstrs(x[i] + x[j] <= 1 for i, j in CG.edges)

    # Solve
    m.optimize()

    return [i for i in G.nodes if x[i].x > 0.5], m.objVal


def gurobi_maxcut(G):
    m = gp.Model()

    # Side assignment for each node and cut-indicator for each edge.
    z = m.addVars(G.nodes, vtype=GRB.BINARY)
    y = m.addVars(G.edges, vtype=GRB.BINARY)

    # y[i,j] = 1 iff z[i] != z[j]:
    #   if z_i = z_j = 0 -> first constraint forces y <= 0
    #   if z_i = z_j = 1 -> second constraint forces y <= 0
    #   otherwise both upper bounds are 1, and `max` pushes y to 1.
    m.addConstrs(y[i, j] <= z[i] + z[j] for i, j in G.edges)
    m.addConstrs(y[i, j] <= 2 - z[i] - z[j] for i, j in G.edges)

    m.setObjective(gp.quicksum(y[e] for e in G.edges), GRB.MAXIMIZE)
    m.optimize()

    return None, m.objVal



def test_mis_mvc_to_mds():
    dm = SyntheticDataModule(
        data_dir="data/",
        format="ba",
        name="small",
        task="mis",
        batch_size=32,
        splits="5-fold",
        split_seed=42,
        num_workers=0,
        n_min=200,
        n_max=300,
        num_edges=4,
        num_samples=6000
    )

    dm_mis_mds = SyntheticDataModule(
        data_dir="data/",
        format="ba",
        name="small",
        task="mds",
        reduction="mis_to_mds",
        batch_size=32,
        splits="5-fold",
        split_seed=42,
        num_workers=0,
        n_min=200,
        n_max=300,
        num_edges=4,
        num_samples=6000
    )

    # Setup the data module to get the test set
    dm.setup()
    dm_mis_mds.setup()

    test_set = dm.data_test
    mis_mds_test_set = dm_mis_mds.data_test
    print(f"Number of graphs in test set: {len(test_set)}")

    for i in tqdm(range(len(test_set)), desc="Computing MVC"):
        data = test_set[i]
        data_mis_mds = mis_mds_test_set[i]

        G = to_networkx(data, to_undirected=True)
        G_mis_mds = to_networkx(data_mis_mds, to_undirected=True)

        # MVC(G) = |V| - MIS(G)
        # MIS(G) = Maximum Clique in Complement(G)
        complement_G = nx.complement(G)
        _, mis_size = gurobi_maxclique(complement_G)
        _, mis_mds_size = gurobi_mds(G_mis_mds)

        mvc_size = complement_G.number_of_nodes() - mis_size
        print(f"MVC: {mvc_size} MDS: {mis_mds_size}")


def test_mc_to_mds():
    dm = SyntheticDataModule(
        data_dir="data/",
        format="ba",
        name="small",
        task="mis",
        batch_size=32,
        splits="5-fold",
        split_seed=42,
        num_workers=0,
        n_min=200,
        n_max=300,
        num_edges=4,
        num_samples=6000
    )

    overrides = [
        "experiment=mds/ba_small/gcon_reduction",
        "data.batch_size=1",
        "model.net.dim_inner=64",
        "model.metrics.size.reduction=maxclique_to_mds",
        "data=ba_small_mds_from_maxclique",
        "ckpt_path=/Users/semo/PycharmProjects/COPT-MT/weights/mds/GCON-GS-64.ckpt",
    ]
    model, dm_mc_mds, _ = load_eval_artifacts(overrides)

    dm.setup(stage="test")
    test_set = dm.data_test
    mc_mds_test_set = dm_mc_mds.data_test
    print(f"Number of graphs in test set: {len(test_set)}")

    for i in tqdm(range(len(test_set)), desc="Computing MVC"):
        data = test_set[i]
        data_mc_mds = mc_mds_test_set[i]

        G = to_networkx(data, to_undirected=True)
        G_mc_mds = to_networkx(data_mc_mds, to_undirected=True)

        _, mc_mds_comp_size = gurobi_mds(G_mc_mds)

        _, mc_size = gurobi_maxclique(G)

        batch = Batch.from_data_list([data_mc_mds])
        with torch.no_grad():
            out = model(batch)
        pred_mc = model.metrics["size"](out).item()

        print(f"MC: {mc_size}  MDS-recovered: {G.number_of_nodes() - mc_mds_comp_size}  model: {pred_mc:.2f}")


def test_maxcut_to_mds():
    dm = SyntheticDataModule(
        data_dir="data/",
        format="ba",
        name="small",
        task="maxcut",
        batch_size=32,
        splits="5-fold",
        split_seed=42,
        num_workers=0,
        n_min=200,
        n_max=300,
        num_edges=4,
        num_samples=6000,
    )

    dm.setup(stage="test")
    test_set = dm.data_test
    print(f"Number of graphs in test set: {len(test_set)}")

    for i in tqdm(range(len(test_set)), desc="Computing MaxCut"):
        data = test_set[i]
        G = to_networkx(data, to_undirected=True) #nx.petersen_graph()
        G.remove_edges_from(list(nx.selfloop_edges(G)))

        G_maxcut_mds, _ = maxcut_to_mds(G)

        _, maxcut_size = gurobi_maxcut(G)
        _, maxcut_mds_size = gurobi_mds(G_maxcut_mds)

        # Reduction relationship:
        #   MDS(G') = 2|E(G)| - MaxCut(G) + n_isolated_in_conflict(H)
        # For BA graphs the conflict graph has no isolates, so the
        # recovered MaxCut should match the direct gurobi MaxCut.
        recovered = 2 * G.number_of_edges() - maxcut_mds_size
        print(f"MaxCut: {maxcut_size}  recovered (2|E|-MDS): {recovered}  MDS: {maxcut_mds_size}")




def main():
    # test_mis_mvc_to_mds()
    test_mc_to_mds()
    # test_maxcut_to_mds()


if __name__ == "__main__":
    main()
