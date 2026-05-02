import time
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index, add_self_loops, \
    remove_self_loops, to_dense_adj
from torch_scatter import scatter


def accuracy(output, target):
    return torch.mean((output.argmax(-1) == target).float())


### MAXCLIQUE ###

def maxclique_size_pyg(batch, dec_length=300, num_seeds=1):
    batch = maxclique_decoder_pyg_parallel(batch, dec_length=dec_length, num_seeds=num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.c_size for data in data_list]

    return torch.Tensor(size_list).mean()


def maxclique_ratio_pyg(batch, dec_length=300, num_seeds=1):
    batch = maxclique_decoder_pyg_parallel(batch, dec_length=dec_length, num_seeds=num_seeds)

    data_list = batch.to_data_list()

    metric_list = []
    for data in data_list:
        metric_list.append(data.c_size / data.y)

    return torch.Tensor(metric_list).mean()


def get_csize(seed, data, dec_length):
    order = torch.argsort(data.x, dim=0, descending=True)
    c = torch.zeros_like(data.x)

    edge_index = remove_self_loops(data.edge_index)[0]
    src, dst = edge_index[0], edge_index[1]

    c[order[seed]] = 1
    for idx in range(seed, min(dec_length, data.num_nodes)):
        c[order[idx]] = 1

        cTWc = torch.sum(c[src] * c[dst])
        if c.sum() ** 2 - cTWc - torch.sum(c ** 2) != 0:
            c[order[idx]] = 0

    return c.sum()


def get_csize_np(seed, x, edge_index, num_nodes, dec_length):
    order = np.argsort(-1 * x, axis=0)
    c = np.zeros_like(x)

    edge_index = remove_self_loops(edge_index)[0]
    src, dst = edge_index[0], edge_index[1]

    c[order[seed]] = 1
    for idx in range(seed, min(dec_length, num_nodes)):
        c[order[idx]] = 1

        cTWc = np.sum(c[src] * c[dst])
        if np.sum(c) ** 2 - cTWc - np.sum(c ** 2) != 0:
            c[order[idx]] = 0

    return np.sum(c)


def maxclique_decoder_pyg(batch, dec_length=300, num_seeds=1):
    data_list = batch.to_data_list()

    for data in data_list:
        c_size_list = []
        for seed in range(num_seeds):

            order = torch.argsort(data.x, dim=0, descending=True)
            c = torch.zeros_like(data.x)

            edge_index = remove_self_loops(data.edge_index)[0]
            src, dst = edge_index[0], edge_index[1]

            c[order[seed]] = 1
            for idx in range(seed, min(dec_length, data.num_nodes)):
                c[order[idx]] = 1

                cTWc = torch.sum(c[src] * c[dst])
                if c.sum() ** 2 - cTWc - torch.sum(c ** 2) != 0:
                    c[order[idx]] = 0

            c_size_list.append(c.sum())

        data.c_size = max(c_size_list)

    return Batch.from_data_list(data_list)


def maxclique_decoder_pyg_parallel(batch, dec_length=300, num_seeds=1):
    """
    Optimized parallel decoder that processes all seeds simultaneously using vectorized operations.
    This avoids multiprocessing overhead and leverages PyTorch's native parallelization.
    """
    data_list = batch.to_data_list()

    for data in data_list:
        order = torch.argsort(data.x, dim=0, descending=True).squeeze()

        edge_index = remove_self_loops(data.edge_index)[0]
        src, dst = edge_index[0], edge_index[1]
        num_nodes = data.num_nodes
        max_idx = min(dec_length, num_nodes)
        device = data.x.device
        
        c = torch.zeros(num_seeds, num_nodes, dtype=torch.float32, device=device)
        seed_positions = torch.arange(num_seeds, device=device)
        seed_positions = torch.clamp(seed_positions, max=max_idx - 1)
        c[torch.arange(num_seeds, device=device), order[seed_positions]] = 1.0

        for idx in range(max_idx):
            # A seed considers node idx if idx >= seed_position[seed]
            active_seeds = idx >= seed_positions
            
            if not active_seeds.any():
                continue
            
            node_idx = order[idx]

            c_temp = c.clone()
            c_temp[active_seeds, node_idx] = 1.0
            c_sum = c_temp.sum(dim=1)
            
            c_src = c_temp[:, src]
            c_dst = c_temp[:, dst]
            cTWc = (c_src * c_dst).sum(dim=1)
            c_sq_sum = (c_temp ** 2).sum(dim=1)
            
            # For a valid clique, c.sum()^2 - cTWc - torch.sum(c^2) = 0
            validity = c_sum ** 2 - cTWc - c_sq_sum
            valid_mask = (torch.abs(validity) < 1e-6) & active_seeds
            c[valid_mask, node_idx] = 1.0
        
        c_sizes = c.sum(dim=1)
        data.c_size = c_sizes.max().item()

    return Batch.from_data_list(data_list)


def maxclique_ratio(output, data, dec_length=300):
    adj = data.get('adj')
    num_nodes = data.get('num_nodes')
    c = maxclique_decoder(output, adj, num_nodes, dec_length=dec_length)

    target = data.get('mc_size')

    return torch.mean(c.sum(-1) / target)


def maxclique_decoder(output, adj, num_nodes, dec_length=300):
    order = [torch.argsort(output[sample_idx][:num_nodes[sample_idx]], dim=0,
                           descending=True) for sample_idx in
             range(output.size(0))]
    c = torch.zeros_like(output)

    for sample_idx in range(output.size(0)):
        c[sample_idx][order[sample_idx][0]] = 1

        for i in range(1, min(dec_length, num_nodes[sample_idx])):
            c[sample_idx][order[sample_idx][i]] = 1

            cTWc = torch.matmul(c[sample_idx].transpose(-1, -2),
                                torch.matmul(adj[sample_idx], c[sample_idx]))
            if c[sample_idx].sum() ** 2 - cTWc - torch.sum(
                    c[sample_idx] ** 2) != 0:
                c[sample_idx][order[sample_idx][i]] = 0

    return c.squeeze(-1)


### MAXCUT ###

def maxcut_acc_pyg(data):
    x = (data.x > 0.5).float()
    x = (x - 0.5) * 2
    y = data.cut_binary
    y = (y - 0.5) * 2

    x_list = unbatch(x, data.batch)
    y_list = unbatch(y, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    comparison_list = []
    for x, y, edge_index in zip(x_list, y_list, edge_index_list):
        x_cut = torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0)
        y_cut = torch.sum(y[edge_index[0]] * y[edge_index[1]] == -1.0)
        comparison_list.append(x_cut >= y_cut)

    return torch.Tensor(comparison_list).mean()


def maxcut_size_pyg(data):
    x = (data.x > 0.5).float()
    x = (x - 0.5) * 2

    x_list = unbatch(x, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    cut_list = []
    for x, edge_index in zip(x_list, edge_index_list):
        cut_list.append(
            torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) / 2)

    return torch.Tensor(cut_list).mean()


def maxcut_acc(data):
    adj = data['adj']
    adj_weight = adj.sum(-1).sum(-1)
    target_size = adj_weight.clone()
    pred_size = adj_weight.clone()

    target = torch.nan_to_num(data['cut_binary'])
    target_size -= torch.matmul(target.transpose(-1, -2),
                                torch.matmul(adj, target)).squeeze()
    target = 1 - target
    target_size -= torch.matmul(target.transpose(-1, -2),
                                torch.matmul(adj, target)).squeeze()
    target_size /= 2

    output = (data['x'] > 0.5).float()
    pred_size -= torch.matmul(output.transpose(-1, -2),
                              torch.matmul(adj, output)).squeeze()
    output = 1 - output
    pred_size -= torch.matmul(output.transpose(-1, -2),
                              torch.matmul(adj, output)).squeeze()
    pred_size /= 2

    return (pred_size >= target_size).float().mean()


### COLORING ###

def color_violations_pyg(batch):
    data_list = batch.to_data_list()
    violation_total = []

    for data in data_list:
        preds = torch.argmax(data.x,dim=-1) 

        edge_index, _ = remove_self_loops(data.edge_index)
        src, dst = edge_index

        violations = (preds[src] == preds[dst]).sum() // 2
        violation_total.append(violations)

    return torch.Tensor(violation_total).mean()

def soft_guided_repair(data):
    """
    Greedy repair decoder using soft assignments to guide color selection.
    - Fixes violated nodes in ascending confidence order (least confident first)
    - Among available colors, picks the one with highest soft probability
    - Excludes current color to force a real change
    """
    X = data.x  # [N, K]
    edge_index, _ = remove_self_loops(data.edge_index)
    src, dst = edge_index

    coloring = X.argmax(dim=-1).clone()
    k = X.shape[-1]
    N = X.shape[0]
    device = X.device

    # Precompute adjacency list
    adj = [dst[src == i] for i in range(N)]

    # Precompute soft-probability color order per node (never changes)
    sorted_colors_all = X.argsort(dim=-1, descending=True).tolist()  # [N, K]

    for _ in range(k):
        # Count conflicts per node
        conflict_count = torch.zeros(N, dtype=torch.long, device=device)
        same_color = (coloring[src] == coloring[dst])
        if not same_color.any():
            break
        conflict_count.scatter_add_(
            0, src[same_color],
            torch.ones(same_color.sum(), dtype=torch.long, device=device)
        )

        violated_nodes = (conflict_count > 0).nonzero(as_tuple=True)[0]

        # Primary order: ascending confidence in current (wrong) color
        # i.e. least confident nodes are fixed first — easier to recolor
        current_confidence = X[violated_nodes, coloring[violated_nodes]]
        # Secondary order: descending conflict count as tie-breaker
        # normalize conflict count to [0, 1] for stable combination
        norm_conflicts = conflict_count[violated_nodes].float()
        norm_conflicts = norm_conflicts / (norm_conflicts.max() + 1e-8)
        # Low confidence + high conflict = fix first
        priority = current_confidence - norm_conflicts  # lower = fix sooner
        order = conflict_count[violated_nodes].argsort(descending=True)
        violated_nodes = violated_nodes[order]

        for node in violated_nodes.tolist():
            # Recheck: is this node still in conflict after previous fixes?
            neighbor_colors_list = coloring[adj[node]].tolist()
            if coloring[node].item() not in neighbor_colors_list:
                continue  # already resolved, skip

            neighbor_colors = set(neighbor_colors_list)
            neighbor_colors.add(coloring[node].item())
            for color in sorted_colors_all[node]:
                if color not in neighbor_colors:
                    coloring[node] = color
                    break

    return coloring, edge_index


def color_repaired_violations_pyg(batch):
    data_list = batch.to_data_list()
    violation_total = []

    for data in data_list:
        preds, edge_index = soft_guided_repair(data)
        src, dst = edge_index

        violations = (preds[src] == preds[dst]).sum() // 2
        violation_total.append(violations)

    return torch.Tensor(violation_total).mean()


def color_acc(output, adj, deg_vect):
    output = (output - 0.5) * 2

    one_hot = output > 0
    bin_enc = (one_hot.float() - 0.5) * 2

    return (torch.matmul(bin_enc.transpose(-1, -2), torch.matmul(adj, bin_enc)).diagonal(dim1=-1, dim2=-2).sum(-1) / deg_vect).mean()


### CLIQUE COVERING ###

def cliquecover_violations_pyg(batch):
    data_list = batch.to_data_list()
    missing_total = []

    for data in data_list:
        preds_idx = torch.argmax(data.x, dim=-1)  # color per node
        n = data.num_nodes

        # Build adjacency (0 = no edge, 1 = edge)
        adj = to_dense_adj(data.edge_index, max_num_nodes=n)[0]
        adj.fill_diagonal_(0)

        # Compare all pairs
        same_color = (preds_idx.unsqueeze(0) == preds_idx.unsqueeze(1)).fill_diagonal_(False)  # [n, n]
        non_edges = (adj == 0)
        missing = (same_color & non_edges).sum() // 2

        missing_total.append(missing)

    return torch.Tensor(missing_total).mean()


### HAMILTONIAN CYCLE PROBLEM ###

def hcp_violations_pyg(batch):
    data_list = batch.to_data_list()

    total_list = []
    pos_list = []
    edge_list = []
    dup_list = []
    miss_list = []

    for data in data_list:
        X = data.x

        if data.num_nodes != 250:
            raise ValueError(f"Expected 250 nodes, got {data.num_nodes}")

        if X.shape[0] != X.shape[1]:
            raise ValueError(f"HCP requires square matrix, got {X.shape}")

        edge_index, _ = remove_self_loops(data.edge_index)
        src, dst = edge_index
        n = X.shape[0]

        # --- assignments ---
        preds = torch.argmax(X, dim=-1)

        # --- position violations (better decomposition) ---
        counts = torch.bincount(preds, minlength=n)

        violations_dup = torch.sum(torch.clamp(counts - 1, min=0))
        violations_miss = torch.sum(counts == 0)
        violations_pos = violations_dup + violations_miss

        # --- position → node mapping ---
        pos2node = torch.full((n,), -1, device=X.device)
        for p in preds.unique():
            idx = (preds == p).nonzero(as_tuple=True)[0]
            pos2node[p] = idx[0]

        # --- cycle ---
        next_pos = torch.roll(torch.arange(n, device=X.device), -1)
        u = pos2node
        v = pos2node[next_pos]

        valid_mask = (u >= 0) & (v >= 0)
        u, v = u[valid_mask], v[valid_mask]

        if u.numel() == 0:
            total = violations_pos
            violations_edge = torch.tensor(0, device=X.device)
        else:
            adj = torch.zeros((n, n), dtype=torch.bool, device=X.device)
            adj[src, dst] = True

            if hasattr(data, "is_undirected") and data.is_undirected():
                adj[dst, src] = True

            violations_edge = (~adj[u, v]).sum()
            total = violations_pos + violations_edge

        total_list.append(total)
        pos_list.append(violations_pos)
        edge_list.append(violations_edge)
        dup_list.append(violations_dup)
        miss_list.append(violations_miss)

    return torch.stack(dup_list).float().mean()


### PLANTEDCLIQUE ###

def plantedclique_acc_pyg(data):
    pred = torch.sigmoid(data.x) >= 0.5

    return torch.mean((pred.float() == data.y).float())


### MDS ###

def is_ds(ds, row, col):
    agg = scatter(ds.float()[row], index=col, reduce='sum')
    visited = agg >= 1.0

    return all(visited)

def mds_size_pyg(data, num_seeds: int = 1, enable: bool = True, test: bool = False):
    if not test:
        num_seeds = 1
        if not enable:
            return torch.tensor(float('nan'))

    data_list = data.to_data_list()

    ds_list = []
    for data in data_list:
        edge_index = add_self_loops(data.edge_index)[0]
        row, col = edge_index[0], edge_index[1]

        mds_size_list = []
        for skip in range(num_seeds):
            ds = torch.zeros_like(data.x).squeeze()
            p = deepcopy(data.x).squeeze()

            if skip > 0:
                for _ in range(skip):
                    idx = torch.argmax(p)
                    p[idx] = - torch.inf

            t0 = time.time()
            while not is_ds(ds, row, col):
                if torch.max(p) == - torch.inf:
                    break   # Break in case skipping top nodes prohibits finding a ds; should prevent infinite loops

                idx = torch.argmax(p)
                ds[idx] = True
                p[idx] = - torch.inf

            if is_ds(ds, row, col):
                mds_size_list.append(ds.sum())
            else:
                # This case should rarely happen (only if break is triggered above).
                # But let's be conservative just in case and set the ds to the entire node set
                mds_size_list.append(len(p))

        ds_list.append(min(mds_size_list))

    return torch.Tensor(ds_list).mean()


def mds_size_pyg_parallel(data, num_seeds: int = 1, enable: bool = True):
    """
    Optimized parallel version that processes all seeds simultaneously using vectorized operations.
    Computes the Minimum Dominating Set size baseline using a greedy strategy guided by model scores (data.x).
    """
    if not enable:
        return torch.tensor(float('nan'))

    data_list = data.to_data_list()

    ds_list = []

    for graph in data_list:
        edge_index = add_self_loops(graph.edge_index)[0]
        row, col = edge_index[0], edge_index[1]
        num_nodes = graph.x.size(0)
        device = graph.x.device

        ds = torch.zeros(num_seeds, num_nodes, dtype=torch.bool, device=device)
        p = graph.x.squeeze().unsqueeze(0).expand(num_seeds, -1).clone()  # [num_seeds, num_nodes]

        for skip in range(num_seeds):
            if skip > 0:
                p_seed = p[skip].clone()
                for _ in range(skip):
                    if torch.max(p_seed) == -torch.inf:
                        break
                    idx = torch.argmax(p_seed)
                    p_seed[idx] = -torch.inf
                p[skip] = p_seed

        def is_ds_batch(ds_batch, row, col):
            """Check dominating set validity for all seeds in parallel."""
            if row.numel() == 0:
                return torch.ones(ds_batch.size(0), dtype=torch.bool, device=ds_batch.device)
            
            # ds_batch: [num_seeds, num_nodes]
            # For each node, check if it's dominated (has a neighbor in ds or is in ds itself)
            # Using scatter: for each node, sum up if any of its neighbors (or itself) are in ds
            num_seeds_batch = ds_batch.size(0)
            num_nodes_batch = ds_batch.size(1)
            
            # Float for scatter op
            ds_float = ds_batch.float()
            
            # Expand row and col for batch processing
            # row: [num_edges], col: [num_edges]
            # We need to scatter for each seed: ds_float[:, row] gives [num_seeds, num_edges]
            ds_at_edges = ds_float[:, row]  # [num_seeds, num_edges]
            
            # Reshape for scatter: flatten seeds and edges, then reshape back
            # scatter expects 1D index, so we need to offset indices for each seed
            # Create offset indices: seed_idx * num_nodes_batch for each seed
            seed_offsets = torch.arange(num_seeds_batch, device=device).unsqueeze(1) * num_nodes_batch  # [num_seeds, 1]
            col_offset = col.unsqueeze(0) + seed_offsets  # [num_seeds, num_edges]
            
            # Flatten for scatter
            ds_flat = ds_at_edges.flatten()  # [num_seeds * num_edges]
            col_flat = col_offset.flatten()  # [num_seeds * num_edges]
            
            # Scatter: aggregate for each (seed, node) pair
            agg_flat = scatter(ds_flat, index=col_flat, dim_size=num_seeds_batch * num_nodes_batch, reduce='sum')
            agg = agg_flat.view(num_seeds_batch, num_nodes_batch)  # [num_seeds, num_nodes]
            
            # Check if all nodes are visited (dominated) for each seed
            visited = agg >= 1.0  # [num_seeds, num_nodes]
            valid_mask = visited.all(dim=1)  # [num_seeds]
            
            return valid_mask

        # Greedy loop: continue until all seeds have valid dominating sets
        max_iterations = num_nodes  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            # Check which seeds still need more nodes
            ds_valid = is_ds_batch(ds, row, col)  # [num_seeds]
            
            if ds_valid.all():
                break
            
            # For seeds that are not yet valid, find the next node to add
            p_masked = p.clone()
            # Set probabilities to -inf for nodes already in the dominating set
            p_masked[ds] = -torch.inf
            
            # For seeds that are already valid, set all to -inf so they don't add more nodes
            p_masked[ds_valid.unsqueeze(1).expand(-1, num_nodes)] = -torch.inf
            
            # Find the next node to add for each seed using vectorized argmax
            max_vals, next_nodes = p_masked.max(dim=1)  # [num_seeds], [num_seeds]
            active_seeds = (max_vals > -torch.inf) & (~ds_valid)  # [num_seeds]
            
            if not active_seeds.any():
                break
            
            # Add the selected nodes to the dominating sets for active seeds
            ds[active_seeds, next_nodes[active_seeds]] = True
            
            # Update probabilities: set selected nodes to -inf for active seeds
            p[active_seeds, next_nodes[active_seeds]] = -torch.inf
            
            iteration += 1

        # Final check and compute sizes
        ds_valid = is_ds_batch(ds, row, col)  # [num_seeds]
        ds_sizes = ds.sum(dim=1).float()  # [num_seeds]
        
        # For seeds that didn't find a valid DS, use num_nodes as fallback
        ds_sizes[~ds_valid] = num_nodes
        
        best_ds_size_for_graph = ds_sizes.min().item()
        ds_list.append(best_ds_size_for_graph)

    return torch.tensor(ds_list, dtype=float).mean()


def mds_acc_pyg(data):
    data_list = data.to_data_list()

    ds_list = []
    for data in data_list:
        p = deepcopy(data.x).squeeze()
        edge_index = add_self_loops(data.edge_index)[0]
        row, col = edge_index[0], edge_index[1]

        ds = (data.x >= 0.5).squeeze()

        p[ds] = - torch.inf

        while not is_ds(ds, row, col):
            idx = torch.argmax(p)
            ds[idx] = True
            p[idx] = - torch.inf

        if is_ds(ds, row, col):
            ds_list.append(True)
        else:
            ds_list.append(False)

    return torch.Tensor(ds_list).mean()


### MIS ###

def mis_size_pyg(batch, dec_length=300, num_seeds=1, complement=False):
    batch = mis_decoder_pyg_parallel(batch, dec_length=dec_length, num_seeds=num_seeds, complement=complement)

    data_list = batch.to_data_list()

    size_list = [data.is_size for data in data_list]

    return torch.Tensor(size_list).mean()


def mis_decoder_pyg(batch, dec_length=300, num_seeds=1):
    data_list = batch.to_data_list()

    for data in data_list:
        is_size_list = []

        for seed in range(num_seeds):

            order = torch.argsort(data.x, dim=0, descending=True)
            c = torch.zeros_like(data.x)

            edge_index = remove_self_loops(data.edge_index)[0]
            src, dst = edge_index[0], edge_index[1]

            c[order[seed]] = 1
            for idx in range(seed, min(dec_length, data.num_nodes)):
                c[order[idx]] = 1

                cTWc = torch.sum(c[src] * c[dst])
                if cTWc != 0:
                    c[order[idx]] = 0

            is_size_list.append(c.sum())

        data.is_size = max(is_size_list)

    return Batch.from_data_list(data_list)


def mis_decoder_pyg_parallel(batch, dec_length=300, num_seeds=1, complement=False):
    """
    Optimized parallel decoder that processes all seeds simultaneously using vectorized operations.
    This avoids sequential processing and leverages PyTorch's native parallelization.
    """
    data_list = batch.to_data_list()

    for data in data_list:
        order = torch.argsort(data.x, dim=0, descending=True).squeeze()
        edge_index = data.edge_index_c if complement else data.edge_index
        edge_index = remove_self_loops(edge_index)[0]
        src, dst = edge_index[0], edge_index[1]
        num_nodes = data.num_nodes
        max_idx = min(dec_length, num_nodes)
        device = data.x.device
        
        # Initialize independent set masks for all seeds: [num_seeds, num_nodes]
        c = torch.zeros(num_seeds, num_nodes, dtype=torch.float32, device=device)
        
        # Initialize seed positions: each seed starts at a different position
        seed_positions = torch.arange(num_seeds, device=device)
        seed_positions = torch.clamp(seed_positions, max=max_idx - 1)
        
        # Set initial node for each seed
        c[torch.arange(num_seeds, device=device), order[seed_positions]] = 1.0
        
        # Process nodes in order for all seeds simultaneously
        for idx in range(max_idx):
            # Determine which seeds should consider this node
            # A seed considers node idx if idx >= seed_position[seed]
            active_seeds = idx >= seed_positions
            
            if not active_seeds.any():
                continue
            
            # For active seeds, try adding this node
            node_idx = order[idx]
            
            # Temporarily add node_idx to all active seeds
            # For MIS, we check if cTWc == 0 (no edges between selected nodes)
            c_temp = c.clone()
            c_temp[active_seeds, node_idx] = 1.0
            
            # Compute cTWc for all seeds in parallel using batched operations
            # cTWc[seed] = sum(c_temp[seed, src] * c_temp[seed, dst])
            # If cTWc != 0, there are edges between selected nodes, so it's not an independent set
            c_src = c_temp[:, src]  # [num_seeds, num_edges]
            c_dst = c_temp[:, dst]  # [num_seeds, num_edges]
            cTWc = (c_src * c_dst).sum(dim=1)  # [num_seeds]
            
            # Valid if cTWc == 0 (no edges between selected nodes)
            valid_mask = (cTWc == 0) & active_seeds  # [num_seeds]
            
            # Update c only for seeds where the node addition is valid
            c[valid_mask, node_idx] = 1.0
        
        # Compute independent set sizes for all seeds
        c_sizes = c.sum(dim=1)  # [num_seeds]
        data.is_size = c_sizes.max().item()

    return Batch.from_data_list(data_list)


def mis_size_threshold_pyg(batch, threshold=0.5):
    """
    Simple threshold-based MIS decoder (no feasibility enforcement).
    Mirrors the Amazon co-with-gnns-example postprocessing: threshold at given
    value, report size and violation count as-is without fixing violations.

    Returns mean IS size across the batch (may include infeasible solutions).
    """
    data_list = batch.to_data_list()

    size_list = []
    for data in data_list:
        bitstring = (data.x.squeeze() >= threshold).float()
        size_list.append(bitstring.sum().item())

    return torch.tensor(size_list).mean()


def mis_violations_threshold_pyg(batch, threshold=0.5):
    """
    Count independence violations for a threshold-based decoder.
    A violation is an edge (u,v) where both u and v are selected.
    """
    data_list = batch.to_data_list()

    violation_list = []
    for data in data_list:
        bitstring = (data.x.squeeze() >= threshold).float()
        edge_index = remove_self_loops(data.edge_index)[0]
        src, dst = edge_index[0], edge_index[1]
        violations = (bitstring[src] * bitstring[dst]).sum().item() / 2
        violation_list.append(violations)

    return torch.tensor(violation_list).mean()


### Min Vertex Cover ###

def is_vc(mask, edge_index):
    """
    Checks if the boolean mask corresponds to a valid Vertex Cover.
    A set is a VC if for every edge (u, v), u is in Set OR v is in Set.
    """
    if edge_index.numel() == 0:  # Handle empty graph case
        return True

    row, col = edge_index
    # Get boolean of whether the edge endpoints are in the mask
    row_selected = mask[row]
    col_selected = mask[col]

    # An edge is covered if EITHER endpoint is selected
    edge_covered = row_selected | col_selected

    return edge_covered.all()


def mvc_size_pyg(data, num_seeds: int = 1, enable: bool = True, test: bool = False):
    """
    Computes the Minimum Vertex Cover size baseline using a greedy strategy
    guided by model scores (data.x).
    """
    if not test:
        num_seeds = 1
        if not enable:
            return torch.tensor(float('nan'))

    data_list = data.to_data_list()

    vc_size_list_batch = []

    for graph in data_list:
        # CHANGE 1: We use the original edge_index.
        # No need for add_self_loops for MVC checks.
        edge_index = graph.edge_index

        # Determine number of nodes from features x
        num_nodes = graph.x.size(0)

        best_vc_size_for_graph = float('inf')

        # CHANGE 2: If the graph has no edges, the Min Vertex Cover is empty (size 0).
        if edge_index.numel() == 0:
            vc_size_list_batch.append(0)
            continue

        for skip in range(num_seeds):
            # vc will be a boolean mask of selected nodes
            vc = torch.zeros(num_nodes, dtype=torch.bool, device=graph.x.device)
            p = deepcopy(graph.x).squeeze()

            # Skip Logic (Same as MDS)
            if skip > 0:
                for _ in range(skip):
                    # If p is all -inf, we can't skip anymore
                    if torch.max(p) == -float('inf'):
                        break
                    idx = torch.argmax(p)
                    p[idx] = -float('inf')

            # Greedy Loop
            # CHANGE 3: The condition is now based on Edge Coverage
            while not is_vc(vc, edge_index):
                if torch.max(p) == -float('inf'):
                    break  # Should not happen unless graph is disconnected/weird states

                idx = torch.argmax(p)
                vc[idx] = True
                p[idx] = -float('inf')

            # Verify and Record
            if is_vc(vc, edge_index):
                best_vc_size_for_graph = min(best_vc_size_for_graph, vc.sum().item())
            else:
                # Fallback: if loop broke without finding VC, take all nodes (valid VC)
                best_vc_size_for_graph = min(best_vc_size_for_graph, num_nodes)

        vc_size_list_batch.append(best_vc_size_for_graph)

    return torch.tensor(vc_size_list_batch, dtype=float).mean()


def mvc_size_pyg_parallel(data, num_seeds: int = 1, enable: bool = True):
    """
    Optimized parallel version that processes all seeds simultaneously using vectorized operations.
    Computes the Minimum Vertex Cover size baseline using a greedy strategy guided by model scores (data.x).
    """
    # if not test:
    #     num_seeds = 1
    if not enable:
        return torch.tensor(float('nan'))

    data_list = data.to_data_list()

    vc_size_list_batch = []

    for graph in data_list:
        edge_index = graph.edge_index
        num_nodes = graph.x.size(0)
        device = graph.x.device

        # If the graph has no edges, the Min Vertex Cover is empty (size 0).
        if edge_index.numel() == 0:
            vc_size_list_batch.append(0)
            continue

        row, col = edge_index[0], edge_index[1]

        # Initialize vertex cover masks for all seeds: [num_seeds, num_nodes]
        vc = torch.zeros(num_seeds, num_nodes, dtype=torch.bool, device=device)
        
        # Initialize probability vectors for all seeds: [num_seeds, num_nodes]
        p = graph.x.squeeze().unsqueeze(0).expand(num_seeds, -1).clone()  # [num_seeds, num_nodes]

        # Skip logic: for each seed, skip the top 'skip' nodes
        for skip in range(num_seeds):
            if skip > 0:
                p_seed = p[skip].clone()
                for _ in range(skip):
                    if torch.max(p_seed) == -float('inf'):
                        break
                    idx = torch.argmax(p_seed)
                    p_seed[idx] = -float('inf')
                p[skip] = p_seed

        # Vectorized vertex cover check function
        def is_vc_batch(vc_batch, row, col):
            """Check vertex cover validity for all seeds in parallel."""
            if row.numel() == 0:
                return torch.ones(vc_batch.size(0), dtype=torch.bool, device=vc_batch.device)
            
            # vc_batch: [num_seeds, num_nodes]
            # For each edge, check if either endpoint is in the cover
            row_selected = vc_batch[:, row]  # [num_seeds, num_edges]
            col_selected = vc_batch[:, col]  # [num_seeds, num_edges]
            edge_covered = row_selected | col_selected  # [num_seeds, num_edges]
            
            # All edges must be covered for a valid VC
            return edge_covered.all(dim=1)  # [num_seeds]

        # Greedy loop: continue until all seeds have valid vertex covers
        max_iterations = num_nodes  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            # Check which seeds still need more nodes
            vc_valid = is_vc_batch(vc, row, col)  # [num_seeds]
            
            if vc_valid.all():
                break
            
            # For seeds that are not yet valid, find the next node to add
            p_masked = p.clone()
            # Set probabilities to -inf for nodes already in the cover
            p_masked[vc] = -float('inf')
            
            # For seeds that are already valid, set all to -inf so they don't add more nodes
            p_masked[vc_valid.unsqueeze(1).expand(-1, num_nodes)] = -float('inf')
            
            # Find the next node to add for each seed using vectorized argmax
            max_vals, next_nodes = p_masked.max(dim=1)  # [num_seeds], [num_seeds]
            active_seeds = (max_vals > -float('inf')) & (~vc_valid)  # [num_seeds]
            
            if not active_seeds.any():
                break
            
            # Add the selected nodes to the vertex covers for active seeds
            vc[active_seeds, next_nodes[active_seeds]] = True
            
            # Update probabilities: set selected nodes to -inf for active seeds
            p[active_seeds, next_nodes[active_seeds]] = -float('inf')
            
            iteration += 1

        # Final check and compute sizes
        vc_valid = is_vc_batch(vc, row, col)  # [num_seeds]
        vc_sizes = vc.sum(dim=1).float()  # [num_seeds]
        
        # For seeds that didn't find a valid VC, use num_nodes as fallback
        vc_sizes[~vc_valid] = num_nodes
        
        best_vc_size_for_graph = vc_sizes.min().item()
        vc_size_list_batch.append(best_vc_size_for_graph)

    return torch.tensor(vc_size_list_batch, dtype=float).mean()


### MAXBIPARTITE ###

def maxbipartite_decoder(output, adj, dec_length):
    return maxclique_decoder(output, torch.matrix_power(adj, 2), dec_length)

# def mis_size_pyg(data):

#     # eval = False
#     # if not eval:
#     #     return 0.

#     data_list = data.to_data_list()

#     iset_list = []
#     for data in data_list:
#         p = deepcopy(data.x).squeeze()
#         edge_index = remove_self_loops(data.edge_index)[0]
#         row, col = edge_index[0], edge_index[1]

#         iset = (data.x >= 0.5).squeeze()

#         if is_iset(iset, row, col) and any(iset):
#             p[iset] = - torch.inf

#             while True:
#                 idx = torch.argmax(p)
#                 iset[idx] = True
#                 p[idx] = - torch.inf

#                 if not is_iset(iset, row, col):
#                     iset[idx] = False
#                     break

#             iset_list.append(iset.sum())

#         else:
#             iset = torch.zeros_like(iset)

#             while True:
#                 idx = torch.argmax(p)
#                 iset[idx] = True
#                 p[idx] = - torch.inf

#                 if not is_iset(iset, row, col):
#                     iset[idx] = False
#                     break

#             iset_list.append(iset.sum())

#     return torch.Tensor(iset_list).mean()


# def is_iset(iset, row, col):

#     edges = iset[row] * iset[col]

#     return all(edges == 0.)
