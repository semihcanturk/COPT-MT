"""
Graph-level reductions between combinatorial optimization problems.

Implements the following reduction chain to MDS:
  - MIS/MVC → MDS          (mid-edge-vertex gadget)
  - MaxClique → MIS → MDS  (complement graph + mid-edge-vertex gadget)
  - MaxCut → MIS → MDS     (conflict graph + mid-edge-vertex gadget)

Each reduction includes:
  1. A forward transform:  G  →  G'  (build the reduced graph)
  2. A solution recovery:  S' →  S   (map a solution on G' back to G)

Requires: networkx, numpy
"""

import networkx as nx
import numpy as np
from itertools import combinations
from typing import Set, Tuple, Dict, Optional


# =============================================================================
# 1. COMPLEMENT GRAPH (MaxClique ↔ MIS)
# =============================================================================

def complement_graph(G: nx.Graph) -> nx.Graph:
    """
    MaxClique on G ↔ MIS on complement(G).

    A set S is a clique in G iff S is an independent set in G̅.
    Vertex set is preserved; edge set is {(u,v) : (u,v) ∉ E(G)}.

    Parameters
    ----------
    G : nx.Graph
        Original graph.

    Returns
    -------
    G_bar : nx.Graph
        Complement graph with same vertex set.
    """
    G_bar = nx.complement(G)
    # Preserve node attributes (features, labels, etc.)
    for v in G.nodes():
        G_bar.nodes[v].update(G.nodes[v])
    return G_bar

# =============================================================================
# 2. MID-EDGE-VERTEX GADGET (MVC/MIS → MDS)
# =============================================================================

def mid_edge_vertex_gadget(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """
    Reduce Minimum Vertex Cover (or MIS) to Minimum Dominating Set.

    Construction (Garey-Johnson / Mount):
      1. Start with G' := copy of G (all original vertices and edges).
      2. For every edge (u, v) ∈ E(G), add a new "mid-edge" vertex w_{uv}
         and edges (u, w_{uv}), (v, w_{uv}).
      3. Track isolated vertices — they must appear in any dominating set.

    Correctness:
      - VC→DS: G has a vertex cover of size k  ⟺  G' has a dominating set
        of size k + n_I, where n_I = number of isolated vertices in G.
      - MIS→DS: G has an MIS of size k  ⟺  G' has a dominating set of size
        (n - k) + n_I.

    Parameters
    ----------
    G : nx.Graph
        Original graph (the MVC or MIS instance).

    Returns
    -------
    G_prime : nx.Graph
        The reduced MDS graph.
    metadata : dict
        Contains:
        - 'original_vertices': set of original vertex IDs
        - 'mid_edge_map': dict mapping mid-edge vertex ID → (u, v)
        - 'isolated_vertices': set of isolated vertex IDs in G
        - 'n_original': number of original vertices
        - 'n_isolated': number of isolated vertices
    """
    G_prime = G.copy()
    original_vertices = set(G.nodes())
    isolated_vertices = set(nx.isolates(G))

    mid_edge_map = {}  # mid-edge vertex → (u, v) original endpoints

    # Generate unique IDs for mid-edge vertices.
    # Use tuples ('w', u, v) to avoid collision with any integer/string node IDs.
    for u, v in G.edges():
        # Canonical ordering to ensure deterministic naming
        a, b = (min(u, v), max(u, v))
        w = ('w', a, b)

        G_prime.add_node(w, node_type='mid_edge', original_edge=(a, b))
        G_prime.add_edge(a, w)
        G_prime.add_edge(b, w)
        mid_edge_map[w] = (a, b)

    metadata = {
        'original_vertices': original_vertices,
        'mid_edge_map': mid_edge_map,
        'isolated_vertices': isolated_vertices,
        'n_original': len(original_vertices),
        'n_isolated': len(isolated_vertices),
    }

    return G_prime, metadata


def recover_vc_from_ds(ds_solution: Set, metadata: Dict) -> Set:
    """
    Given a dominating set on G' (the gadget graph), recover the
    corresponding vertex cover on G.

    Procedure:
      1. Remove isolated vertices from DS (they had to be there, but aren't
         part of the VC).
      2. For any mid-edge vertex w_{uv} in DS, replace it with one of its
         endpoints (say u). This preserves domination.
      3. The remaining original vertices form the vertex cover.
    """
    mid_edge_map = metadata['mid_edge_map']
    isolated = metadata['isolated_vertices']
    original = metadata['original_vertices']

    vc = set()
    for v in ds_solution:
        if v in isolated:
            continue  # Isolated vertices are forced into DS but not part of VC
        elif v in mid_edge_map:
            # Replace mid-edge vertex with one of its endpoints
            u, w = mid_edge_map[v]
            vc.add(u)
        elif v in original:
            vc.add(v)

    return vc


def recover_mis_from_ds(ds_solution: Set, metadata: Dict) -> Set:
    """
    Given a dominating set on G', recover the corresponding MIS on G.
    MIS = V \\ VC, so we recover the VC first and complement it.
    """
    vc = recover_vc_from_ds(ds_solution, metadata)
    return metadata['original_vertices'] - vc


# =============================================================================
# 3. CONFLICT GRAPH (MaxCut → MIS)
# =============================================================================

def maxcut_to_mis_conflict_graph(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """
    Reduce Maximum Cut to Maximum Independent Set via a conflict graph.

    Construction:
      For each edge e_i = (u, v) ∈ E(G), create two vertices in H:
        - (e_i, 0): representing "u on side 0, v on side 1" (edge is cut)
        - (e_i, 1): representing "u on side 1, v on side 0" (edge is cut)

      Two vertices (e_i, a) and (e_j, b) in H are connected (conflict) iff
      they assign the same original vertex to different sides.

      Also: (e_i, 0) and (e_i, 1) always conflict (can't cut the same edge
      both ways).

    Correctness:
      A maximum independent set in H corresponds to a maximum set of
      consistently cuttable edges — i.e., a maximum cut in G.

    Parameters
    ----------
    G : nx.Graph
        The MaxCut instance.

    Returns
    -------
    H : nx.Graph
        The conflict graph (an MIS instance).
    metadata : dict
        Contains:
        - 'edge_list': list of original edges (indexed)
        - 'original_graph': the original graph G
        - 'vertex_to_side': helper for solution recovery
    """
    edges = list(G.edges())
    H = nx.Graph()
    edge_index = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edges)}

    # Create two vertices per edge
    for i, (u, v) in enumerate(edges):
        H.add_node((i, 0), edge_idx=i, assignment=0, original_edge=(u, v))
        H.add_node((i, 1), edge_idx=i, assignment=1, original_edge=(u, v))
        # Same edge, opposite assignments always conflict
        H.add_edge((i, 0), (i, 1))

    # Build conflict edges efficiently: O(sum of deg(v)^2) instead of O(m^2).
    #
    # For each original vertex v, collect all conflict-graph nodes that
    # involve v, partitioned by which side v is assigned to.  Every cross-
    # partition pair conflicts.
    from collections import defaultdict
    # side_groups[v][s] = list of conflict-graph node IDs where v is on side s
    side_groups = defaultdict(lambda: defaultdict(list))

    for i, (u, v) in enumerate(edges):
        # assignment 0: u → side 0, v → side 1
        side_groups[u][0].append((i, 0))
        side_groups[v][1].append((i, 0))
        # assignment 1: u → side 1, v → side 0
        side_groups[u][1].append((i, 1))
        side_groups[v][0].append((i, 1))

    for v_orig in G.nodes():
        grp = side_groups[v_orig]
        nodes_side0 = grp[0]
        nodes_side1 = grp[1]
        # Every pair across sides conflicts on v_orig
        for na in nodes_side0:
            for nb in nodes_side1:
                if na != nb and na[0] != nb[0]:  # skip same-edge (already added)
                    H.add_edge(na, nb)

    metadata = {
        'edge_list': edges,
        'original_graph': G,
        'n_original_vertices': G.number_of_nodes(),
        'n_original_edges': G.number_of_edges(),
    }

    return H, metadata


def recover_maxcut_from_mis(mis_solution: Set, metadata: Dict) -> Tuple[Set, Set]:
    """
    Given an MIS on the conflict graph H, recover the MaxCut partition.

    Returns
    -------
    (side_0, side_1) : tuple of sets
        The two sides of the partition.
    """
    edges = metadata['edge_list']
    all_vertices = set(metadata['original_graph'].nodes())

    # Read off side assignments from the MIS
    side_assignment = {}  # original vertex → side (0 or 1)

    for (edge_idx, assignment) in mis_solution:
        u, v = edges[edge_idx]
        if assignment == 0:
            side_assignment.setdefault(u, 0)
            side_assignment.setdefault(v, 1)
        else:
            side_assignment.setdefault(u, 1)
            side_assignment.setdefault(v, 0)

    # Vertices not covered by any cut edge can go on either side
    for v in all_vertices:
        if v not in side_assignment:
            side_assignment[v] = 0  # Arbitrary

    side_0 = {v for v, s in side_assignment.items() if s == 0}
    side_1 = {v for v, s in side_assignment.items() if s == 1}

    return side_0, side_1


# =============================================================================
# 4. FULL PIPELINE COMPOSITIONS
# =============================================================================

def maxclique_to_mds(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """
    MaxClique on G → MDS on G'.
    Chain: complement graph → mid-edge-vertex gadget.
    """
    G_bar = complement_graph(G)
    G_prime, gadget_meta = mid_edge_vertex_gadget(G_bar)

    metadata = {
        'stage': 'maxclique_to_mds',
        'complement_graph': G_bar,
        'gadget_metadata': gadget_meta,
        'original_graph': G,
    }
    return G_prime, metadata


def maxcut_to_mds(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """
    MaxCut on G → MDS on G'.
    Chain: conflict graph (MaxCut → MIS) → mid-edge-vertex gadget (MIS → MDS).
    """
    H, conflict_meta = maxcut_to_mis_conflict_graph(G)
    G_prime, gadget_meta = mid_edge_vertex_gadget(H)

    metadata = {
        'stage': 'maxcut_to_mds',
        'conflict_graph': H,
        'conflict_metadata': conflict_meta,
        'gadget_metadata': gadget_meta,
        'original_graph': G,
    }
    return G_prime, metadata


def mis_to_mds(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """
    MIS on G → MDS on G'.
    Uses mid-edge-vertex gadget directly (MIS↔MVC is solution-level).
    """
    G_prime, gadget_meta = mid_edge_vertex_gadget(G)
    metadata = {
        'stage': 'mis_to_mds',
        'gadget_metadata': gadget_meta,
        'original_graph': G,
    }
    return G_prime, metadata


def mvc_to_mds(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """
    MVC on G → MDS on G'.
    Uses mid-edge-vertex gadget directly.
    """
    G_prime, gadget_meta = mid_edge_vertex_gadget(G)
    metadata = {
        'stage': 'mvc_to_mds',
        'gadget_metadata': gadget_meta,
        'original_graph': G,
    }
    return G_prime, metadata


# =============================================================================
# 5. GRAPH STATISTICS (for documenting blow-up)
# =============================================================================

def graph_stats(G: nx.Graph, label: str = "") -> Dict:
    """Return basic statistics for documenting graph blow-up."""
    stats = {
        'label': label,
        'n_vertices': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
    }
    return stats


def print_stats(stats: Dict):
    label = stats['label']
    print(f"  [{label}]  V={stats['n_vertices']:,}  E={stats['n_edges']:,}  "
          f"density={stats['density']:.4f}  avg_deg={stats['avg_degree']:.2f}")


# =============================================================================
# 6. DEMO / SMOKE TEST
# =============================================================================

def demo():
    """
    Run each reduction on a small random graph and print the blow-up
    statistics to verify correctness and measure size expansion.
    """
    np.random.seed(42)

    # --- Small graph for correctness checks ---
    G_small = nx.petersen_graph()  # 10 vertices, 15 edges, 3-regular
    print("=" * 70)
    print("CORRECTNESS CHECK on Petersen graph (n=10, m=15)")
    print("=" * 70)

    # MIS → MDS
    G_mds, meta = mis_to_mds(G_small)
    print("\nMIS → MDS (mid-edge-vertex gadget):")
    print_stats(graph_stats(G_small, "Original"))
    print_stats(graph_stats(G_mds, "MDS gadget"))
    print(f"  Expected: V = n + m = {10 + 15}, E = m + 2m = {15 + 30}")

    # MaxClique → MDS
    G_mds2, meta2 = maxclique_to_mds(G_small)
    G_bar = meta2['complement_graph']
    print("\nMaxClique → MIS (complement) → MDS:")
    print_stats(graph_stats(G_small, "Original"))
    print_stats(graph_stats(G_bar, "Complement"))
    print_stats(graph_stats(G_mds2, "MDS gadget"))

    # MaxCut → MDS
    G_mds3, meta3 = maxcut_to_mds(G_small)
    H_conflict = meta3['conflict_graph']
    print("\nMaxCut → MIS (conflict graph) → MDS:")
    print_stats(graph_stats(G_small, "Original"))
    print_stats(graph_stats(H_conflict, "Conflict graph"))
    print_stats(graph_stats(G_mds3, "MDS gadget"))

    # --- Larger RB-like graphs for blow-up analysis ---
    print("\n" + "=" * 70)
    print("BLOW-UP ANALYSIS on random graphs of increasing size")
    print("=" * 70)

    for n in [20, 50, 100, 200]:
        p = 0.15  # Sparse, roughly RB-density
        G = nx.erdos_renyi_graph(n, p, seed=42)
        m = G.number_of_edges()

        print(f"\n--- Original: n={n}, m={m}, p={p} ---")

        # MIS/MVC → MDS
        G1, _ = mis_to_mds(G)
        print_stats(graph_stats(G1, "MIS→MDS"))

        # MaxClique → MDS
        G2, meta2 = maxclique_to_mds(G)
        G_bar = meta2['complement_graph']
        print_stats(graph_stats(G_bar, "Complement"))
        print_stats(graph_stats(G2, "MaxClique→MDS"))

        # MaxCut → MDS
        G3, meta3 = maxcut_to_mds(G)
        H = meta3['conflict_graph']
        print_stats(graph_stats(H, "Conflict graph"))
        print_stats(graph_stats(G3, "MaxCut→MDS"))

    # --- Verify correctness: brute-force on tiny graph ---
    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION on K4 (brute-force)")
    print("=" * 70)

    K4 = nx.complete_graph(4)

    # MIS of K4: any single vertex (size 1)
    G_mds, meta = mis_to_mds(K4)

    # Find MDS of G_mds by brute force
    best_ds = None
    for size in range(1, G_mds.number_of_nodes() + 1):
        found = False
        for candidate in combinations(G_mds.nodes(), size):
            candidate_set = set(candidate)
            if nx.is_dominating_set(G_mds, candidate_set):
                best_ds = candidate_set
                found = True
                break
        if found:
            break

    print(f"\n  K4: n=4, m=6")
    print(f"  MDS gadget graph: V={G_mds.number_of_nodes()}, E={G_mds.number_of_edges()}")
    print(f"  Minimum DS found (size {len(best_ds)}): {best_ds}")

    # Recover MIS
    recovered_mis = recover_mis_from_ds(best_ds, meta['gadget_metadata'])
    print(f"  Recovered MIS: {recovered_mis}")

    # Verify: MIS of K4 should be any single vertex
    is_independent = all(
        not K4.has_edge(u, v)
        for u in recovered_mis for v in recovered_mis if u != v
    )
    print(f"  Is independent set? {is_independent}")
    print(f"  Expected MIS size for K4: 1, got: {len(recovered_mis)}")


if __name__ == "__main__":
    demo()
