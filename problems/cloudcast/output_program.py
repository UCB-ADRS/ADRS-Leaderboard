# EVOLVE-BLOCK-START
import networkx as nx
import json
import time
from typing import Dict, List, Tuple, Set
import math


def search_algorithm(src, dsts, G, num_partitions):
    """
    Greedy Steiner-like broadcast tree with capacity-aware tie-breaking and selective path splitting.
    - Builds a shared broadcast tree that minimizes incremental cost and reuses edges.
    - Prefers higher-throughput branches when costs tie to reduce max transfer time.
    - Optionally splits a few worst bottleneck destinations across edge-disjoint equal-cost paths.
    """
    # Import here so it's available when the code is loaded by evaluator
    from broadcast import BroadCastTopology
    
    # Sanitize the graph: disallow self-loops, inbound edges to src, invalid cost, and non-positive throughput.
    # Use a read-only view to avoid copying and enable fast iterations. Also create a basic fallback view.
    def edge_filter(u, v):
        if u == v or v == src:
            return False
        d = G[u][v]
        c = d.get("cost", None)
        tp = d.get("throughput", None)
        try:
            c_ok = (c is not None) and math.isfinite(float(c))
        except Exception:
            c_ok = False
        try:
            tp_val = float(tp)
            tp_ok = tp_val > 0.0 and math.isfinite(tp_val)
        except Exception:
            tp_ok = False
        return c_ok and tp_ok

    H = nx.subgraph_view(G, filter_edge=edge_filter)

    # Basic fallback view (only remove inbound-to-src and self-loops).
    def basic_edge_filter(u, v):
        if u == v or v == src:
            return False
        return True

    H_basic = nx.subgraph_view(G, filter_edge=basic_edge_filter)

    # Weight function: cost if available, else a large sentinel to discourage usage
    INF = 1e18

    def edge_cost(u, v):
        d = G[u][v]
        c = d.get("cost", None)
        try:
            return float(c)
        except Exception:
            return INF

    # Track edges selected into the broadcast tree as a set of (u, v)
    tree_edges: Set[Tuple[str, str]] = set()
    connected: Set[str] = set([src])
    remaining_dsts: Set[str] = set(dsts)
    # Map each dst to its base path in the built tree
    base_paths: Dict[str, List[str]] = {}

    # Weight function that treats already-selected edges as zero incremental cost
    def tree_aware_weight(u, v, d):
        if (u, v) in tree_edges:
            return 0.0
        return edge_cost(u, v)

    # Utility to compute bottleneck throughput (min throughput along path)
    def bottleneck_throughput(path_nodes: List[str]) -> float:
        if len(path_nodes) < 2:
            return math.inf
        bn = math.inf
        for a, b in zip(path_nodes, path_nodes[1:]):
            tp = G[a][b].get("throughput", 0.0)
            # Ensure numeric and positive
            try:
                tp = float(tp)
            except Exception:
                return 0.0
            if not math.isfinite(tp) or tp <= 0.0:
                return 0.0
            if tp < bn:
                bn = tp
        return bn

    # Utility to compute path cost (sum of per-edge cost)
    def path_cost(path_nodes: List[str]) -> float:
        total = 0.0
        for a, b in zip(path_nodes, path_nodes[1:]):
            c = edge_cost(a, b)
            if c >= INF:
                return INF
            total += c
        return total

    # Grow the tree until all destinations are connected
    while remaining_dsts:
        # Multi-source Dijkstra from current connected set, with tree-aware weights
        try:
            dist_map, path_map = nx.multi_source_dijkstra(H, list(connected), weight=tree_aware_weight)
        except Exception:
            # Fallback: if multi-source is not available or fails, degrade to single-source from src
            dist_map, path_map = nx.single_source_dijkstra(H, src, weight=tree_aware_weight)

        # Select the next destination to attach: min incremental cost, tie-break by higher bottleneck
        best_dst = None
        best_cost = INF
        best_bn = -1.0
        best_path = None

        for d in list(remaining_dsts):
            if d not in path_map:
                continue
            p = path_map[d]
            inc_cost = 0.0
            # Compute incremental cost explicitly to keep consistent with zero-cost reused edges
            for a, b in zip(p, p[1:]):
                if (a, b) in tree_edges:
                    continue
                c = edge_cost(a, b)
                inc_cost += c
            if inc_cost < best_cost - 1e-12:
                best_cost = inc_cost
                best_dst = d
                best_path = p
                best_bn = bottleneck_throughput(p)
            elif abs(inc_cost - best_cost) <= 1e-12:
                # Tie-break by bottleneck throughput, then by lexicographic dst id for determinism
                bn = bottleneck_throughput(p)
                if bn > best_bn + 1e-12 or (abs(bn - best_bn) <= 1e-12 and (best_dst is None or d < best_dst)):
                    best_cost = inc_cost
                    best_dst = d
                    best_path = p
                    best_bn = bn

        # If some destinations are unreachable in this sanitized view, try direct shortest path as a fallback
        if best_dst is None:
            d = min(remaining_dsts)  # deterministic pick
            try:
                p = nx.dijkstra_path(H, src, d, weight=edge_cost)
            except Exception:
                try:
                    p = nx.dijkstra_path(H_basic, src, d, weight=edge_cost)
                except Exception:
                    # As last resort, return a degenerate path placeholder
                    p = [src, d]
            best_dst, best_path = d, p

        # Merge best_path edges into the tree
        for a, b in zip(best_path, best_path[1:]):
            tree_edges.add((a, b))
            connected.add(a)
            connected.add(b)

        # Record this destination's base path
        base_paths[best_dst] = best_path
        remaining_dsts.remove(best_dst)

    # Now we have a broadcast tree (tree_edges) and base paths for each destination
    # Optionally, split a few worst bottleneck destinations across equal-cost edge-disjoint backups
    # Strict budget: augment at most K=2 destinations
    K = 2
    epsilon_cost = 1e-9  # accept only equal-cost alternatives by default
    # Rank destinations by bottleneck ascending (worst first)
    ranked = sorted(dsts, key=lambda d: bottleneck_throughput(base_paths[d]) if d in base_paths else 0.0)
    augmented: Dict[str, Tuple[List[str], List[str]]] = {}  # dst -> (p1, p2)

    start_aug_time = time.time()
    aug_time_budget = 0.008  # ~8 ms augmentation budget

    for d in ranked[:K]:
        if (time.time() - start_aug_time) > aug_time_budget:
            break
        p1 = base_paths[d]
        c1 = path_cost(p1)

        # First try Suurballe for two edge-disjoint shortest paths of equal cost
        p2 = None
        try:
            paths, _ = nx.suurballe(H, src, d, weight="cost")
            # Filter candidates that match shortest cost and differ from primary
            cand = []
            for pp in paths:
                if pp == p1:
                    continue
                cc = 0.0
                valid = True
                for a, b in zip(pp, pp[1:]):
                    ec = edge_cost(a, b)
                    if ec >= INF:
                        valid = False
                        break
                    cc += ec
                if valid and abs(cc - c1) <= epsilon_cost:
                    cand.append(pp)
            if cand:
                # Choose alternative with highest bottleneck
                p2 = max(cand, key=lambda nodes: bottleneck_throughput(nodes))
        except Exception:
            p2 = None

        # If no Suurballe alt, try removing p1 edges to enforce edge-disjoint equal-cost alternative
        if p2 is None:
            removed = list(zip(p1, p1[1:]))
            H2 = nx.subgraph_view(H, filter_edge=lambda u, v, rem=set(removed): (u, v) not in rem)
            try:
                p2 = nx.dijkstra_path(H2, src, d, weight=edge_cost)
                c2 = path_cost(p2)
                if not (c2 <= c1 + epsilon_cost) or p2 == p1:
                    p2 = None
            except Exception:
                p2 = None

        if p2 is not None:
            augmented[d] = (p1, p2)

    # Prepare output topology
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Cache to reuse identical edge sequences across partitions and destinations
    trip_cache: Dict[Tuple[str, ...], List[List]] = {}

    # Helper to convert node path -> edge-triplet sequence (cached, list-based)
    def path_to_triplets(path_nodes: List[str]) -> List[List]:
        if not path_nodes or len(path_nodes) < 2:
            return []
        key = tuple(path_nodes)
        if key in trip_cache:
            return trip_cache[key]
        seq = [[a, b, G[a][b]] for a, b in zip(path_nodes, path_nodes[1:])]
        trip_cache[key] = seq
        return seq

    # Emit per-partition paths
    set_paths = bc_topology.set_dst_partition_paths

    for d in dsts:
        p = base_paths.get(d, [src, d])
        if d in augmented:
            p1, p2 = augmented[d]
            trip1 = path_to_triplets(p1)
            trip2 = path_to_triplets(p2)
            bn1 = bottleneck_throughput(p1)
            bn2 = bottleneck_throughput(p2)
            # Proportional split using largest remainder; ensure both used when both have positive weight
            w1 = max(0.0, bn1)
            w2 = max(0.0, bn2)
            total_w = w1 + w2
            if total_w <= 0.0:
                n1 = num_partitions
                n2 = 0
            else:
                exact1 = num_partitions * (w1 / total_w)
                n1 = int(math.floor(exact1))
                n2 = num_partitions - n1
                # Distribute remainder to the larger fractional part
                if n1 + n2 < num_partitions:
                    frac1 = exact1 - math.floor(exact1)
                    frac2 = (num_partitions - exact1) - math.floor(num_partitions - exact1)
                    # Compare fractions; assign the last partition accordingly
                    if frac1 >= frac2:
                        n1 += 1
                    else:
                        n2 += 1
                # Ensure both paths used if both have positive weight and enough partitions
                if w1 > 0 and w2 > 0 and num_partitions >= 2:
                    n1 = max(1, min(num_partitions - 1, n1))
                    n2 = num_partitions - n1

            idx = 0
            for _ in range(n1):
                set_paths(d, idx, trip1)
                idx += 1
            for _ in range(n2):
                set_paths(d, idx, trip2)
                idx += 1
        else:
            trip = path_to_triplets(p)
            for j in range(num_partitions):
                set_paths(d, j, trip)

    return bc_topology




def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
    import pandas as pd

    if cost_path is None:
        cost = pd.read_csv("profiles/cost.csv")
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv("profiles/throughput.csv")
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        try:
            tp = float(row["throughput_sent"]) / 1e9
        except Exception:
            tp = 0.0
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * tp)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            try:
                c = float(row["cost"])
            except Exception:
                c = None
            G[row["src"]][row["dest"]]["cost"] = c

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src_r, dst_r = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src_r, dst_r))
    print("Unable to get costs for: ", no_cost_pairs)

    return G
# EVOLVE-BLOCK-END

# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    from broadcast import BroadCastTopology
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)

def run_search_algorithm_timed(idx: int, src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    start_time = time.time()
    result = run_search_algorithm(src, dsts, G, num_partitions)
    end_time = time.time()
    return idx, result, end_time - start_time


# ----------------------------
# Solution class interface
# ----------------------------

import inspect
from typing import Any