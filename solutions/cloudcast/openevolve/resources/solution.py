# EVOLVE-BLOCK-START
import networkx as nx
import json
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    # Import here so it's available when the code is loaded by evaluator
    from broadcast import BroadCastTopology
    
    # Steiner Tree Approximation via Minimum Spanning Arborescence on Metric Closure
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    
    # Weight configuration: Cost primary, throughput secondary
    for u, v, d in h.edges(data=True):
        c = d.get("cost")
        if c is None: c = 1000.0
        t = d.get("throughput", 1e-9)
        if t <= 0: t = 1e-9
        d["weight"] = c + (1e-5 / t)

    # Metric Closure
    terminals = sorted(list(set([src] + dsts)))
    closure = nx.DiGraph()
    path_cache = {}
    
    for u in terminals:
        for v in terminals:
            if u != v and v != src:
                try:
                    p = nx.dijkstra_path(h, u, v, weight="weight")
                    w = nx.path_weight(h, p, weight="weight")
                    closure.add_edge(u, v, weight=w)
                    path_cache[(u, v)] = p
                except nx.NetworkXNoPath:
                    pass

    # MSA
    try:
        arb = nx.minimum_spanning_arborescence(closure, attr="weight")
    except Exception:
        arb = nx.DiGraph()

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Path Reconstruction
    for dst in dsts:
        final_edges = []
        try:
            # Prefer MSA path
            if dst in arb and nx.has_path(arb, src, dst):
                meta_path = nx.shortest_path(arb, src, dst)
                for i in range(len(meta_path) - 1):
                    u_t, v_t = meta_path[i], meta_path[i+1]
                    seg = path_cache[(u_t, v_t)]
                    for k in range(len(seg) - 1):
                        u, v = seg[k], seg[k+1]
                        final_edges.append([u, v, G[u][v]])
            else:
                raise nx.NetworkXNoPath
        except nx.NetworkXNoPath:
            # Fallback to direct path
            try:
                p = nx.dijkstra_path(h, src, dst, weight="weight")
                for k in range(len(p) - 1):
                    u, v = p[k], p[k+1]
                    final_edges.append([u, v, G[u][v]])
            except nx.NetworkXNoPath:
                pass
        
        for j in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, final_edges)

    return bc_topology



def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    if cost_path is None:
        cost_path = "profiles/cost.csv"
    if throughput_path is None:
        throughput_path = "profiles/throughput.csv"

    try:
        cost = pd.read_csv(cost_path)
        throughput = pd.read_csv(throughput_path)
    except Exception:
        return nx.DiGraph()

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    return G


# EVOLVE-BLOCK-END

def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance."""
    from broadcast import BroadCastTopology
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)


import inspect
from typing import Any


class Solution:
    """OpenEvolve-evolved solution for cloudcast broadcast optimization."""

    def __init__(self) -> None:
        pass

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        """
        Solve the cloudcast broadcast optimization problem.
        
        Args:
            spec_path: Path to specification file (optional, not used)
            
        Returns:
            Dict with 'code' containing the search algorithm implementation.
        """
        # Read the file and extract everything before the Solution class
        import pathlib
        file_path = pathlib.Path(__file__)
        content = file_path.read_text()
        
        # Find where the Solution class starts and extract everything before it
        solution_class_start = content.find("\nclass Solution:")
        if solution_class_start == -1:
            # If Solution class not found, extract everything
            code = content
        else:
            # Extract everything up to (but not including) the Solution class
            code = content[:solution_class_start].strip()
        
        return {"code": code}
