import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import math


# ----------------------------
# Broadcast topology data types
# ----------------------------


# ----------------------------
# Graph building helper (optional)
# ----------------------------

def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
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
        # throughput_sent assumed to be in bits/sec; divide by 1e9 to Gbps
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1].get("cost", None) is None:
            no_cost_pairs.append((src, dst))
    if no_cost_pairs:
        # Not raising; evaluator may accept missing costs as bad edges
        pass

    return G


# ----------------------------
# Utility helpers
# ----------------------------

PROVIDER_LIMITS = {
    "aws": {"ingress": 5.0, "egress": 10.0},
    "gcp": {"ingress": 7.0, "egress": 16.0},
    "azure": {"ingress": 16.0, "egress": 16.0},
}


def _provider_of(node: str) -> str:
    if not isinstance(node, str) or ":" not in node:
        return ""
    return node.split(":", 1)[0].lower()


def _edge_cost(G: nx.DiGraph, u: str, v: str) -> float:
    data = G[u][v]
    c = data.get("cost", None)
    if c is None:
        return float("inf")
    return float(c)


def _edge_thr(G: nx.DiGraph, u: str, v: str) -> float:
    data = G[u][v]
    t = data.get("throughput", None)
    if t is None or t <= 0:
        return 0.0
    return float(t)


def _prov_egress_cap(node: str) -> float:
    prov = _provider_of(node)
    return PROVIDER_LIMITS.get(prov, {}).get("egress", float("inf"))


def _prov_ingress_cap(node: str) -> float:
    prov = _provider_of(node)
    return PROVIDER_LIMITS.get(prov, {}).get("ingress", float("inf"))


def _edge_effective_thr(G: nx.DiGraph, u: str, v: str) -> float:
    # Edge throughput additionally limited by egress of u's provider and ingress of v's provider
    t = _edge_thr(G, u, v)
    e = _prov_egress_cap(u)
    i = _prov_ingress_cap(v)
    return max(0.0, min(t, e if e > 0 else float("inf"), i if i > 0 else float("inf")))


def _edge_effective_thr_wrapper(u: str, v: str, data: dict) -> float:
    # When only edge data is available during path search, approximate provider caps
    t = data.get("throughput", 0.0)
    e = _prov_egress_cap(u)
    i = _prov_ingress_cap(v)
    return max(0.0, min(float(t), e if e > 0 else float("inf"), i if i > 0 else float("inf")))


def _path_edges(path: List[str]) -> List[Tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _path_cost(G: nx.DiGraph, path: List[str]) -> float:
    return sum(_edge_cost(G, u, v) for u, v in _path_edges(path))


def _path_effective_thr(G: nx.DiGraph, path: List[str]) -> float:
    if len(path) < 2:
        return 0.0
    return min((_edge_effective_thr(G, u, v) for u, v in _path_edges(path)), default=0.0)


def _first_hop_edges(T: nx.DiGraph, src: str) -> List[Tuple[str, str]]:
    if src not in T:
        return []
    return [(src, v) for _, v in T.out_edges(src)]


def _first_hop_rate(G: nx.DiGraph, T: nx.DiGraph, src: str) -> float:
    hops = _first_hop_edges(T, src)
    if not hops:
        return 0.0
    return min(_edge_effective_thr(G, u, v) for (u, v) in hops)


def _clean_graph(G: nx.DiGraph, src: str) -> nx.DiGraph:
    H = G.copy()
    try:
        H.remove_edges_from(list(nx.selfloop_edges(H)))
    except Exception:
        pass

    if src in H:
        H.remove_edges_from(list(H.in_edges(src)))

    bad_edges = []
    for u, v, data in list(H.edges(data=True)):
        c = data.get("cost", None)
        t = data.get("throughput", None)
        if c is None or t is None or t <= 0:
            bad_edges.append((u, v))
    if bad_edges:
        H.remove_edges_from(bad_edges)
    return H


def _empirical_cost_stats(H: nx.DiGraph) -> Tuple[float, float, float]:
    costs = [float(d.get("cost", 0.0)) for _, _, d in H.edges(data=True) if d.get("cost", None) is not None]
    if not costs:
        return 0.02, 0.05, 0.1
    costs_sorted = sorted(costs)
    n = len(costs_sorted)

    def pct(p):
        idx = min(n - 1, max(0, int(math.ceil(p * n) - 1)))
        return costs_sorted[idx]

    p50 = pct(0.5)
    p90 = pct(0.9)
    p95 = pct(0.95)
    return p50, p90, p95


def _weight_fn_factory(alpha: float, high_cost_threshold: float = None, high_cost_penalty: float = 0.0):
    """
    Returns a dijkstra weight function that trades cost vs throughput:
    weight = cost + alpha / throughput, with an extra additive penalty for very expensive edges.
    """
    def weight(u, v, data):
        c = data.get("cost", None)
        t = data.get("throughput", None)
        if c is None or t is None or t <= 0:
            return float("inf")
        eff_t = _edge_effective_thr_wrapper(u, v, data)
        if eff_t <= 0:
            return float("inf")
        w = float(c) + float(alpha) / float(eff_t)
        if high_cost_threshold is not None and float(c) > high_cost_threshold:
            w += high_cost_penalty * (float(c) - high_cost_threshold)
        return w
    return weight


def _throughput_pref_weight(beta_cost: float = 0.02):
    """
    Prefer high-throughput routes: weight = (1 / eff_throughput) + beta_cost * cost
    """
    def weight(u, v, data):
        c = data.get("cost", None)
        t = data.get("throughput", None)
        if c is None or t is None or t <= 0:
            return float("inf")
        eff_t = _edge_effective_thr_wrapper(u, v, data)
        if eff_t <= 0:
            return float("inf")
        return (1.0 / eff_t) + beta_cost * float(c)
    return weight


def _tree_paths(T: nx.DiGraph, src: str, dsts: List[str]) -> Dict[str, List[str]]:
    paths = {}
    for d in dsts:
        try:
            p = nx.shortest_path(T, source=src, target=d)
            paths[d] = p
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return paths


def _eval_tree(G: nx.DiGraph, T: nx.DiGraph, src: str, dsts: List[str]) -> Tuple[float, float, float, Dict[str, List[str]]]:
    """
    Returns:
      - total cost per partition (sum of edge costs in tree)
      - worst per-destination path throughput (considering provider caps)
      - first-hop throughput from src (cap for this tree)
      - dict dst -> path nodes in T
    """
    if T.number_of_edges() == 0:
        return float("inf"), 0.0, 0.0, {}

    cost_per_partition = 0.0
    for u, v in T.edges():
        cost_per_partition += _edge_cost(G, u, v)

    paths = _tree_paths(T, src, dsts)
    if not paths or len(paths) < len([d for d in dsts if d in G]):
        # If not all feasible dsts are connected, evaluator might still accept if later filled; return partial safely
        pass

    worst_thr = float("inf")
    for d, p in paths.items():
        thr = _path_effective_thr(G, p)
        if thr <= 0:
            return float("inf"), 0.0, 0.0, {}
        worst_thr = min(worst_thr, thr)
    if worst_thr == float("inf"):
        worst_thr = 0.0

    fh_rate = _first_hop_rate(G, T, src)
    return cost_per_partition, worst_thr, fh_rate, paths


def _add_path_edges(T: nx.DiGraph, H: nx.DiGraph, path: List[str]):
    for (u, v) in _path_edges(path):
        if not T.has_edge(u, v):
            T.add_edge(u, v, **H[u][v])


# ----------------------------
# Steiner-like tree builders
# ----------------------------

def _build_greedy_steiner_tree(H: nx.DiGraph, src: str, dsts: List[str],
                               alpha: float = 0.0,
                               high_cost_threshold: Optional[float] = None,
                               high_cost_penalty: float = 0.0,
                               seed_nodes: Optional[Set[str]] = None,
                               seed_path: Optional[List[str]] = None,
                               force_first_hop: Optional[str] = None,
                               prefer_within_provider: bool = False) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
    """
    Greedy directed Steiner-like arborescence with options:
    - alpha: cost/throughput trade-off
    - high_cost_threshold / penalty: discourage very expensive edges
    - seed_nodes: nodes pre-connected in T
    - seed_path: path to add initially (e.g., to a hard destination)
    - force_first_hop: fix first hop neighbor from src
    - prefer_within_provider: mild bias to stay within provider
    """
    if src not in H:
        return nx.DiGraph(), {}

    remaining: Set[str] = {d for d in dsts if d in H}
    if not remaining:
        return nx.DiGraph(), {}

    T = nx.DiGraph()
    connected_nodes: Set[str] = set([src])

    if seed_nodes:
        connected_nodes.update({n for n in seed_nodes if n in H})

    if seed_path is not None and len(seed_path) >= 2:
        _add_path_edges(T, H, seed_path)
        connected_nodes.update(seed_path)
        # remove any destinations reached already
        for d in list(remaining):
            if d in seed_path:
                remaining.discard(d)

    if force_first_hop is not None and H.has_edge(src, force_first_hop):
        T.add_edge(src, force_first_hop, **H[src][force_first_hop])
        connected_nodes.add(force_first_hop)

    base_weight = _weight_fn_factory(alpha, high_cost_threshold, high_cost_penalty)

    def weight(u, v, data):
        w = base_weight(u, v, data)
        if prefer_within_provider and _provider_of(u) == _provider_of(v):
            w *= 0.96
        return w

    # Iteratively attach destinations
    while remaining:
        best_dst = None
        best_path = None
        best_inc_weight = float("inf")

        for d in list(remaining):
            for s in list(connected_nodes):
                if s not in H:
                    continue
                try:
                    path = nx.dijkstra_path(H, s, d, weight=weight)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                inc = 0.0
                for (u, v) in _path_edges(path):
                    if not T.has_edge(u, v):
                        inc += weight(u, v, H[u][v])
                if inc < best_inc_weight:
                    best_inc_weight = inc
                    best_path = path
                    best_dst = d

        if best_path is None or best_dst is None:
            break

        _add_path_edges(T, H, best_path)
        connected_nodes.update(best_path)
        remaining.discard(best_dst)

    paths_in_T = _tree_paths(T, src, dsts)
    return T, paths_in_T


def _build_provider_gateway_tree(H: nx.DiGraph, src: str, dsts: List[str],
                                 alpha_inside: float = 0.0,
                                 high_cost_threshold: Optional[float] = None,
                                 high_cost_penalty: float = 0.0,
                                 prefer_within_provider: bool = False) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
    """
    Try to enter each provider via a cheap gateway path from src, then fan out inside it.
    """
    if src not in H:
        return nx.DiGraph(), {}

    providers = {}
    for d in dsts:
        p = _provider_of(d)
        if d in H:
            providers.setdefault(p, []).append(d)
    if not providers:
        return nx.DiGraph(), {}

    weight_cost_only = _weight_fn_factory(alpha=0.0, high_cost_threshold=high_cost_threshold, high_cost_penalty=high_cost_penalty)

    T = nx.DiGraph()
    connected: Set[str] = set([src])

    # choose gateway per provider
    for prov, _ in providers.items():
        prov_nodes = [n for n in H.nodes if _provider_of(n) == prov]
        best_path = None
        best_cost = float("inf")
        # try a limited set of nodes for efficiency
        for n in prov_nodes[:1000]:
            try:
                pth = nx.dijkstra_path(H, src, n, weight=weight_cost_only)
                c = _path_cost(H, pth)
                if c < best_cost:
                    best_cost = c
                    best_path = pth
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if best_path is not None:
            _add_path_edges(T, H, best_path)
            connected.update(best_path)

    def inside_weight(u, v, data):
        base = _weight_fn_factory(alpha_inside, high_cost_threshold, high_cost_penalty)(u, v, data)
        if prefer_within_provider and _provider_of(u) == _provider_of(v):
            base *= 0.95
        return base

    remaining: Set[str] = {d for d in dsts if d in H}
    while remaining:
        best_dst, best_path, best_inc = None, None, float("inf")
        for d in list(remaining):
            for s in list(connected):
                try:
                    path = nx.dijkstra_path(H, s, d, weight=inside_weight)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                inc = 0.0
                for (u, v) in _path_edges(path):
                    if not T.has_edge(u, v):
                        inc += inside_weight(u, v, H[u][v])
                if inc < best_inc:
                    best_dst, best_path, best_inc = d, path, inc
        if best_dst is None:
            break
        _add_path_edges(T, H, best_path)
        connected.update(best_path)
        remaining.discard(best_dst)

    paths_in_T = _tree_paths(T, src, dsts)
    return T, paths_in_T


def _fill_missing_paths(H: nx.DiGraph, src: str, dsts: List[str], T: nx.DiGraph,
                        paths_in_T: Dict[str, List[str]], use_alpha: float,
                        high_cost_threshold: Optional[float], high_cost_penalty: float,
                        prefer_within_provider: bool = False) -> Dict[str, List[str]]:
    base_w = _weight_fn_factory(use_alpha, high_cost_threshold, high_cost_penalty)

    def weight(u, v, data):
        w = base_w(u, v, data)
        if prefer_within_provider and _provider_of(u) == _provider_of(v):
            w *= 0.97
        return w

    out = dict(paths_in_T)
    for d in dsts:
        if d not in H:
            continue
        if d in out:
            continue
        try:
            p = nx.dijkstra_path(H, src, d, weight=weight)
            out[d] = p
            _add_path_edges(T, H, p)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    return out


# ----------------------------
# Candidate generation and selection
# ----------------------------

class CandidateTree:
    def __init__(self, name: str, T: nx.DiGraph, paths_in_T: Dict[str, List[str]],
                 cost_per_partition: float, worst_thr: float, first_hop_thr: float, src: str, G: nx.DiGraph):
        self.name = name
        self.T = T
        self.paths_in_T = paths_in_T
        self.cost_per_partition = float(cost_per_partition)
        self.worst_thr = float(worst_thr)
        self.first_hop_thr = float(first_hop_thr)
        self.tree_thr = float(min(self.worst_thr, self.first_hop_thr))
        self.src = src
        self.G = G

    def efficiency(self) -> float:
        if self.cost_per_partition <= 0 or not math.isfinite(self.cost_per_partition):
            return 0.0
        return (self.tree_thr) / self.cost_per_partition

    def src_first_hops(self) -> List[Tuple[str, str]]:
        return _first_hop_edges(self.T, self.src)


def _generate_candidates(H: nx.DiGraph, G: nx.DiGraph, src: str, dsts: List[str]) -> List[CandidateTree]:
    feasible_dsts = [d for d in dsts if d in H]
    if not feasible_dsts:
        return []

    p50, p90, p95 = _empirical_cost_stats(H)

    # weights
    alpha_fast = max(0.02, 3.0 * p50)
    alpha_inside = max(0.0, 1.25 * p50)
    high_cost_penalty = 2.0

    candidates: List[CandidateTree] = []

    def package_candidate(name: str, T: nx.DiGraph, paths: Dict[str, List[str]]):
        c_cost, c_thr, c_fh, _ = _eval_tree(H, T, src, feasible_dsts)
        if math.isfinite(c_cost) and c_thr > 0 and c_fh > 0:
            candidates.append(CandidateTree(name, T, paths, c_cost, c_thr, c_fh, src, H))

    # 1) Cheapest-biased tree
    cheap_T, cheap_paths = _build_greedy_steiner_tree(H, src, feasible_dsts,
                                                      alpha=0.0,
                                                      high_cost_threshold=p95,
                                                      high_cost_penalty=high_cost_penalty,
                                                      prefer_within_provider=True)
    cheap_paths = _fill_missing_paths(H, src, feasible_dsts, cheap_T, cheap_paths,
                                      use_alpha=0.0,
                                      high_cost_threshold=p95,
                                      high_cost_penalty=high_cost_penalty,
                                      prefer_within_provider=True)
    package_candidate("cheap", cheap_T, cheap_paths)

    # 2) Throughput-aware tree
    fast_T, fast_paths = _build_greedy_steiner_tree(H, src, feasible_dsts,
                                                    alpha=alpha_fast,
                                                    high_cost_threshold=p95,
                                                    high_cost_penalty=high_cost_penalty)
    fast_paths = _fill_missing_paths(H, src, feasible_dsts, fast_T, fast_paths,
                                     use_alpha=alpha_fast,
                                     high_cost_threshold=p95,
                                     high_cost_penalty=high_cost_penalty)
    package_candidate("fast", fast_T, fast_paths)

    # 3) Provider gateway variants
    prov_T, prov_paths = _build_provider_gateway_tree(H, src, feasible_dsts,
                                                      alpha_inside=alpha_inside,
                                                      high_cost_threshold=p95,
                                                      high_cost_penalty=high_cost_penalty,
                                                      prefer_within_provider=False)
    prov_paths = _fill_missing_paths(H, src, feasible_dsts, prov_T, prov_paths,
                                     use_alpha=alpha_inside,
                                     high_cost_threshold=p95,
                                     high_cost_penalty=high_cost_penalty)
    package_candidate("provider_gateway", prov_T, prov_paths)

    prov_in_T, prov_in_paths = _build_provider_gateway_tree(H, src, feasible_dsts,
                                                            alpha_inside=alpha_inside,
                                                            high_cost_threshold=p95,
                                                            high_cost_penalty=high_cost_penalty,
                                                            prefer_within_provider=True)
    prov_in_paths = _fill_missing_paths(H, src, feasible_dsts, prov_in_T, prov_in_paths,
                                        use_alpha=alpha_inside,
                                        high_cost_threshold=p95,
                                        high_cost_penalty=high_cost_penalty,
                                        prefer_within_provider=True)
    package_candidate("provider_gateway_intra_bias", prov_in_T, prov_in_paths)

    # 4) Forced-first-hop top-K neighbors by throughput-per-cost
    neighbor_scores = []
    if src in H:
        dst_provider_counts = {}
        for d in feasible_dsts:
            dst_provider_counts[_provider_of(d)] = dst_provider_counts.get(_provider_of(d), 0) + 1
        for _, v, data in H.out_edges(src, data=True):
            c = data.get("cost", None)
            t = _edge_effective_thr_wrapper(src, v, data)
            if c is None or t is None or t <= 0:
                continue
            score = float(t) / (float(c) + 1e-6)
            prov_v = _provider_of(v)
            score *= (1.0 + 0.05 * dst_provider_counts.get(prov_v, 0))
            neighbor_scores.append((score, v))
        neighbor_scores.sort(reverse=True)
    K = 6
    for idx, (_, nb) in enumerate(neighbor_scores[:K]):
        forced_T, forced_paths = _build_greedy_steiner_tree(H, src, feasible_dsts,
                                                            alpha=alpha_fast,
                                                            high_cost_threshold=p95,
                                                            high_cost_penalty=high_cost_penalty,
                                                            force_first_hop=nb)
        forced_paths = _fill_missing_paths(H, src, feasible_dsts, forced_T, forced_paths,
                                           use_alpha=alpha_fast,
                                           high_cost_threshold=p95,
                                           high_cost_penalty=high_cost_penalty)
        package_candidate(f"forced_{idx}_{nb}", forced_T, forced_paths)

    # 5) Hard-destination seeded trees: improve bottleneck paths
    # Determine per-dst high-throughput path and find bottom-M by path throughput on cheapest paths
    dst_thr_estimates: List[Tuple[float, str, List[str]]] = []
    thr_weight = _throughput_pref_weight(beta_cost=0.03)
    for d in feasible_dsts:
        try:
            p_thr = nx.dijkstra_path(H, src, d, weight=thr_weight)
            thr_val = _path_effective_thr(H, p_thr)
            if thr_val <= 0:
                continue
            dst_thr_estimates.append((thr_val, d, p_thr))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    dst_thr_estimates.sort(key=lambda x: x[0])  # slowest first
    M = min(4, len(dst_thr_estimates))
    for i in range(M):
        thr_val, d_hard, p_seed = dst_thr_estimates[i]
        seeded_T, seeded_paths = _build_greedy_steiner_tree(H, src, feasible_dsts,
                                                            alpha=alpha_inside,
                                                            high_cost_threshold=p95,
                                                            high_cost_penalty=high_cost_penalty,
                                                            seed_path=p_seed,
                                                            prefer_within_provider=True)
        seeded_paths = _fill_missing_paths(H, src, feasible_dsts, seeded_T, seeded_paths,
                                           use_alpha=alpha_inside,
                                           high_cost_threshold=p95,
                                           high_cost_penalty=high_cost_penalty,
                                           prefer_within_provider=True)
        package_candidate(f"seeded_hard_{i}_{d_hard}", seeded_T, seeded_paths)

    # Remove dominated candidates: dominated if another has <= cost and >= tree_thr with at least one strict
    filtered: List[CandidateTree] = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            if b.cost_per_partition <= a.cost_per_partition and b.tree_thr >= a.tree_thr and \
               (b.cost_per_partition < a.cost_per_partition or b.tree_thr > a.tree_thr):
                dominated = True
                break
        if not dominated:
            filtered.append(a)

    # Sort primarily by efficiency desc, then by cost asc
    filtered.sort(key=lambda x: (-x.efficiency(), x.cost_per_partition))
    return filtered


def _select_trees_and_allocate_partitions(src: str, P: int, candidates: List[CandidateTree]) -> Tuple[List[CandidateTree], List[int]]:
    """
    Select a subset of candidate trees to maximize throughput while controlling cost.
    Key ideas:
      - Add diverse first-hop trees while aggregate throughput improves significantly.
      - Stop near the source egress cap or when improvements diminish.
      - Assign exactly one partition per selected tree to gain parallelism.
      - Assign all remaining partitions to the single cheapest selected tree (no throughput gain, minimize cost).
    """
    if P <= 0 or not candidates:
        return [], []

    # Source caps
    src_prov = _provider_of(src)
    src_egress_cap = float(PROVIDER_LIMITS.get(src_prov, {}).get("egress", float("inf")))
    if src_egress_cap <= 0:
        src_egress_cap = float("inf")

    # Helper
    def hop_signature(t: CandidateTree) -> Tuple:
        hops = sorted([v for _, v in t.src_first_hops()])
        return tuple(hops[:6])

    # Start from best efficiency candidate
    selected: List[CandidateTree] = []
    used_signatures: Set[Tuple] = set()
    agg_thr = 0.0
    agg_cost_weighted = 0.0  # for diagnostics if needed

    max_trees = min(P, 6)  # cap number of trees to avoid runaway cost
    for cand in candidates:
        if cand.tree_thr <= 0:
            continue

        sig = hop_signature(cand)
        is_diverse = sig not in used_signatures

        # Predict new aggregate thr if added (sum tree thr, capped by source egress)
        new_agg_thr = min(src_egress_cap, agg_thr + cand.tree_thr)
        improvement = (new_agg_thr - agg_thr) / (agg_thr + 1e-9) if agg_thr > 0 else float("inf")

        # Allow adding if:
        # - meaningful improvement (>= 8%), or
        # - still far from source cap (less than 70% utilized) and diverse first-hop
        if (improvement >= 0.08) or (is_diverse and agg_thr < 0.7 * src_egress_cap):
            selected.append(cand)
            used_signatures.add(sig)
            agg_cost_weighted += cand.cost_per_partition * cand.tree_thr
            agg_thr = new_agg_thr

        if len(selected) >= max_trees or agg_thr >= 0.98 * src_egress_cap:
            break

    if not selected:
        selected = [min(candidates, key=lambda x: x.cost_per_partition)]

    # Allocation:
    # - one partition per selected tree to gain throughput
    # - all remaining partitions to the cheapest selected tree (no throughput gain; minimize cost)
    alloc = [1 for _ in selected]
    remaining = P - len(selected)
    if remaining > 0:
        cheapest_idx = min(range(len(selected)), key=lambda i: selected[i].cost_per_partition)
        alloc[cheapest_idx] += remaining

    return selected, alloc


# ----------------------------
# Core algorithm
# ----------------------------

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int):
    """
    Multi-tree broadcast strategy with capacity-aware evaluation and cost-aware partitioning:
    - Build multiple Steiner-like candidate trees with different biases and first-hop diversity.
    - Add trees seeded by high-throughput paths to the slowest destinations.
    - Evaluate each tree with provider caps and first-hop limits.
    - Select a small diverse subset that raises aggregate throughput near the source egress cap.
    - Assign one partition per selected tree to gain parallelism; assign remaining partitions to the cheapest selected tree to minimize cost.
    - Each partition uses a single tree; evaluator handles bandwidth sharing and instance runtime.
    """
    # Import here so it's available when the code is loaded by evaluator
    from broadcast import BroadCastTopology
    
    topology = BroadCastTopology(src, dsts, num_partitions)

    # Defensive checks
    if G is None or not isinstance(G, nx.DiGraph) or src not in G or num_partitions <= 0:
        return topology

    # Clean graph: remove self-loops, bad edges, and incoming edges to src
    H = _clean_graph(G, src)

    feasible_dsts = [d for d in dsts if d in H]
    if not feasible_dsts:
        return topology

    # Generate candidate trees
    candidates = _generate_candidates(H, G, src, feasible_dsts)

    if not candidates:
        # Fallback: per-destination cheapest routes for all partitions
        for part in range(num_partitions):
            for d in feasible_dsts:
                try:
                    p = nx.dijkstra_path(H, src, d, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                for (u, v) in _path_edges(p):
                    edge_data = G[u][v] if u in G and v in G[u] else H[u][v]
                    topology.append_dst_partition_path(d, part, [u, v, edge_data])
        return topology

    # Select subset and allocate partitions
    selected, alloc = _select_trees_and_allocate_partitions(src, num_partitions, candidates)

    if not selected or not alloc or sum(alloc) != num_partitions:
        # Ensure we still return a valid topology
        best = min(candidates, key=lambda x: x.cost_per_partition)
        selected = [best]
        alloc = [num_partitions]

    # Build per-partition paths using selected trees
    part_idx = 0
    for t_idx, cand in enumerate(selected):
        count = alloc[t_idx] if t_idx < len(alloc) else 0
        if count <= 0:
            continue
        paths_in_T = dict(cand.paths_in_T)

        # Ensure we have at least one path to every feasible dst
        for d in feasible_dsts:
            if d not in paths_in_T:
                # try cost-biased path, then throughput-biased fallback
                p_nodes = None
                try:
                    p_nodes = nx.dijkstra_path(H, src, d, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    try:
                        p_nodes = nx.dijkstra_path(H, src, d, weight=_weight_fn_factory(0.0))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        p_nodes = None
                if p_nodes is not None:
                    paths_in_T[d] = p_nodes

        for _ in range(count):
            for d in dsts:
                if d not in feasible_dsts:
                    continue
                path_nodes = paths_in_T.get(d, None)
                if not path_nodes or len(path_nodes) < 2:
                    # Skip unreachable
                    continue
                for (u, v) in _path_edges(path_nodes):
                    edge_data = G[u][v] if u in G and v in G[u] else H[u][v]
                    topology.append_dst_partition_path(d, part_idx, [u, v, edge_data])
            part_idx += 1
            if part_idx >= num_partitions:
                break
        if part_idx >= num_partitions:
            break

    # If some partitions remain (shouldn't happen), attach them to the cheapest selected candidate
    while part_idx < num_partitions:
        best = min(selected, key=lambda x: x.cost_per_partition)
        paths_in_T = best.paths_in_T
        for d in dsts:
            if d not in feasible_dsts:
                continue
            path_nodes = paths_in_T.get(d, None)
            if not path_nodes or len(path_nodes) < 2:
                try:
                    p = nx.dijkstra_path(H, src, d, weight="cost")
                    path_nodes = p
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            for (u, v) in _path_edges(path_nodes):
                edge_data = G[u][v] if u in G and v in G[u] else H[u][v]
                topology.append_dst_partition_path(d, part_idx, [u, v, edge_data])
        part_idx += 1

    return topology


def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance."""
    from broadcast import BroadCastTopology
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)


import inspect
from typing import Any, Dict


class Solution:
    """GEPA-evolved solution for cloudcast broadcast optimization."""

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