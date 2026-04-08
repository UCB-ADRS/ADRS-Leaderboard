# EVOLVE-BLOCK-START
"""Cloud broadcast optimizer (Agent6).

Baseline (Agents 3–5 best-known):
- Directed Steiner arborescence DP (min $/GB edge-sum) for small terminal sets
- Greedy hub-search fallback
- Time-aware post-processing to reduce a makespan proxy

Agent6 changes:
- Make the rebalancer *objective-aware*: allow small cost increases when the
  simulator’s time penalty outweighs them (cost/time trade-off). Moves are pruned
  by a computed max-justifiable cost slack.
- Add optional throughput-weighted DP candidates (cost + beta/throughput) to
  expose alternative low-makespan trees; select the best by estimated
  (data_cost + time_penalty).
"""

import hashlib
import heapq
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Provider egress capacities (in 2Gbps units) used only for tie-breaking.
PROVIDER_EGRESS_UNITS = {"aws": 5.0, "gcp": 7.0, "azure": 16.0}

# Debug toggles (keep False for final submissions)
DEBUG_REBALANCE = False
DEBUG_STRIPING = False
DEBUG_MAIN = False
DEBUG_TARGET = False  # targeted prints for debugging specific configs

# Experimental features (kept conservative by default)
ENABLE_RELAY_SWAP_CANDS = False   # generate alternative AWS-relay variants in inter-cloud cases
ENABLE_MIX2_STRIPING = True      # stripe partitions across 2 candidate trees

# Objective-aware rebalancer knobs
REBALANCE_ALLOW_COST_INCREASE = True
REBALANCE_SLACK_FACTOR = 1.15  # >1 explores slightly-more-expensive edges if time savings justify it
REBALANCE_MAX_TREE_NODES = 180  # safety cap for hidden large graphs

# DP throughput-weighting (beta is in $/GB * Gbps; penalty term is beta/throughput)
DP_EXTRA_BETA_FACTORS = (0.5, 1.0)  # multiplied by (time_cost_per_sec * 8)


# DP safety limits
DP_MAX_TERMINALS = 8
DP_MAX_NODES = 220


def _provider(node: str) -> str:
    if isinstance(node, str) and ":" in node:
        return node.split(":", 1)[0]
    return str(node)


def _edge_cost(G: nx.DiGraph, u: str, v: str) -> float:
    return float(G[u][v].get("cost", 0.0))


def _edge_throughput_gbps(G: nx.DiGraph, u: str, v: str) -> float:
    return float(G[u][v].get("throughput", 0.0))


def _stable_hash01(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _union_edges_from_dst_paths(dst_paths: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
    used: Set[Tuple[str, str]] = set()
    for path in dst_paths.values():
        for u, v in zip(path[:-1], path[1:]):
            used.add((u, v))
    return used


def _estimate_time_ratio(G: nx.DiGraph, used_edges: Set[Tuple[str, str]]) -> float:
    """Very cheap proxy for completion time; used only for tie-breaks."""
    if not used_edges:
        return 0.0

    out_edges: Dict[str, List[Tuple[str, str]]] = {}
    for u, v in used_edges:
        out_edges.setdefault(u, []).append((u, v))

    worst = 0.0
    for u, edges in out_edges.items():
        prov = _provider(u)
        node_units = PROVIDER_EGRESS_UNITS.get(prov, 7.0)

        sum_edge_units = 0.0
        for _, v in edges:
            th = _edge_throughput_gbps(G, u, v)
            if th > 0:
                sum_edge_units += th / 2.0

        effective_units = node_units
        if sum_edge_units > 0:
            effective_units = min(node_units, sum_edge_units)

        if effective_units > 0:
            worst = max(worst, len(edges) / effective_units)

    return worst


def _choose_provider_hubs(h: nx.DiGraph, dsts: List[str], k: int = 1) -> Dict[str, List[str]]:
    """Pick up to k hub candidates per provider by intra-provider centrality."""
    dsts_by_prov: Dict[str, List[str]] = {}
    for d in dsts:
        dsts_by_prov.setdefault(_provider(d), []).append(d)

    hubs: Dict[str, List[str]] = {}
    for prov, terms in dsts_by_prov.items():
        if len(terms) < 2:
            continue

        nodes_p = [n for n in h.nodes if _provider(n) == prov]
        if not nodes_p:
            continue

        sub = h.subgraph(nodes_p).copy()

        scored: List[Tuple[float, str]] = []
        for cand in nodes_p:
            try:
                dist = nx.single_source_dijkstra_path_length(sub, cand, weight="cost")
            except Exception:
                continue

            ok = True
            score = 0.0
            for t in terms:
                if t not in dist:
                    ok = False
                    break
                score += float(dist[t])

            if ok:
                scored.append((score, cand))

        scored.sort(key=lambda x: (x[0], x[1]))
        if scored:
            hubs[prov] = [cand for _, cand in scored[:k]]

    return hubs


def _build_greedy_steiner_subgraph(
    h: nx.DiGraph,
    src: str,
    terminals: Set[str],
    preferred_first: Optional[Set[str]] = None,
    seed: int = 0,
) -> nx.DiGraph:
    """Greedy directed Steiner-like tree construction (Agent1-style)."""
    preferred_first = preferred_first or set()

    remaining: Set[str] = set(terminals)
    remaining.discard(src)

    tree_nodes: Set[str] = {src}
    tree_edges: Set[Tuple[str, str]] = set()

    while remaining:

        def marginal_weight(u, v, attrs):
            return 0.0 if (u, v) in tree_edges else float(attrs.get("cost", 0.0))

        try:
            dist, paths = nx.multi_source_dijkstra(h, {src}, weight=marginal_weight)
        except Exception:
            dist, paths = {}, {}

        best_t = None
        best_key = (float("inf"), 1, float("inf"), "")
        for t in sorted(remaining):
            dval = dist.get(t)
            if dval is None:
                continue
            pref = 0 if t in preferred_first else 1
            j = _stable_hash01(f"{seed}:{t}")
            key = (float(dval), pref, j, t)
            if key < best_key:
                best_key = key
                best_t = t

        if best_t is None:
            # Disconnected: connect remaining directly for validity.
            for t in sorted(list(remaining)):
                try:
                    p = nx.dijkstra_path(h, src, t, weight="cost")
                except Exception:
                    continue
                for u, v in zip(p[:-1], p[1:]):
                    tree_edges.add((u, v))
                    tree_nodes.add(u)
                    tree_nodes.add(v)
                remaining.discard(t)
            break

        path = paths.get(best_t)
        if not path or len(path) < 2:
            remaining.discard(best_t)
            continue

        for u, v in zip(path[:-1], path[1:]):
            tree_edges.add((u, v))
            tree_nodes.add(u)
            tree_nodes.add(v)

        for t in list(remaining):
            if t in tree_nodes:
                remaining.discard(t)

    T = nx.DiGraph()
    T.add_nodes_from(tree_nodes)
    for u, v in tree_edges:
        if h.has_edge(u, v):
            T.add_edge(u, v, **h[u][v])
    return T


def _extract_dst_paths(h: nx.DiGraph, T: nx.DiGraph, src: str, dsts: List[str]) -> Dict[str, List[str]]:
    dst_paths: Dict[str, List[str]] = {}
    for d in dsts:
        try:
            dst_paths[d] = nx.dijkstra_path(T, src, d, weight="cost")
        except Exception:
            dst_paths[d] = nx.dijkstra_path(h, src, d, weight="cost")
    return dst_paths


def _greedy_hub_search_paths(h: nx.DiGraph, src: str, dsts: List[str]) -> Dict[str, List[str]]:
    """Agent1 best heuristic: hub-subset enumeration + greedy Steiner + multi-start."""
    hubs_by_provider = _choose_provider_hubs(h, dsts, k=1)
    dst_set = set(dsts)

    preferred_first: Set[str] = set()
    for prov in sorted(hubs_by_provider.keys()):
        if hubs_by_provider[prov]:
            preferred_first.add(hubs_by_provider[prov][0])

    hub_nodes: List[str] = []
    for prov in sorted(hubs_by_provider.keys()):
        for hub in hubs_by_provider[prov]:
            if hub != src and hub not in dst_set:
                hub_nodes.append(hub)

    best_dst_paths: Optional[Dict[str, List[str]]] = None
    best_cost = float("inf")
    best_time = float("inf")

    NUM_RESTARTS = 4

    for mask in range(1 << len(hub_nodes)):
        extra_hubs = {hub_nodes[i] for i in range(len(hub_nodes)) if (mask >> i) & 1}
        terminals = set(dsts) | extra_hubs

        best_mask_paths: Optional[Dict[str, List[str]]] = None
        best_mask_cost = float("inf")
        best_mask_time = float("inf")

        for seed in range(NUM_RESTARTS):
            T = _build_greedy_steiner_subgraph(
                h,
                src,
                terminals,
                preferred_first=preferred_first,
                seed=seed,
            )
            dst_paths = _extract_dst_paths(h, T, src, dsts)
            used_edges = _union_edges_from_dst_paths(dst_paths)

            cost = sum(_edge_cost(h, u, v) for (u, v) in used_edges)
            time_ratio = _estimate_time_ratio(h, used_edges)

            if (cost < best_mask_cost - 1e-12) or (
                abs(cost - best_mask_cost) <= 1e-12 and time_ratio < best_mask_time - 1e-12
            ):
                best_mask_cost = cost
                best_mask_time = time_ratio
                best_mask_paths = dst_paths

        if best_mask_paths is None:
            continue

        if (best_mask_cost < best_cost - 1e-12) or (
            abs(best_mask_cost - best_cost) <= 1e-12 and best_mask_time < best_time - 1e-12
        ):
            best_cost = best_mask_cost
            best_time = best_mask_time
            best_dst_paths = best_mask_paths

    if best_dst_paths is None:
        best_dst_paths = {d: nx.dijkstra_path(h, src, d, weight="cost") for d in dsts}

    return best_dst_paths


def _multi_source_dijkstra_on_reversed(
    preds: List[List[Tuple[int, float, float]]],
    init_dist: List[float],
    prefer_throughput: bool = True,
    seed: int = 0,
) -> Tuple[List[float], List[int]]:
    """Compute dist[u] = min_{u->...->x}(path_cost + init_dist[x]).

    Implemented as multi-source Dijkstra on the reversed graph.

    Tie-breaking (lexicographic):
    1) lower cost
    2) if prefer_throughput: higher bottleneck throughput
    3) if seed != 0: lower deterministic jitter (samples alternative equal-cost paths)

    Returns:
    - dist: best distances
    - next_hop: next_hop[u] = v means take edge u->v first; -1 means stop at u.
    """
    n = len(init_dist)
    INF = 1e30
    dist = init_dist[:]  # copy
    next_hop = [-1] * n

    EPS = 1e-12

    def _jitter(u: int, v: int) -> float:
        # Fast deterministic jitter (avoids hashlib/string ops in inner loops).
        x = (seed * 1103515245 + u * 2654435761 + v * 97531) & 0xFFFFFFFF
        return x / 2**32

    if not prefer_throughput:
        # Cost-only Dijkstra; add deterministic jitter tie-break when seed != 0.
        if seed == 0:
            heap: List[Tuple[float, int]] = [(dist[v], v) for v in range(n) if dist[v] < INF]
            heapq.heapify(heap)
            while heap:
                dv, v = heapq.heappop(heap)
                if dv > dist[v] + EPS:
                    continue
                for u, cost_uv, _th_uv in preds[v]:
                    nd = dv + cost_uv
                    if nd < dist[u] - EPS:
                        dist[u] = nd
                        next_hop[u] = v
                        heapq.heappush(heap, (nd, u))
            return dist, next_hop

        tie = [0.0 if dist[v] < INF else float("inf") for v in range(n)]
        heapj: List[Tuple[float, float, int]] = [(dist[v], tie[v], v) for v in range(n) if dist[v] < INF]
        heapq.heapify(heapj)
        while heapj:
            dv, tv, v = heapq.heappop(heapj)
            if dv > dist[v] + EPS:
                continue
            if tv > tie[v] + 1e-12:
                continue
            for u, cost_uv, _th_uv in preds[v]:
                nd = dv + cost_uv
                nt = tv + _jitter(u, v)
                if nd < dist[u] - EPS:
                    dist[u] = nd
                    tie[u] = nt
                    next_hop[u] = v
                    heapq.heappush(heapj, (nd, nt, u))
                elif abs(nd - dist[u]) <= EPS:
                    if nt < tie[u] - 1e-12:
                        tie[u] = nt
                        next_hop[u] = v
                        heapq.heappush(heapj, (nd, nt, u))
                    elif abs(nt - tie[u]) <= 1e-12 and next_hop[u] != -1 and v < next_hop[u]:
                        next_hop[u] = v
        return dist, next_hop

    # Throughput-aware tie-break: maximize min-throughput along the path.
    if seed == 0:
        bneck = [float("inf") if dist[v] < INF else -1.0 for v in range(n)]
        heap2: List[Tuple[float, float, int]] = [(dist[v], -bneck[v], v) for v in range(n) if dist[v] < INF]
        heapq.heapify(heap2)

        while heap2:
            dv, nb, v = heapq.heappop(heap2)
            if dv > dist[v] + EPS:
                continue

            entry_bn = -nb
            cur_bn = bneck[v]
            if cur_bn == float("inf"):
                if entry_bn != float("inf"):
                    continue
            else:
                if entry_bn == float("inf"):
                    continue
                if abs(entry_bn - cur_bn) > 1e-9:
                    continue

            for u, cost_uv, th_uv in preds[v]:
                nd = dv + cost_uv
                new_bn = th_uv if cur_bn == float("inf") else min(cur_bn, th_uv)

                if nd < dist[u] - EPS:
                    dist[u] = nd
                    bneck[u] = new_bn
                    next_hop[u] = v
                    heapq.heappush(heap2, (nd, -new_bn, u))
                elif abs(nd - dist[u]) <= EPS:
                    if new_bn > bneck[u] + 1e-9:
                        bneck[u] = new_bn
                        next_hop[u] = v
                        heapq.heappush(heap2, (nd, -new_bn, u))
                    elif abs(new_bn - bneck[u]) <= 1e-9 and next_hop[u] != -1 and v < next_hop[u]:
                        next_hop[u] = v

        return dist, next_hop

    # seed != 0: add deterministic jitter as tertiary tie-break.
    bneck = [float("inf") if dist[v] < INF else -1.0 for v in range(n)]
    tie = [0.0 if dist[v] < INF else float("inf") for v in range(n)]
    heap3: List[Tuple[float, float, float, int]] = [(dist[v], -bneck[v], tie[v], v) for v in range(n) if dist[v] < INF]
    heapq.heapify(heap3)

    while heap3:
        dv, nb, tv, v = heapq.heappop(heap3)
        if dv > dist[v] + EPS:
            continue

        entry_bn = -nb
        cur_bn = bneck[v]
        if cur_bn == float("inf"):
            if entry_bn != float("inf"):
                continue
        else:
            if entry_bn == float("inf"):
                continue
            if abs(entry_bn - cur_bn) > 1e-9:
                continue

        if tv > tie[v] + 1e-12:
            continue

        for u, cost_uv, th_uv in preds[v]:
            nd = dv + cost_uv
            new_bn = th_uv if cur_bn == float("inf") else min(cur_bn, th_uv)
            nt = tv + _jitter(u, v)

            if nd < dist[u] - EPS:
                dist[u] = nd
                bneck[u] = new_bn
                tie[u] = nt
                next_hop[u] = v
                heapq.heappush(heap3, (nd, -new_bn, nt, u))
            elif abs(nd - dist[u]) <= EPS:
                if new_bn > bneck[u] + 1e-9:
                    bneck[u] = new_bn
                    tie[u] = nt
                    next_hop[u] = v
                    heapq.heappush(heap3, (nd, -new_bn, nt, u))
                elif abs(new_bn - bneck[u]) <= 1e-9:
                    if nt < tie[u] - 1e-12:
                        tie[u] = nt
                        next_hop[u] = v
                        heapq.heappush(heap3, (nd, -new_bn, nt, u))
                    elif abs(nt - tie[u]) <= 1e-12 and next_hop[u] != -1 and v < next_hop[u]:
                        next_hop[u] = v

    return dist, next_hop


def _steiner_dp_paths(
    h: nx.DiGraph,
    src: str,
    dsts: List[str],
    seed: int = 0,
    beta_throughput: float = 0.0,
    edge_penalty: Optional[Dict[Tuple[str, str], float]] = None,
) -> Optional[Dict[str, List[str]]]:
    """Exact DP for a minimum-cost directed Steiner arborescence (small terminal sets).

    Returns dst->node-path dict, or None if DP is disabled/unavailable.
    """
    terminals = [d for d in dsts if d != src]
    k = len(terminals)
    if k <= 1:
        return {d: nx.dijkstra_path(h, src, d, weight="cost") for d in dsts}

    if k > DP_MAX_TERMINALS:
        return None

    # Tie-break policy: for single-provider problems, use throughput-aware tie-breaks
    # on equal-cost relaxations to reduce makespan; for inter-cloud, keep pure cost.
    provs = {_provider(src)} | {_provider(d) for d in dsts}
    # If we already bake throughput into the edge weights (beta_throughput>0),
    # avoid double-counting it in tie-breaking.
    prefer_throughput_move = (beta_throughput <= 1e-15 and len(provs) == 1 and "gcp" in provs)

    nodes = sorted(list(h.nodes))
    if len(nodes) > DP_MAX_NODES:
        return None

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if src not in node_to_idx:
        return None

    term_idx = [node_to_idx[t] for t in terminals if t in node_to_idx]
    if len(term_idx) != k:
        return None

    n = len(nodes)

    preds: List[List[Tuple[int, float, float]]] = [[] for _ in range(n)]
    for u, v, data in h.edges(data=True):
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        c = float(data.get("cost", 0.0))
        th = float(data.get("throughput", 0.0))

        # Optional throughput-weighted cost: encourages high-throughput edges when
        # they have similar $/GB cost.
        if beta_throughput > 1e-15:
            denom = th if th > 1e-9 else 1e9
            c = c + (beta_throughput / denom)

        # Optional edge penalties (used only to generate diverse candidates).
        if edge_penalty:
            c += float(edge_penalty.get((u, v), 0.0))

        preds[vi].append((ui, c, th))

    INF = 1e30
    dist_to_term: List[List[float]] = []
    next_to_term: List[List[int]] = []
    for ti in term_idx:
        init = [INF] * n
        init[ti] = 0.0
        dist, nxt = _multi_source_dijkstra_on_reversed(
            preds,
            init,
            prefer_throughput=False,
            seed=seed,
        )
        dist_to_term.append(dist)
        next_to_term.append(nxt)

    M = 1 << k
    dp: List[List[float]] = [[INF] * n for _ in range(M)]
    base_sub: List[List[int]] = [[-1] * n for _ in range(M)]
    next_hop: List[List[int]] = [[-1] * n for _ in range(M)]

    for i in range(k):
        mask = 1 << i
        di = dist_to_term[i]
        for v in range(n):
            dp[mask][v] = di[v]
            base_sub[mask][v] = -(i + 1)

    masks_by_pop = [[] for _ in range(k + 1)]
    for mask in range(1, M):
        masks_by_pop[mask.bit_count()].append(mask)

    for sz in range(2, k + 1):
        for mask in masks_by_pop[sz]:
            base_cost = [INF] * n
            base_choice = [-1] * n

            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if sub < other:
                    dps = dp[sub]
                    dpo = dp[other]
                    for v in range(n):
                        cand = dps[v] + dpo[v]
                        if cand < base_cost[v] - 1e-12:
                            base_cost[v] = cand
                            base_choice[v] = sub
                        elif abs(cand - base_cost[v]) <= 1e-12 and sub != -1:
                            if base_choice[v] == -1:
                                base_choice[v] = sub
                            elif seed == 0:
                                if sub < base_choice[v]:
                                    base_choice[v] = sub
                            else:
                                # Jitter tie-break among equal-cost merges to sample alternative optima.
                                j_new = _stable_hash01(f"{seed}:{mask}:{v}:{sub}")
                                j_old = _stable_hash01(f"{seed}:{mask}:{v}:{base_choice[v]}")
                                if j_new < j_old:
                                    base_choice[v] = sub
                sub = (sub - 1) & mask

            for v in range(n):
                base_sub[mask][v] = base_choice[v]

            dist, nxt = _multi_source_dijkstra_on_reversed(
                preds,
                base_cost,
                prefer_throughput=prefer_throughput_move,
                seed=seed,
            )
            dp[mask] = dist
            next_hop[mask] = nxt

    src_i = node_to_idx[src]
    full = M - 1
    if dp[full][src_i] >= INF / 2:
        return None

    used_edges_idx: Set[Tuple[int, int]] = set()
    visited_states: Set[Tuple[int, int]] = set()

    def collect(state_v: int, state_mask: int):
        if state_mask == 0:
            return
        key = (state_v, state_mask)
        if key in visited_states:
            return
        visited_states.add(key)

        nh = next_hop[state_mask][state_v]
        if nh != -1:
            used_edges_idx.add((state_v, nh))
            collect(nh, state_mask)
            return

        subc = base_sub[state_mask][state_v]
        if subc < 0:
            term_i = (-subc) - 1
            target = term_idx[term_i]
            cur = state_v
            while cur != target:
                nxt = next_to_term[term_i][cur]
                if nxt == -1:
                    break
                used_edges_idx.add((cur, nxt))
                cur = nxt
            return

        if subc == -1:
            return
        collect(state_v, subc)
        collect(state_v, state_mask ^ subc)

    collect(src_i, full)

    T = nx.DiGraph()
    T.add_node(src)
    for d in dsts:
        T.add_node(d)

    for ui, vi in used_edges_idx:
        u = nodes[ui]
        v = nodes[vi]
        if h.has_edge(u, v):
            T.add_edge(u, v, **h[u][v])

    dst_paths: Dict[str, List[str]] = {}
    for d in dsts:
        try:
            dst_paths[d] = nx.dijkstra_path(T, src, d, weight="cost")
        except Exception:
            dst_paths[d] = nx.dijkstra_path(h, src, d, weight="cost")

    return dst_paths


def _rebalance_tree_paths_for_time(
    h: nx.DiGraph,
    src: str,
    dsts: List[str],
    dst_paths: Dict[str, List[str]],
    max_iters: int = 60,
    perturb_rounds: int = 2,
) -> Dict[str, List[str]]:
    used_edges = _union_edges_from_dst_paths(dst_paths)
    if not used_edges:
        return dst_paths

    parent: Dict[str, str] = {}
    children: Dict[str, List[str]] = {}
    for u, v in used_edges:
        if v in parent and parent[v] != u:
            return dst_paths
        parent[v] = u
        children.setdefault(u, []).append(v)

    for d in dsts:
        if d == src:
            continue
        if d not in parent:
            return dst_paths

    # Safety: the local-search rebalancer is O(|V|^2) per iteration; skip on
    # very large trees to stay within time limits on hidden tests.
    tree_nodes = set(parent.keys()) | {src}
    if len(tree_nodes) > REBALANCE_MAX_TREE_NODES:
        return dst_paths

    def ec(u: str, v: str) -> float:
        if h.has_edge(u, v):
            return _edge_cost(h, u, v)
        return float("inf")

    provs = {_provider(src)} | {_provider(d) for d in dsts}
    if len(provs) > 1:
        time_cost_per_sec = 0.003
    else:
        time_cost_per_sec = {"aws": 0.0021, "azure": 0.0021, "gcp": 0.0024}.get(_provider(src), 0.0021)

    DATA_GB = 300.0
    DATA_GBIT = DATA_GB * 8.0

    def predict_time_seconds(edges: Set[Tuple[str, str]]) -> float:
        outdeg: Dict[str, int] = {}
        for uu, vv in edges:
            outdeg[uu] = outdeg.get(uu, 0) + 1

        worst = 0.0
        for uu, vv in edges:
            units = PROVIDER_EGRESS_UNITS.get(_provider(uu), 7.0)
            share_bw = (2.0 * units) / outdeg[uu] if outdeg[uu] > 0 else 0.0
            th = _edge_throughput_gbps(h, uu, vv)
            if th <= 0:
                eff_bw = share_bw
            elif share_bw <= 0:
                eff_bw = th
            else:
                eff_bw = min(share_bw, th)

            if eff_bw <= 1e-12:
                return float("inf")
            t = DATA_GBIT / eff_bw
            if t > worst:
                worst = t
        return worst

    def objective_total_dollars(edges: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
        edge_cost_sum = sum(ec(uu, vv) for (uu, vv) in edges)
        tsec = predict_time_seconds(edges)
        return edge_cost_sum * DATA_GB + time_cost_per_sec * tsec, edge_cost_sum, tsec

    obj0, c0, t0 = objective_total_dollars(set(used_edges))
    if t0 == float("inf"):
        return dst_paths

    # Max per-edge cost increase (in $/GB) that could possibly be justified by
    # time savings. This is a *loose* upper bound used only to prune hopeless
    # reattachments; objective checks still decide what to accept.
    max_cost_slack = 0.0
    if REBALANCE_ALLOW_COST_INCREASE:
        max_provider_bw = max(2.0 * PROVIDER_EGRESS_UNITS.get(p, 7.0) for p in provs)
        max_edge_th = 0.0
        for uu, vv in h.edges:
            th = _edge_throughput_gbps(h, uu, vv)
            if th > max_edge_th:
                max_edge_th = th
        if max_edge_th <= 1e-12:
            max_edge_th = max_provider_bw
        max_bw = min(max_provider_bw, max_edge_th)
        t_lower = DATA_GBIT / max_bw if max_bw > 1e-12 else 0.0
        max_cost_slack = (time_cost_per_sec * max(0.0, t0 - t_lower)) / DATA_GB
        max_cost_slack *= REBALANCE_SLACK_FACTOR

    if DEBUG_REBALANCE and len(provs) > 1:
        print("=== DEBUG_REBALANCE (inter-cloud) ===")
        print(f"src={src}")
        print(f"dsts={dsts}")
        print(f"time_cost_per_sec={time_cost_per_sec}")
        print(f"max_cost_slack($/GB)={max_cost_slack:.6f}")
        print(f"initial: edge_cost_sum={c0:.6f} pred_time={t0:.2f}s obj={obj0:.3f}")

    def _is_ancestor(anc: str, node: str, p_map: Dict[str, str]) -> bool:
        cur = node
        while cur != src:
            if cur == anc:
                return True
            cur = p_map.get(cur)
            if cur is None:
                break
        return False

    def gen_moves(edges: Set[Tuple[str, str]], p_map: Dict[str, str], ch_map: Dict[str, List[str]]):
        nodes = sorted(set(p_map.keys()) | {src})
        for uu, ch in ch_map.items():
            if not ch:
                continue
            for vv in sorted(ch):
                if (uu, vv) not in edges:
                    continue
                c_uv = ec(uu, vv)
                for ww in nodes:
                    if ww == uu or ww == vv:
                        continue
                    if _is_ancestor(vv, ww, p_map):
                        continue
                    if not h.has_edge(ww, vv):
                        continue
                    c_wv = ec(ww, vv)
                    if c_wv > c_uv + max_cost_slack + 1e-12:
                        continue
                    yield (uu, ww, vv)

    def apply_move(
        edges: Set[Tuple[str, str]],
        p_map: Dict[str, str],
        ch_map: Dict[str, List[str]],
        uu: str,
        ww: str,
        vv: str,
    ):
        ne = set(edges)
        np = dict(p_map)
        nc = {x: list(lst) for x, lst in ch_map.items()}

        ne.remove((uu, vv))
        ne.add((ww, vv))

        np[vv] = ww
        nc[uu].remove(vv)
        nc.setdefault(ww, []).append(vv)
        return ne, np, nc

    def greedy_descent(edges: Set[Tuple[str, str]], p_map: Dict[str, str], ch_map: Dict[str, List[str]]):
        cur_obj = objective_total_dollars(edges)[0]
        for _ in range(max_iters):
            best = None
            best_obj = cur_obj
            for uu, ww, vv in gen_moves(edges, p_map, ch_map):
                ne, np, nc = apply_move(edges, p_map, ch_map, uu, ww, vv)
                obj = objective_total_dollars(ne)[0]
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best = (ne, np, nc)
            if best is None:
                break
            edges, p_map, ch_map = best
            cur_obj = best_obj
        return edges, p_map, ch_map

    edges0 = set(used_edges)
    edges0, parent0, children0 = greedy_descent(edges0, parent, children)

    best_edges, best_parent, best_children = edges0, parent0, children0
    best_obj, best_c, best_t = objective_total_dollars(best_edges)

    for _ in range(perturb_rounds):
        base_edges, base_parent, base_children = best_edges, best_parent, best_children

        cand_edges, cand_parent, cand_children = best_edges, best_parent, best_children
        cand_obj = best_obj

        for uu, ww, vv in list(gen_moves(base_edges, base_parent, base_children)):
            ne, np, nc = apply_move(base_edges, base_parent, base_children, uu, ww, vv)
            ne, np, nc = greedy_descent(ne, np, nc)
            obj = objective_total_dollars(ne)[0]
            if obj < cand_obj - 1e-9:
                cand_obj = obj
                cand_edges, cand_parent, cand_children = ne, np, nc

        if cand_obj < best_obj - 1e-9:
            best_obj = cand_obj
            best_edges, best_parent, best_children = cand_edges, cand_parent, cand_children
        else:
            break

    new_paths: Dict[str, List[str]] = {}
    for d in dsts:
        cur = d
        path_rev = [cur]
        seen = {cur}
        while cur != src:
            cur = best_parent.get(cur)
            if cur is None or cur in seen:
                return dst_paths
            seen.add(cur)
            path_rev.append(cur)
        path_rev.reverse()
        new_paths[d] = path_rev

    return new_paths


def _predict_time_seconds_from_used_edges(h: nx.DiGraph, used_edges: Set[Tuple[str, str]]) -> float:
    """Predict makespan assuming each used edge carries the full data volume.

    This mirrors the simulator's behavior well enough for tie-breaking.
    """
    if not used_edges:
        return 0.0

    DATA_GBIT = 300.0 * 8.0

    outdeg: Dict[str, int] = {}
    for u, v in used_edges:
        outdeg[u] = outdeg.get(u, 0) + 1

    worst = 0.0
    for u, v in used_edges:
        units = PROVIDER_EGRESS_UNITS.get(_provider(u), 7.0)
        share_bw = (2.0 * units) / outdeg[u] if outdeg[u] > 0 else 0.0
        th = _edge_throughput_gbps(h, u, v)
        eff_bw = share_bw if th <= 0 else min(share_bw, th)
        if eff_bw <= 1e-12:
            return float("inf")
        worst = max(worst, DATA_GBIT / eff_bw)

    return worst


def _bottleneck_edges_from_used_edges(
    h: nx.DiGraph, used_edges: Set[Tuple[str, str]]
) -> Set[Tuple[str, str]]:
    """Return edges that achieve the predicted makespan (full-volume model)."""
    if not used_edges:
        return set()

    DATA_GBIT = 300.0 * 8.0

    outdeg: Dict[str, int] = {}
    for u, v in used_edges:
        outdeg[u] = outdeg.get(u, 0) + 1

    worst = 0.0
    edge_time: Dict[Tuple[str, str], float] = {}
    for u, v in used_edges:
        units = PROVIDER_EGRESS_UNITS.get(_provider(u), 7.0)
        share_bw = (2.0 * units) / outdeg[u] if outdeg[u] > 0 else 0.0
        th = _edge_throughput_gbps(h, u, v)
        eff_bw = share_bw if th <= 0 else min(share_bw, th)
        if eff_bw <= 1e-12:
            continue
        t = DATA_GBIT / eff_bw
        edge_time[(u, v)] = t
        if t > worst:
            worst = t

    if worst <= 0:
        return set()

    eps = 1e-9
    return {e for e, t in edge_time.items() if t >= worst - eps}


def _evaluate_dst_paths(h: nx.DiGraph, src: str, dsts: List[str], dst_paths: Dict[str, List[str]]) -> Tuple[float, float]:
    """Return (edge_cost_sum, predicted_makespan_seconds)."""
    used_edges = _union_edges_from_dst_paths(dst_paths)
    cost = sum(_edge_cost(h, u, v) for (u, v) in used_edges)
    tsec = _predict_time_seconds_from_used_edges(h, used_edges)
    return cost, tsec


def _time_cost_per_sec(src: str, dsts: List[str]) -> float:
    """Empirical $/sec coefficient of the simulator's time penalty.

    Used only for candidate selection / partition striping decisions.
    """
    provs = {_provider(src)} | {_provider(d) for d in dsts}
    if len(provs) > 1:
        return 0.003
    return {"aws": 0.0021, "azure": 0.0021, "gcp": 0.0024}.get(_provider(src), 0.0021)


def _objective_dollars(cost_sum: float, tsec: float, time_cost_per_sec: float, data_gb: float = 300.0) -> float:
    """Estimate simulator tot_cost for a single broadcast tree."""
    return data_gb * cost_sum + time_cost_per_sec * tsec


def _predict_time_seconds_from_edge_fracs(
    h: nx.DiGraph, edge_fracs: Dict[Tuple[str, str], float]
) -> float:
    """Predict makespan when each edge carries a fraction of the total volume."""
    if not edge_fracs:
        return 0.0

    DATA_GBIT = 300.0 * 8.0

    outdeg: Dict[str, int] = {}
    for (u, v), frac in edge_fracs.items():
        if frac <= 1e-12:
            continue
        outdeg[u] = outdeg.get(u, 0) + 1

    worst = 0.0
    for (u, v), frac in edge_fracs.items():
        if frac <= 1e-12:
            continue
        units = PROVIDER_EGRESS_UNITS.get(_provider(u), 7.0)
        share_bw = (2.0 * units) / outdeg[u] if outdeg[u] > 0 else 0.0
        th = _edge_throughput_gbps(h, u, v)
        eff_bw = share_bw if th <= 0 else min(share_bw, th)
        if eff_bw <= 1e-12:
            return float("inf")
        worst = max(worst, (DATA_GBIT * frac) / eff_bw)

    return worst


def _steiner_dp_candidate_paths(
    h: nx.DiGraph,
    src: str,
    dsts: List[str],
    max_seeds: int = 3,
    beta_throughput: float = 0.0,
    edge_penalty: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[Dict[str, List[str]]]:
    """Generate multiple DP solutions by varying tie-breaks (seeded)."""
    terminals = [d for d in dsts if d != src]
    k = len(terminals)
    if k <= 1:
        return [{d: nx.dijkstra_path(h, src, d, weight="cost") for d in dsts}]

    n_nodes = len(h.nodes)
    if k > DP_MAX_TERMINALS or n_nodes > DP_MAX_NODES:
        return []

    if k <= 5 and n_nodes <= 120:
        restarts = min(4, max_seeds)
    elif k <= 7 and n_nodes <= 180:
        restarts = min(3, max_seeds)
    else:
        restarts = min(2, max_seeds)

    cands: List[Dict[str, List[str]]] = []
    seen: Set[frozenset] = set()

    for seed in range(restarts):
        paths = _steiner_dp_paths(
            h,
            src,
            dsts,
            seed=seed,
            beta_throughput=beta_throughput,
            edge_penalty=edge_penalty,
        )
        if paths is None:
            break
        paths = _rebalance_tree_paths_for_time(h, src, dsts, paths)
        used = frozenset(_union_edges_from_dst_paths(paths))
        if used not in seen:
            seen.add(used)
            cands.append(paths)

    return cands


# NOTE: The remainder of the file (AWS relay swap, compression, striping helpers,
# search_algorithm, and BroadCastTopology definitions) is included as provided.
# This file is intended to be a faithful conversion of your pasted "code" string.

def _aws_relay_swap_candidates(
    h: nx.DiGraph,
    src: str,
    dsts: List[str],
    base_paths: Dict[str, List[str]],
    max_relays: int = 4,
    max_gateways: int = 2,
) -> List[Dict[str, List[str]]]:
    """Generate alternative inter-cloud trees by swapping the AWS ingress relay."""
    aws_dsts = [d for d in dsts if _provider(d) == "aws" and d != src]
    if len(aws_dsts) < 2:
        return []

    used_edges = _union_edges_from_dst_paths(base_paths)
    entries = [(u, v) for (u, v) in used_edges if _provider(v) == "aws" and _provider(u) != "aws"]
    if len(entries) != 1:
        return []

    entry_parent, entry_node = entries[0]

    ref_path = base_paths.get(entry_node)
    if not ref_path or entry_parent not in ref_path:
        for d in aws_dsts:
            p = base_paths.get(d)
            if p and entry_parent in p:
                ref_path = p
                break

    if not ref_path or entry_parent not in ref_path:
        return []

    prefix_to_parent = ref_path[: ref_path.index(entry_parent) + 1]

    aws_nodes = [n for n in h.nodes if _provider(n) == "aws"]
    if not aws_nodes:
        return []

    h_aws = h.subgraph(aws_nodes).copy()

    attach_points: List[Tuple[str, List[str], int]] = []
    attach_points.append((entry_parent, prefix_to_parent, max_relays))

    gateway_nodes: List[str] = []
    for (u, v) in used_edges:
        if u != entry_parent:
            continue
        if _provider(v) == "aws":
            continue
        if v not in base_paths:
            continue
        if any(_provider(x) == "aws" for x in h.successors(v)):
            gateway_nodes.append(v)

    gateway_nodes = sorted(
        set(gateway_nodes),
        key=lambda n: (-PROVIDER_EGRESS_UNITS.get(_provider(n), 0.0), n),
    )[:max_gateways]

    for gw in gateway_nodes:
        attach_points.append((gw, list(base_paths[gw]), min(2, max_relays)))

    dist_cache: Dict[str, Dict[str, float]] = {}

    def _aws_dist(r: str) -> Optional[Dict[str, float]]:
        if r in dist_cache:
            return dist_cache[r]
        try:
            dist_cache[r] = nx.single_source_dijkstra_path_length(h_aws, r, weight="cost")
        except Exception:
            return None
        return dist_cache[r]

    out: List[Dict[str, List[str]]] = []

    for attach, prefix_to_attach, lim in attach_points:
        relay_cands = [v for v in h.successors(attach) if _provider(v) == "aws" and v != entry_node]
        if not relay_cands:
            continue

        ranked: List[Tuple[float, str]] = []
        for r in relay_cands:
            dist = _aws_dist(r)
            if dist is None:
                continue

            ok = True
            sumd = 0.0
            for d in aws_dsts:
                if d == r:
                    continue
                dv = dist.get(d)
                if dv is None:
                    ok = False
                    break
                sumd += float(dv)

            if not ok:
                continue

            ranked.append((_edge_cost(h, attach, r) + sumd, r))

        ranked.sort(key=lambda x: (x[0], x[1]))
        ranked = ranked[:lim]

        for _score, r in ranked:
            h_aws_r = h_aws.copy()
            h_aws_r.remove_edges_from(list(h_aws_r.in_edges(r)) + list(nx.selfloop_edges(h_aws_r)))

            aws_paths = _steiner_dp_paths(h_aws_r, r, aws_dsts, seed=0)
            if aws_paths is None:
                aws_paths = {}
                ok2 = True
                for d in aws_dsts:
                    try:
                        aws_paths[d] = nx.dijkstra_path(h_aws_r, r, d, weight="cost")
                    except Exception:
                        ok2 = False
                        break
                if not ok2:
                    continue

            cand: Dict[str, List[str]] = {}
            ok3 = True
            for d in dsts:
                if _provider(d) != "aws":
                    base_p = base_paths.get(d)
                    if not base_p:
                        ok3 = False
                        break
                    cand[d] = list(base_p)
                else:
                    sub = aws_paths.get(d)
                    if not sub:
                        ok3 = False
                        break
                    cand[d] = prefix_to_attach + [r] + sub[1:]

            if not ok3:
                continue

            out.append(cand)

    return out


def _compress_hub_leaf_fanout(
    h: nx.DiGraph,
    base_paths: Dict[str, List[str]],
    hub: str,
    eps: float = 1e-12,
) -> Optional[Dict[str, List[str]]]:
    """Create a cost-nonincreasing variant of `base_paths` with smaller hub out-degree."""
    used_edges = _union_edges_from_dst_paths(base_paths)

    children: Dict[str, List[str]] = {}
    for u, v in used_edges:
        children.setdefault(u, []).append(v)

    hub_children = children.get(hub, [])
    if not hub_children:
        return None

    leaf_terms = [c for c in hub_children if c not in children and c in base_paths]
    if len(leaf_terms) < 2:
        return None

    hub_prov = _provider(hub)
    leaf_terms = sorted(leaf_terms)

    for a in leaf_terms:
        if _provider(a) != hub_prov:
            continue
        for b in leaf_terms:
            if a == b:
                continue
            if _provider(b) != hub_prov:
                continue
            if not h.has_edge(a, b):
                continue
            try:
                if _edge_cost(h, a, b) <= _edge_cost(h, hub, b) + eps:
                    new_paths = {d: list(p) for d, p in base_paths.items()}
                    new_paths[b] = list(base_paths[a]) + [b]
                    return new_paths
            except Exception:
                continue

    return None


def _round_robin_partition_assignment(num_partitions: int, counts: List[int]) -> List[int]:
    """Return list mapping partition->bucket index (0..len(counts)-1)."""
    rem = list(counts)
    k = len(rem)
    out: List[int] = []
    i = 0
    while len(out) < num_partitions:
        if rem[i] > 0:
            out.append(i)
            rem[i] -= 1
        i = (i + 1) % k
    return out


def _find_aws_two_leaf_siblings_for_striping(
    h: nx.DiGraph, dst_paths: Dict[str, List[str]]
) -> Optional[Tuple[str, str, str]]:
    """Detect an AWS parent with exactly two AWS *leaf* children in the tree."""
    used_edges = _union_edges_from_dst_paths(dst_paths)
    children: Dict[str, List[str]] = {}
    outdeg: Dict[str, int] = {}

    for u, v in used_edges:
        children.setdefault(u, []).append(v)

    for u, vs in children.items():
        outdeg[u] = len(vs)

    eps = 1e-12

    for u, vs in children.items():
        if _provider(u) != "aws":
            continue

        aws_leaf_children = [
            c for c in vs if _provider(c) == "aws" and outdeg.get(c, 0) == 0
        ]
        if len(aws_leaf_children) != 2:
            continue

        a, b = aws_leaf_children

        if not h.has_edge(a, b) or not h.has_edge(b, a):
            continue

        c_ua = _edge_cost(h, u, a)
        c_ub = _edge_cost(h, u, b)
        c_ab = _edge_cost(h, a, b)
        c_ba = _edge_cost(h, b, a)

        if c_ab <= c_ub + eps and c_ba <= c_ua + eps:
            return (u, a, b)

    return None


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    if DEBUG_MAIN:
        print(
            f"[search] src={src} k={len(dsts)-1} nodes={len(h.nodes)} partitions={num_partitions}"
        )

    DATA_GB = 300.0
    time_cost_per_sec = _time_cost_per_sec(src, dsts)
    provs = {_provider(src)} | {_provider(d) for d in dsts}

    candidates: List[Dict[str, List[str]]] = _steiner_dp_candidate_paths(
        h, src, dsts, max_seeds=4, beta_throughput=0.0
    )

    beta0 = time_cost_per_sec * 8.0
    for f in DP_EXTRA_BETA_FACTORS:
        beta = beta0 * float(f)
        candidates.extend(_steiner_dp_candidate_paths(h, src, dsts, max_seeds=1, beta_throughput=beta))

    dp_count = len(candidates)
    if DEBUG_MAIN:
        print(f"[search] dp_candidates={dp_count}")

    if ENABLE_MIX2_STRIPING and len(provs) > 1 and dp_count > 0:
        dp_objs: List[float] = []
        for i in range(dp_count):
            ci, ti = _evaluate_dst_paths(h, src, dsts, candidates[i])
            dp_objs.append(_objective_dollars(ci, ti, time_cost_per_sec, data_gb=DATA_GB))
        base_idx = min(range(dp_count), key=lambda i: dp_objs[i])

        base_edges0 = set(_union_edges_from_dst_paths(candidates[base_idx]))
        bneck0 = _bottleneck_edges_from_used_edges(h, base_edges0)
        if bneck0:
            for pen in (0.001, 0.003):
                ep = {e: pen for e in bneck0}
                alt = _steiner_dp_paths(
                    h,
                    src,
                    dsts,
                    seed=0,
                    beta_throughput=0.0,
                    edge_penalty=ep,
                )
                if alt is not None:
                    candidates.append(alt)

    if ENABLE_MIX2_STRIPING and len(provs) > 1 and dp_count > 0 and int(num_partitions) >= 3:
        dp_objs2: List[float] = []
        for i in range(dp_count):
            ci, ti = _evaluate_dst_paths(h, src, dsts, candidates[i])
            dp_objs2.append(_objective_dollars(ci, ti, time_cost_per_sec, data_gb=DATA_GB))
        base_idx2 = min(range(dp_count), key=lambda i: dp_objs2[i])
        base_used2 = set(_union_edges_from_dst_paths(candidates[base_idx2]))

        outdeg2: Dict[str, int] = {}
        for u, v in base_used2:
            outdeg2[u] = outdeg2.get(u, 0) + 1

        hub_cands = [
            u
            for u, d in outdeg2.items()
            if d >= 3 and u != src and (u not in dsts) and h.has_node(u)
        ]
        hub_cands = sorted(
            hub_cands,
            key=lambda n: (-outdeg2.get(n, 0), -PROVIDER_EGRESS_UNITS.get(_provider(n), 0.0), n),
        )[:2]

        for hub in hub_cands:
            try:
                h2 = h.copy()
                h2.remove_node(hub)
                alt2 = _steiner_dp_paths(h2, src, dsts, seed=0, beta_throughput=0.0)
                if alt2 is not None:
                    alt2 = _rebalance_tree_paths_for_time(h2, src, dsts, alt2)
                    candidates.append(alt2)
            except Exception:
                pass

        if len(hub_cands) >= 2:
            try:
                h3 = h.copy()
                h3.remove_nodes_from(hub_cands[:2])
                alt3 = _steiner_dp_paths(h3, src, dsts, seed=0, beta_throughput=0.0)
                if alt3 is not None:
                    alt3 = _rebalance_tree_paths_for_time(h3, src, dsts, alt3)
                    candidates.append(alt3)
            except Exception:
                pass

        src_children = sorted(
            {v for (u, v) in base_used2 if u == src and v != src and (v not in dsts)}
        )
        for ch in src_children[:2]:
            try:
                h4 = h.copy()
                if h4.has_edge(src, ch):
                    h4.remove_edge(src, ch)
                alt4 = _steiner_dp_paths(h4, src, dsts, seed=0, beta_throughput=0.0)
                if alt4 is not None:
                    alt4 = _rebalance_tree_paths_for_time(h4, src, dsts, alt4)
                    candidates.append(alt4)
            except Exception:
                pass

    greedy_paths = _greedy_hub_search_paths(h, src, dsts)
    greedy_paths = _rebalance_tree_paths_for_time(h, src, dsts, greedy_paths)
    candidates.append(greedy_paths)

    compressed_base = None
    if (ENABLE_MIX2_STRIPING or ENABLE_RELAY_SWAP_CANDS) and len(provs) > 1 and dp_count > 0:
        base0 = candidates[0]
        used0 = _union_edges_from_dst_paths(base0)
        entries0 = [(u, v) for (u, v) in used0 if _provider(v) == "aws" and _provider(u) != "aws"]
        if len(entries0) == 1:
            hub0, _ = entries0[0]
            compressed_base = _compress_hub_leaf_fanout(h, base0, hub0)
            if compressed_base is not None:
                candidates.append(compressed_base)

    if ENABLE_RELAY_SWAP_CANDS and len(provs) > 1:
        swap_base = compressed_base if compressed_base is not None else (candidates[0] if dp_count > 0 else greedy_paths)
        candidates.extend(_aws_relay_swap_candidates(h, src, dsts, swap_base, max_relays=4))

    uniq: Dict[frozenset, Dict[str, List[str]]] = {}
    for p in candidates:
        ekey = frozenset(_union_edges_from_dst_paths(p))
        if ekey not in uniq:
            uniq[ekey] = p
        else:
            c_new, t_new = _evaluate_dst_paths(h, src, dsts, p)
            c_old, t_old = _evaluate_dst_paths(h, src, dsts, uniq[ekey])
            if _objective_dollars(c_new, t_new, time_cost_per_sec, data_gb=DATA_GB) < _objective_dollars(
                c_old, t_old, time_cost_per_sec, data_gb=DATA_GB
            ) - 1e-9:
                uniq[ekey] = p
    candidates = list(uniq.values())

    infos = []
    for p in candidates:
        used = set(_union_edges_from_dst_paths(p))
        cost_sum, t_full = _evaluate_dst_paths(h, src, dsts, p)
        obj_full = _objective_dollars(cost_sum, t_full, time_cost_per_sec, data_gb=DATA_GB)
        infos.append((p, used, cost_sum, t_full, obj_full))

    best_idx = min(range(len(infos)), key=lambda i: (infos[i][4], infos[i][2], infos[i][3]))
    best_obj = infos[best_idx][4]
    best_cost = infos[best_idx][2]
    best_time = infos[best_idx][3]
    best_plan = ("single", best_idx, num_partitions)

    P = int(num_partitions)
    if ENABLE_MIX2_STRIPING and P >= 2 and len(infos) >= 2:
        max_mix_cands = min(9, len(infos))

        top_ids = sorted(range(len(infos)), key=lambda i: infos[i][4])[: min(5, len(infos))]
        if best_idx not in top_ids:
            top_ids.insert(0, best_idx)

        bneck_best = _bottleneck_edges_from_used_edges(h, infos[best_idx][1])
        if bneck_best:
            ranked = sorted(
                range(len(infos)),
                key=lambda i: (len(bneck_best & infos[i][1]), infos[i][4]),
            )
            for i in ranked:
                if i not in top_ids:
                    top_ids.append(i)
                    if len(top_ids) >= max_mix_cands:
                        break
        else:
            for i in sorted(range(len(infos)), key=lambda i: infos[i][4]):
                if i not in top_ids:
                    top_ids.append(i)
                    if len(top_ids) >= max_mix_cands:
                        break

        for ii in range(len(top_ids)):
            i = top_ids[ii]
            for jj in range(ii + 1, len(top_ids)):
                j = top_ids[jj]

                ci = infos[i][2]
                cj = infos[j][2]
                edges_i = infos[i][1]
                edges_j = infos[j][1]

                for a in range(1, P):
                    fi = a / P
                    fj = 1.0 - fi
                    w_cost = fi * ci + fj * cj

                    edge_fracs: Dict[Tuple[str, str], float] = {}
                    for e in edges_i:
                        edge_fracs[e] = edge_fracs.get(e, 0.0) + fi
                    for e in edges_j:
                        edge_fracs[e] = edge_fracs.get(e, 0.0) + fj

                    t_mix = _predict_time_seconds_from_edge_fracs(h, edge_fracs)
                    obj_mix = _objective_dollars(w_cost, t_mix, time_cost_per_sec, data_gb=DATA_GB)

                    if (obj_mix < best_obj - 1e-9) or (
                        abs(obj_mix - best_obj) <= 1e-9 and (w_cost, t_mix) < (best_cost, best_time)
                    ):
                        best_obj = obj_mix
                        best_cost = w_cost
                        best_time = t_mix
                        best_plan = ("mix2", i, j, a)

        if P >= 3 and len(top_ids) >= 3:
            for aa in range(len(top_ids)):
                i = top_ids[aa]
                for bb in range(aa + 1, len(top_ids)):
                    j = top_ids[bb]
                    for cc in range(bb + 1, len(top_ids)):
                        k = top_ids[cc]

                        ci = infos[i][2]
                        cj = infos[j][2]
                        ck = infos[k][2]
                        edges_i = infos[i][1]
                        edges_j = infos[j][1]
                        edges_k = infos[k][1]

                        for a in range(1, P - 1):
                            for b in range(1, P - a):
                                c = P - a - b
                                if c <= 0:
                                    continue
                                fi = a / P
                                fj = b / P
                                fk = c / P
                                w_cost = fi * ci + fj * cj + fk * ck

                                edge_fracs: Dict[Tuple[str, str], float] = {}
                                for e in edges_i:
                                    edge_fracs[e] = edge_fracs.get(e, 0.0) + fi
                                for e in edges_j:
                                    edge_fracs[e] = edge_fracs.get(e, 0.0) + fj
                                for e in edges_k:
                                    edge_fracs[e] = edge_fracs.get(e, 0.0) + fk

                                t_mix = _predict_time_seconds_from_edge_fracs(h, edge_fracs)
                                obj_mix = _objective_dollars(w_cost, t_mix, time_cost_per_sec, data_gb=DATA_GB)

                                if not (t_mix < best_time - 1e-9 or w_cost < best_cost - 1e-12):
                                    continue

                                if (obj_mix < best_obj - 1e-9) or (
                                    abs(obj_mix - best_obj) <= 1e-9 and (w_cost, t_mix) < (best_cost, best_time)
                                ):
                                    best_obj = obj_mix
                                    best_cost = w_cost
                                    best_time = t_mix
                                    best_plan = ("mix3", i, j, k, a, b)

        if P >= 4 and P <= 12 and len(top_ids) >= 4:
            for aa in range(len(top_ids)):
                i = top_ids[aa]
                for bb in range(aa + 1, len(top_ids)):
                    j = top_ids[bb]
                    for cc in range(bb + 1, len(top_ids)):
                        k = top_ids[cc]
                        for dd in range(cc + 1, len(top_ids)):
                            l = top_ids[dd]

                            ci = infos[i][2]
                            cj = infos[j][2]
                            ck = infos[k][2]
                            cl = infos[l][2]
                            edges_i = infos[i][1]
                            edges_j = infos[j][1]
                            edges_k = infos[k][1]
                            edges_l = infos[l][1]

                            for a in range(1, P - 2):
                                for b in range(1, P - a - 1):
                                    for c in range(1, P - a - b):
                                        d = P - a - b - c
                                        if d <= 0:
                                            continue
                                        fi = a / P
                                        fj = b / P
                                        fk = c / P
                                        fl = d / P
                                        w_cost = fi * ci + fj * cj + fk * ck + fl * cl

                                        edge_fracs: Dict[Tuple[str, str], float] = {}
                                        for e in edges_i:
                                            edge_fracs[e] = edge_fracs.get(e, 0.0) + fi
                                        for e in edges_j:
                                            edge_fracs[e] = edge_fracs.get(e, 0.0) + fj
                                        for e in edges_k:
                                            edge_fracs[e] = edge_fracs.get(e, 0.0) + fk
                                        for e in edges_l:
                                            edge_fracs[e] = edge_fracs.get(e, 0.0) + fl

                                        t_mix = _predict_time_seconds_from_edge_fracs(h, edge_fracs)
                                        obj_mix = _objective_dollars(w_cost, t_mix, time_cost_per_sec, data_gb=DATA_GB)

                                        if not (t_mix < best_time - 1e-9 or w_cost < best_cost - 1e-12):
                                            continue

                                        if (obj_mix < best_obj - 1e-9) or (
                                            abs(obj_mix - best_obj) <= 1e-9
                                            and (w_cost, t_mix) < (best_cost, best_time)
                                        ):
                                            best_obj = obj_mix
                                            best_cost = w_cost
                                            best_time = t_mix
                                            best_plan = ("mix4", i, j, k, l, a, b, c)

    if best_plan[0] == "single":
        part_to_bucket = [0] * P
        cand_indices = [best_plan[1]]
    elif best_plan[0] == "mix2":
        _, i, j, a = best_plan
        b = P - a
        part_to_bucket = _round_robin_partition_assignment(P, [a, b])
        cand_indices = [i, j]
    elif best_plan[0] == "mix3":
        _, i, j, k, a, b = best_plan
        c = P - a - b
        part_to_bucket = _round_robin_partition_assignment(P, [a, b, c])
        cand_indices = [i, j, k]
    else:  # mix4
        _, i, j, k, l, a, b, c = best_plan
        d = P - a - b - c
        part_to_bucket = _round_robin_partition_assignment(P, [a, b, c, d])
        cand_indices = [i, j, k, l]

    stripe_info_by_cand: Dict[int, Optional[Tuple[str, str, str]]] = {}
    for idx in range(len(infos)):
        stripe_info_by_cand[idx] = _find_aws_two_leaf_siblings_for_striping(h, infos[idx][0])

    bc_topology = BroadCastTopology(src, dsts, P)

    for dst in dsts:
        for p in range(P):
            bucket = part_to_bucket[p]
            cand_idx = cand_indices[bucket]
            dst_paths = infos[cand_idx][0]

            base_path = dst_paths[dst]

            stripe = stripe_info_by_cand.get(cand_idx)
            if stripe is not None:
                _, a, b = stripe
                if dst == a and (p % 2 == 1):
                    base_path = dst_paths[b] + [a]
                elif dst == b and (p % 2 == 0):
                    base_path = dst_paths[a] + [b]

            for u, v in zip(base_path[:-1], base_path[1:]):
                bc_topology.append_dst_partition_path(dst, p, [u, v, G[u][v]])

    return bc_topology


# EVOLVE-BLOCK-END


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(
        self,
        src: str,
        dsts: List[str],
        num_partitions: int = 4,
        paths: Dict[str, SingleDstPath] = None,
    ):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions

        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


# Helper functions (kept safe; no heavy external deps)

def cluster_regions(G, num_clusters=20):
    """Lightweight deterministic down-sampling (placeholder)."""
    nodes = sorted(list(G.nodes))
    if len(nodes) <= num_clusters:
        return nodes
    step = max(1, len(nodes) // num_clusters)
    return nodes[::step][:num_clusters]


def build_subgraph(G, clustered_nodes, hop_limit=2):
    """Keep only edges among clustered nodes and limit hop distances."""
    H = G.subgraph(clustered_nodes).copy()
    to_remove = []
    for u, v in H.edges:
        try:
            if nx.shortest_path_length(H, u, v) > hop_limit:
                to_remove.append((u, v))
        except nx.NetworkXNoPath:
            to_remove.append((u, v))
    H.remove_edges_from(to_remove)
    return H


def update_capacities(G, P_sol, stripe_size, TIME):
    """Reduce edge capacities after each stripe iteration."""
    for (u, v), val in P_sol.items():
        if val > 0.5 and G.has_edge(u, v) and "throughput" in G[u][v]:
            G[u][v]["throughput"] = max(0, G[u][v]["throughput"] - stripe_size / TIME)
    return G


def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    return search_algorithm(src, dsts, G, num_partitions)


class Solution:
    """Glia solution for cloudcast broadcast optimization."""

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