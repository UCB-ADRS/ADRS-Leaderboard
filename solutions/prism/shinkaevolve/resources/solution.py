# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs
   New approach: exact feasibility via parametric bin packing + binary search on T.
"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs (dict: gpu_idx -> list of model objects)
    """

    if gpu_num <= 0:
        return {}

    # Extract per-model parameters
    a_list = [float(m.req_rate) / float(m.slo) if m.slo != 0 else float('inf') for m in models]  # pressure a_i
    s_list = [float(m.model_size) for m in models]                                               # size s_i
    n = len(models)
    C = float(GPU_MEM_SIZE)
    eps = 1e-12

    # Helpers
    def _kvpr_from(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    # Lower bound on T
    def _lower_bound_T():
        total_a = sum(a_list)
        total_s = sum(s_list)
        total_den = gpu_num * C - total_s
        lb_global = _kvpr_from(total_a, total_den)

        lb_single = 0.0
        for a, s in zip(a_list, s_list):
            den = C - s
            lb_single = max(lb_single, _kvpr_from(a, den))

        return max(0.0, lb_global, lb_single)

    # A quick greedy placement that returns both placement and achieved max KVPR (for UB init)
    def _greedy_kvpr_placement():
        # Order by "value per GB" then by pressure to reduce crowding early
        def density(idx):
            sz = s_list[idx] if s_list[idx] > 0 else 1e-9
            return a_list[idx] / sz
        order = sorted(range(n), key=lambda i: (density(i), a_list[i]), reverse=True)

        placement = {i: [] for i in range(gpu_num)}
        rem = [C for _ in range(gpu_num)]
        wr = [0.0 for _ in range(gpu_num)]

        for i in order:
            a = a_list[i]
            s = s_list[i]

            best_g = None
            best_global = float('inf')
            best_local = float('inf')
            best_rem = -1.0

            for g in range(gpu_num):
                if s <= rem[g] + eps:
                    new_wr = wr[g] + a
                    new_rem = rem[g] - s
                    # resulting global max KVPR
                    new_local = _kvpr_from(new_wr, new_rem)
                    cand_global = new_local
                    for j in range(gpu_num):
                        if j == g:
                            continue
                        kv = _kvpr_from(wr[j], rem[j])
                        if kv > cand_global:
                            cand_global = kv
                    if (cand_global < best_global or
                        (abs(cand_global - best_global) <= 1e-12 and
                         (new_local < best_local or
                          (abs(new_local - best_local) <= 1e-12 and new_rem > best_rem)))):
                        best_g = g
                        best_global = cand_global
                        best_local = new_local
                        best_rem = new_rem

            if best_g is None:
                # memory infeasible with this simple greedy; place on GPU with most space as last resort
                g = max(range(gpu_num), key=lambda x: rem[x])
                placement[g].append(i)
                wr[g] += a
                rem[g] -= s
            else:
                placement[best_g].append(i)
                wr[best_g] += a
                rem[best_g] -= s

        gmax = 0.0
        for g in range(gpu_num):
            gmax = max(gmax, _kvpr_from(wr[g], rem[g]))
        return placement, wr, rem, gmax

    # Feasibility check for a given T via bin packing over weights w_i(T) = a_i + T*s_i, bin cap = T * C
    def _pack_feasible(T, node_cap=150000):
        cap = T * C
        # Build items
        items = list(range(n))
        w = [a_list[i] + T * s_list[i] for i in items]
        # Quick necessary checks
        if any(wi > cap + 1e-12 for wi in w):
            return None
        total_w = sum(w)
        if total_w > gpu_num * cap + 1e-9:
            return None

        # Sort items by descending weight (tie by larger size then pressure)
        items.sort(key=lambda i: (w[i], s_list[i], a_list[i]), reverse=True)

        # First try a fast greedy (best-fit) to avoid DFS if possible
        used_w = [0.0] * gpu_num
        used_s = [0.0] * gpu_num
        cnt = [0] * gpu_num
        assign = [-1] * n

        def greedy_fit():
            for k, idx in enumerate(items):
                best_g = None
                best_res = float('inf')
                best_sres = float('inf')
                wi = w[idx]
                si = s_list[idx]
                for g in range(gpu_num):
                    if used_w[g] + wi <= cap + 1e-12 and used_s[g] + si <= C + 1e-12:
                        res = cap - (used_w[g] + wi)
                        sres = C - (used_s[g] + si)
                        # Best fit by weight residual, tie by memory residual, then by fewer items
                        if (res < best_res - 1e-12 or
                            (abs(res - best_res) <= 1e-12 and (sres < best_sres - 1e-12 or
                                                               (abs(sres - best_sres) <= 1e-12 and cnt[g] < (cnt[best_g] if best_g is not None else float('inf')))))):
                            best_g = g
                            best_res = res
                            best_sres = sres
                if best_g is None:
                    return False
                assign[idx] = best_g
                used_w[best_g] += wi
                used_s[best_g] += si
                cnt[best_g] += 1
            return True

        if greedy_fit():
            # Build placement from assign
            placement = {i: [] for i in range(gpu_num)}
            for idx in range(n):
                placement[assign[idx]].append(idx)
            return placement

        # Reset for DFS
        used_w = [0.0] * gpu_num
        used_s = [0.0] * gpu_num
        cnt = [0] * gpu_num
        assign = [-1] * n

        # Symmetry-aware DFS with memoization
        from functools import lru_cache
        nodes = [0]  # mutable counter

        def state_key(k):
            # Sort bin loads to avoid permutation duplicates; round to reduce float noise
            # Include both weight and memory usage to tighten pruning
            sig = tuple(sorted((round(used_w[g], 6), round(used_s[g], 6)) for g in range(gpu_num)))
            return (k, sig)

        seen = set()

        def dfs(k):
            if nodes[0] > node_cap:
                return False
            if k == len(items):
                return True
            key = state_key(k)
            if key in seen:
                return False
            seen.add(key)

            idx = items[k]
            wi = w[idx]
            si = s_list[idx]

            # Candidate bins: those where it fits; order by resulting residual (best-fit)
            candidates = []
            for g in range(gpu_num):
                if used_w[g] + wi <= cap + 1e-12 and used_s[g] + si <= C + 1e-12:
                    res = cap - (used_w[g] + wi)
                    sres = C - (used_s[g] + si)
                    candidates.append((res, sres, cnt[g], g))
            if not candidates:
                return False
            candidates.sort(key=lambda t: (t[0], t[1], t[2]))  # best-fit by weight, then memory, then fewer items

            # Symmetry breaking: try at most one empty bin at this level; skip bins with identical (used_w, used_s)
            tried_empty = False
            tried_signatures = set()

            for _, _, _, g in candidates:
                is_empty = (cnt[g] == 0)
                sig = (round(used_w[g], 6), round(used_s[g], 6))
                if sig in tried_signatures:
                    continue
                if is_empty and tried_empty:
                    continue

                # Place
                assign[idx] = g
                used_w[g] += wi
                used_s[g] += si
                cnt[g] += 1
                nodes[0] += 1

                if dfs(k + 1):
                    return True

                # Undo
                cnt[g] -= 1
                used_s[g] -= si
                used_w[g] -= wi
                assign[idx] = -1

                tried_signatures.add(sig)
                if is_empty:
                    tried_empty = True

            return False

        ok = dfs(0)
        if not ok:
            return None

        placement = {i: [] for i in range(gpu_num)}
        for idx in range(n):
            placement[assign[idx]].append(idx)
        return placement

    # Search minimal T with exponential expansion then binary search
    lb = _lower_bound_T()
    if lb == float('inf') or lb != lb:  # nan guard
        # Degenerate: place greedily just to return something deterministic
        greedy_res = _greedy_kvpr_placement()[0]
        return {g: [models[i] for i in greedy_res[g]] for g in range(gpu_num)}

    # Exponential search for feasible UB
    T = max(lb, 0.0)
    # Use a small positive if lb is 0 to avoid cap=0/NaN issues when some a_i=0 and s_i>0
    if T == 0.0:
        T = 1e-9
    ub_placement = None
    max_expand = 32
    factor = 1.6

    # Try greedy-derived UB to accelerate
    _, wr_g, rem_g, gmax_greedy = _greedy_kvpr_placement()
    if gmax_greedy != float('inf') and gmax_greedy == gmax_greedy:
        T_try = max(T, gmax_greedy)
    else:
        T_try = T

    # Expand until feasible
    it = 0
    while it < max_expand:
        pl = _pack_feasible(T_try)
        if pl is not None:
            ub = T_try
            ub_placement = pl
            break
        T_try *= factor
        it += 1
    else:
        # As a final fallback, try a very large T
        T_try = max(T_try, 1e6)
        pl = _pack_feasible(T_try)
        if pl is None:
            # If still not feasible, return greedy placement (should be rare)
            greedy_pl = _greedy_kvpr_placement()[0]
            return {g: [models[i] for i in greedy_pl[g]] for g in range(gpu_num)}
        ub = T_try
        ub_placement = pl

    lb_cur = lb
    ub_cur = ub
    best_pl = ub_placement

    # Binary search on T with feasibility by packing
    for _ in range(28):
        mid = 0.5 * (lb_cur + ub_cur)
        pl = _pack_feasible(mid)
        if pl is not None:
            best_pl = pl
            ub_cur = mid
        else:
            lb_cur = mid
        if ub_cur - lb_cur <= max(1e-9, 1e-6 * (1.0 + ub_cur)):
            break

    # Build final placement mapping model objects
    placement = {g: [] for g in range(gpu_num)}
    for g in range(gpu_num):
        for idx in best_pl.get(g, []):
            placement[g].append(models[idx])

    return placement

# EVOLVE-BLOCK-END


def run_placement(gpu_num, models):
    """
    Main entry point that will be called by the evaluator.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        Dictionary containing GPU placements
    """
    return compute_model_placement(gpu_num, models)


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Any, Dict

class Solution:
    """Shinka solution for prism GPU model placement."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code
        code = Path(__file__).read_text(encoding="utf-8")
        # Extract everything up to the Solution class
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}