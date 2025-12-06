GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    KVPR(gpu) = sum(req_rate/slo for models on gpu) / (GPU_MEM_SIZE - sum(model_size))
    If remaining memory is 0 or less, KVPR is treated as infinity.

    Strategy:
      - Use a feasibility-based search on target max KVPR (T).
      - For a fixed T, each GPU must satisfy: sum(w_i) / (GPU_MEM_SIZE - sum(s_i)) <= T
        which is equivalent to sum(w_i + T*s_i) <= T * GPU_MEM_SIZE with s_i <= GPU_MEM_SIZE.
      - Perform binary search on T and use a greedy Best-Fit-Decreasing assignment on the
        transformed sizes (w_i + T*s_i) while respecting memory capacity.
      - If binary search fails due to greedy myopia, fall back to a global look-ahead greedy.
      - Optionally apply a small local improvement step (single-model moves).

    Args:
        gpu_num: Number of GPUs (0..gpu_num-1 are valid IDs)
        models: List of Model objects to place. Each Model has:
                - model_name: str
                - model_size: int (GB)
                - req_rate: int
                - slo: int
                - cur_gpu_id: int (unused)

    Returns:
        A dict mapping gpu_id -> list[Model], with all models placed.
    """
    eps = 1e-12

    if gpu_num <= 0:
        return {}

    # Helper functions
    def weight_of(m):
        if m.slo == 0:
            # If both req_rate and slo are zero, their weight is 0; else infeasible
            return 0.0 if m.req_rate == 0 else float('inf')
        return m.req_rate / m.slo

    def kvpr_of(weights, rems, gid):
        rem = rems[gid]
        if rem <= 0:
            return float('inf')
        return weights[gid] / rem

    def compute_max_kvpr(weights, rems):
        worst = 0.0
        for i in range(gpu_num):
            v = kvpr_of(weights, rems, i)
            if v > worst:
                worst = v
        return worst

    # Validate immediate infeasibilities and precompute
    sizes = []
    weights = []
    total_size = 0.0
    total_weight = 0.0
    for m in models:
        if m.model_size > GPU_MEM_SIZE:
            raise ValueError(f"Model {m.model_name} size {m.model_size}GB exceeds GPU capacity {GPU_MEM_SIZE}GB")
        w = weight_of(m)
        if w == float('inf'):
            raise ValueError(f"Model {m.model_name} has infinite weight (req_rate>0 and slo=0), infeasible.")
        s = float(m.model_size)
        sizes.append(s)
        weights.append(float(w))
        total_size += s
        total_weight += w

    if total_size - eps > gpu_num * GPU_MEM_SIZE:
        raise ValueError(
            f"Total model size {total_size:.3f}GB exceeds aggregate GPU memory {gpu_num * GPU_MEM_SIZE}GB"
        )

    # Lower bound for T from simple necessary conditions
    def safe_div(a, b):
        return a / b if b > eps else float('inf')

    lb_items = max((safe_div(weights[i], max(eps, GPU_MEM_SIZE - sizes[i])) for i in range(len(models))), default=0.0)
    lb_global = safe_div(total_weight, max(eps, gpu_num * GPU_MEM_SIZE - total_size))
    T_lower = max(0.0, lb_items, lb_global)

    # Greedy feasibility check for a given T using Best-Fit-Decreasing on transformed sizes
    def feasible_with_assignment(T):
        # Transformed capacity and item sizes: cap1 = T * 80, item = w + T*s
        cap1 = T * GPU_MEM_SIZE
        items = []
        for idx, m in enumerate(models):
            transformed = weights[idx] + T * sizes[idx]
            items.append((idx, transformed, sizes[idx]))
        # Sort by descending transformed size then by size
        items.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Initialize per-GPU capacities
        cap1_rem = [cap1 for _ in range(gpu_num)]
        mem_rem = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
        placement_idx = {i: [] for i in range(gpu_num)}

        for idx, tsize, s in items:
            best_gpu = None
            best_key = None
            for gid in range(gpu_num):
                if mem_rem[gid] + eps < s:
                    continue
                if cap1_rem[gid] + eps < tsize:
                    continue
                # Best-Fit-Decreasing on transformed capacity, tie-break by more mem left after placement
                residual1 = cap1_rem[gid] - tsize
                mem_after = mem_rem[gid] - s
                key = (residual1, -mem_after, gid)
                if (best_key is None) or (key < best_key):
                    best_key = key
                    best_gpu = gid
            if best_gpu is None:
                return False, None
            placement_idx[best_gpu].append(idx)
            cap1_rem[best_gpu] -= tsize
            mem_rem[best_gpu] -= s

        # Build placement by models
        placement = {i: [] for i in range(gpu_num)}
        for gid in range(gpu_num):
            for idx in placement_idx[gid]:
                placement[gid].append(models[idx])
        return True, placement

    # Find an upper bound by doubling until feasible (or give up after a limit)
    T = max(T_lower, 1e-9)
    best_T = None
    best_placement = None
    found = False
    for _ in range(60):
        ok, placement_try = feasible_with_assignment(T)
        if ok:
            best_T = T
            best_placement = placement_try
            found = True
            break
        T *= 2.0

    # If doubling failed (rare with reasonable data), fall back to global greedy with look-ahead
    if not found:
        # Fallback: Greedy with global look-ahead minimizing future max KVPR
        sorted_models = sorted(
            models,
            key=lambda m: (weight_of(m), m.model_size),
            reverse=True
        )
        placement = {i: [] for i in range(gpu_num)}
        rem = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
        wsum = [0.0 for _ in range(gpu_num)]

        for m in sorted_models:
            s = float(m.model_size)
            w = weight_of(m)
            best_gpu = None
            best_key = None
            for gid in range(gpu_num):
                if rem[gid] < s - eps:
                    continue
                # simulate
                new_rem = rem[gid] - s
                new_w = wsum[gid] + w
                # compute future max kvpr
                max_val = 0.0
                for j in range(gpu_num):
                    if j == gid:
                        v = float('inf') if new_rem <= 0 else (new_w / new_rem)
                    else:
                        v = float('inf') if rem[j] <= 0 else (wsum[j] / rem[j])
                    if v > max_val:
                        max_val = v
                # tie-break: minimize future max; then keep more remaining memory; then lower gpu id
                key = (max_val, -new_rem, gid)
                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gid
            if best_gpu is None:
                raise ValueError("Unable to place all models under memory constraints.")
            placement[best_gpu].append(m)
            wsum[best_gpu] += w
            rem[best_gpu] -= s

        # One pass of local improvement (single-model move from worst GPU)
        def try_single_move():
            kvprs = [kvpr_of(wsum, rem, i) for i in range(gpu_num)]
            worst_gpu = max(range(gpu_num), key=lambda i: kvprs[i])
            current_max = kvprs[worst_gpu]
            if not placement[worst_gpu]:
                return False
            for m in list(placement[worst_gpu]):
                s = float(m.model_size)
                w = weight_of(m)
                for dest in range(gpu_num):
                    if dest == worst_gpu:
                        continue
                    if rem[dest] + eps < s:
                        continue
                    # simulate
                    new_rem_worst = rem[worst_gpu] + s
                    new_w_worst = wsum[worst_gpu] - w
                    new_rem_dest = rem[dest] - s
                    new_w_dest = wsum[dest] + w
                    # compute new max
                    max_val = 0.0
                    for j in range(gpu_num):
                        if j == worst_gpu:
                            v = float('inf') if new_rem_worst <= 0 else (new_w_worst / new_rem_worst)
                        elif j == dest:
                            v = float('inf') if new_rem_dest <= 0 else (new_w_dest / new_rem_dest)
                        else:
                            v = float('inf') if rem[j] <= 0 else (wsum[j] / rem[j])
                        if v > max_val:
                            max_val = v
                    if max_val + 1e-12 < current_max:
                        # apply
                        placement[worst_gpu].remove(m)
                        placement[dest].append(m)
                        wsum[worst_gpu] = new_w_worst
                        rem[worst_gpu] = new_rem_worst
                        wsum[dest] = new_w_dest
                        rem[dest] = new_rem_dest
                        return True
            return False

        if try_single_move():
            pass  # applied one improving move

        return placement

    # Binary search between T_lower and best_T to refine
    low, high = T_lower, best_T
    final_placement = best_placement
    for _ in range(40):
        mid = (low + high) / 2.0
        ok, placement_try = feasible_with_assignment(mid)
        if ok:
            high = mid
            final_placement = placement_try
        else:
            low = mid

    # Optional tiny local improvement on the final placement
    # Compute current per-GPU stats
    rem = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    wsum = [0.0 for _ in range(gpu_num)]
    for gid in range(gpu_num):
        for m in final_placement[gid]:
            rem[gid] -= float(m.model_size)
            wsum[gid] += weight_of(m)

    def try_single_move_final():
        kvprs = [kvpr_of(wsum, rem, i) for i in range(gpu_num)]
        worst_gpu = max(range(gpu_num), key=lambda i: kvprs[i])
        current_max = kvprs[worst_gpu]
        if not final_placement[worst_gpu]:
            return False
        for m in list(final_placement[worst_gpu]):
            s = float(m.model_size)
            w = weight_of(m)
            for dest in range(gpu_num):
                if dest == worst_gpu:
                    continue
                if rem[dest] + eps < s:
                    continue
                # simulate
                new_rem_worst = rem[worst_gpu] + s
                new_w_worst = wsum[worst_gpu] - w
                new_rem_dest = rem[dest] - s
                new_w_dest = wsum[dest] + w
                # compute new max
                max_val = 0.0
                for j in range(gpu_num):
                    if j == worst_gpu:
                        v = float('inf') if new_rem_worst <= 0 else (new_w_worst / new_rem_worst)
                    elif j == dest:
                        v = float('inf') if new_rem_dest <= 0 else (new_w_dest / new_rem_dest)
                    else:
                        v = float('inf') if rem[j] <= 0 else (wsum[j] / rem[j])
                    if v > max_val:
                        max_val = v
                if max_val + 1e-12 < current_max:
                    # apply
                    final_placement[worst_gpu].remove(m)
                    final_placement[dest].append(m)
                    wsum[worst_gpu] = new_w_worst
                    rem[worst_gpu] = new_rem_worst
                    wsum[dest] = new_w_dest
                    rem[dest] = new_rem_dest
                    return True
        return False

    try_single_move_final()

    return final_placement

# EVOLVE-BLOCK-END


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Any, Dict

class Solution:
    """GEPA solution for prism GPU model placement."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code
        code = Path(__file__).read_text(encoding="utf-8")
        # Extract everything up to the Solution class
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}