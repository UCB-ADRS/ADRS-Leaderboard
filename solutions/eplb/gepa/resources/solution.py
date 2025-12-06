# EVOLVE-BLOCK-START

import torch


@torch.no_grad()
def _inverse_perm_rows(perm: torch.Tensor) -> torch.Tensor:
    """
    Invert a per-row permutation mapping.

    perm: [B, N] where each row is a permutation of [0..N-1]
    returns inv: [B, N] such that inv[b, perm[b, i]] = i
    """
    i64 = torch.int64
    B, N = perm.shape
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(N, dtype=i64, device=perm.device).expand(B, -1),
    )
    return inv


@torch.no_grad()
def balanced_packing(weight: torch.Tensor, num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n items into m packs with exactly n/m items per pack per row, minimizing imbalance.

    Heuristic: round-based LPT
      - Sort items by weight descending.
      - For each round, assign the next m heaviest items to the m packs with smallest current load.

    Args:
        weight: [X, n], weights per item
        num_packs: m

    Returns:
        pack_index: [X, n], pack id for each item
        rank_in_pack: [X, n], round id (0..groups_per_pack-1)
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0, "balanced_packing requires equal pack sizes."
    groups_per_pack = num_items // num_packs
    device = weight.device
    i64 = torch.int64

    if num_packs == 1:
        pack_index = torch.zeros(num_layers, num_items, dtype=i64, device=device)
        rank_in_pack = torch.arange(num_items, dtype=i64, device=device).unsqueeze(0).expand(num_layers, -1)
        return pack_index, rank_in_pack

    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=i64, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros(num_layers, num_packs, dtype=i64, device=device)
        return pack_index, rank_in_pack

    # Sort weights descending
    w_sorted, idx_sorted = weight.float().sort(-1, descending=True)  # [X, n]

    # Accumulated load per pack
    pack_sums = torch.zeros(num_layers, num_packs, dtype=w_sorted.dtype, device=device)

    pack_index = torch.empty(num_layers, num_items, dtype=i64, device=device)
    rank_in_pack = torch.empty(num_layers, num_items, dtype=i64, device=device)

    # Reusable buffer for scatter_add
    add = torch.empty_like(pack_sums)

    # Tiny deterministic jitter to stabilize tie-breaking across rounds
    jitter = (torch.arange(num_packs, device=device, dtype=w_sorted.dtype) * 1e-7).unsqueeze(0).expand(num_layers, -1)

    for r in range(groups_per_pack):
        start = r * num_packs
        end = start + num_packs

        # Next m heaviest items for this round (per row)
        idx_block = idx_sorted[:, start:end]   # [X, m]
        w_block = w_sorted[:, start:end]       # [X, m]

        # Packs sorted by current load ascending (with deterministic tiny jitter)
        order_packs = (pack_sums + jitter).argsort(dim=1, descending=False)  # [X, m]

        # Assign indices and ranks
        pack_index.scatter_(1, idx_block, order_packs)
        r_full = torch.full(order_packs.shape, r, dtype=i64, device=device)
        rank_in_pack.scatter_(1, idx_block, r_full)

        # Update pack loads
        add.zero_()
        add.scatter_add_(1, order_packs, w_block)
        pack_sums += add

    return pack_index, rank_in_pack


@torch.no_grad()
def _replicate_experts_apportionment(weight: torch.Tensor, num_phy: int) -> torch.Tensor:
    """
    Fast apportionment-based replication counts with at least 1 replica per logical expert.

    Allocates S = num_phy - num_log extra replicas proportionally to expert weights
    using Hamilton's method (largest fractional remainders).

    Args:
        weight: [X, num_log]
        num_phy: int

    Returns:
        logcnt: [X, num_log] (int64)
    """
    i64 = torch.int64
    n, num_log = weight.shape
    device = weight.device
    assert num_phy >= num_log

    if num_phy == num_log:
        return torch.ones(n, num_log, dtype=i64, device=device)

    sum_w = weight.sum(dim=1)  # [n]
    S = num_phy - num_log
    logcnt = torch.ones(n, num_log, dtype=i64, device=device)

    nonzero = sum_w > 0
    if nonzero.any():
        rows = torch.nonzero(nonzero, as_tuple=False).squeeze(-1)
        wv = weight.index_select(0, rows).float()  # [R, num_log]
        sumwv = sum_w.index_select(0, rows).unsqueeze(1)  # [R,1]

        quotas = (wv / sumwv) * float(S)  # [R, num_log]
        base = torch.floor(quotas).to(i64)
        add_sum = base.sum(dim=1)  # [R]
        rem = (S - add_sum).to(i64)  # [R]

        counts = base + 1  # at least one per expert

        # Distribute remaining by largest fractional part
        frac = quotas - base.float()
        # Handle variable-k topk
        kmax = int(rem.max().item()) if rem.numel() > 0 else 0
        kmax = min(kmax, num_log)
        if kmax > 0:
            _, cols_top = frac.topk(kmax, dim=1, largest=True, sorted=False)  # [R, kmax]
            col_range = torch.arange(kmax, device=device, dtype=i64).unsqueeze(0).expand(rows.numel(), -1)
            use_mask = col_range < rem.unsqueeze(1)
            sel_rows_local = torch.arange(rows.numel(), device=device, dtype=i64).unsqueeze(1).expand(-1, kmax)[use_mask]
            sel_cols = cols_top[use_mask]
            if sel_rows_local.numel() > 0:
                sel_rows = rows.index_select(0, sel_rows_local)
                counts.index_put_((sel_rows, sel_cols),
                                  torch.ones_like(sel_rows, dtype=i64),
                                  accumulate=True)
        logcnt.index_copy_(0, rows, counts)

    # All-zero rows: distribute uniformly (at least one each)
    zero_rows = ~nonzero
    if zero_rows.any():
        z_idx = torch.nonzero(zero_rows, as_tuple=False).squeeze(-1)
        Z = z_idx.numel()
        base = torch.ones(Z, num_log, dtype=i64, device=device)
        slack = num_phy - base.sum(dim=1)  # [Z]
        if (slack > 0).any():
            per = (slack // num_log).unsqueeze(1)
            counts = base + per.expand(-1, num_log)
            remu = (slack % num_log)
            if (remu > 0).any():
                kmax = int(remu.max().item())
                k = min(kmax, num_log)
                if k > 0:
                    rows = torch.arange(Z, device=device, dtype=i64).unsqueeze(1).expand(-1, k)
                    cols = torch.arange(num_log, device=device, dtype=i64).unsqueeze(0).expand(Z, -1)[:, :k]
                    col_range = torch.arange(k, device=device, dtype=i64).unsqueeze(0).expand(Z, -1)
                    use_mask = col_range < remu.unsqueeze(1)
                    sel_rows = rows[use_mask]
                    sel_cols = cols[use_mask]
                    if sel_rows.numel() > 0:
                        counts.index_put_((sel_rows, sel_cols),
                                          torch.ones_like(sel_rows, dtype=i64),
                                          accumulate=True)
        else:
            counts = base
        logcnt.index_copy_(0, z_idx, counts)

    return logcnt


@torch.no_grad()
def replicate_experts(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate logical experts to reach num_phy replicas per row while minimizing max per-replica load.

    Hybrid method:
      - Fast apportionment path (Hamilton) for near-uniform rows (no binary search).
      - For skewed rows, use minimax threshold method with vectorized binary search and
        single-pass top-k allocation.

    Args:
        weight: [X, num_log], token loads per logical expert
        num_phy: total replicas per row

    Returns:
        phy2log: [X, num_phy], logical id for each physical expert
        rank:   [X, num_phy], replica rank within its logical expert
        logcnt: [X, num_log], replica count per logical expert
    """
    n, num_log = weight.shape
    assert num_phy >= num_log, "num_phy must be >= num_log."
    device = weight.device
    i64 = torch.int64

    if num_phy == num_log:
        phy2log = torch.arange(num_log, dtype=i64, device=device).repeat(n, 1)
        rank = torch.zeros(n, num_log, dtype=i64, device=device)
        logcnt = torch.ones(n, num_log, dtype=i64, device=device)
        return phy2log, rank, logcnt

    weight_f = weight.float()
    sum_w = weight_f.sum(dim=1)
    max_w = weight_f.max(dim=1).values
    all_zero = (max_w == 0)

    # 1) Fast path: Hamilton apportionment for replication counts
    logcnt = _replicate_experts_apportionment(weight_f, num_phy)  # [n, num_log]

    # 2) Detect skewed rows where apportionment may be suboptimal wrt minimax objective
    #    Use ratio of achieved max per-replica load vs. row lower bound (sum/m).
    with torch.no_grad():
        per_replica = weight_f / logcnt.clamp_min(1).float()
        Lmax = per_replica.max(dim=1).values  # [n]
        LB = (sum_w / float(num_phy)).clamp_min(1e-8)
        ratio = torch.where(LB > 0, Lmax / LB, torch.zeros_like(Lmax))
        # Consider also coefficient of variation as a skew signal
        mean_w = (sum_w / float(num_log)).clamp_min(1e-8)
        cv = torch.sqrt(((weight_f - mean_w.unsqueeze(1))**2).mean(dim=1)) / mean_w
        # Row is "good enough" if ratio close to 1 and not highly skewed.
        good = (ratio <= 1.06) | (cv <= 0.10)  # thresholds tuned for speed/quality trade-off
        # Force re-opt if there is an extremely heavy single expert
        heavy_single = (max_w >= 0.65 * sum_w) & (sum_w > 0)
        good = good & (~heavy_single)

    # 3) For bad rows, run minimax binary search + top-k refinement
    bad_mask = ~good & (~all_zero)
    if bad_mask.any():
        b_idx = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1)
        wb = weight_f.index_select(0, b_idx)  # [B, num_log]
        B = wb.size(0)

        maxwb = wb.max(dim=1).values
        sumwb = wb.sum(dim=1)

        # Bracket for T (per-row)
        low = (sumwb / float(num_phy)).clamp_min(1e-8)     # [B]
        high = torch.maximum(maxwb, low)                   # [B]

        # Vectorized binary search with early stop on relative gap
        for _ in range(12):
            mid = (low + high) * 0.5
            s = torch.ceil(wb / mid.unsqueeze(1)).clamp_min(1.0).sum(dim=1)
            too_many = s > num_phy
            low = torch.where(too_many, mid, low)
            high = torch.where(too_many, high, mid)
            if ((high - low) <= (1e-5 * torch.maximum(high, torch.ones_like(high)))).all():
                break

        counts = torch.ceil(wb / high.unsqueeze(1)).clamp_min(1.0).to(i64)  # [B, num_log]
        base_sum = counts.sum(dim=1)
        rem = (num_phy - base_sum).to(i64)                                  # [B]

        rows = torch.nonzero(rem > 0, as_tuple=False).squeeze(-1)
        if rows.numel() > 0:
            c_sub = counts.index_select(0, rows).float()
            w_sub = wb.index_select(0, rows)
            next_thr = w_sub / (c_sub + 1.0)  # [R, num_log]

            rem_sub = rem.index_select(0, rows)
            kmax = int(rem_sub.max().item())
            kmax = min(kmax, num_log)
            if kmax > 0:
                _, cols_top = next_thr.topk(kmax, dim=1, largest=True, sorted=False)  # [R, kmax]
                col_range = torch.arange(kmax, device=device, dtype=i64).unsqueeze(0).expand(rows.numel(), -1)
                use_mask = col_range < rem_sub.unsqueeze(1)
                sel_rows_local = torch.arange(rows.numel(), device=device, dtype=i64).unsqueeze(1).expand(-1, kmax)[use_mask]
                sel_cols = cols_top[use_mask]
                if sel_rows_local.numel() > 0:
                    sel_rows = rows.index_select(0, sel_rows_local)
                    counts.index_put_((sel_rows, sel_cols),
                                      torch.ones_like(sel_rows, dtype=i64),
                                      accumulate=True)

        # Commit refined counts back
        logcnt.index_copy_(0, b_idx, counts)

    if __debug__:
        row_sums = logcnt.sum(dim=1)
        assert torch.all(row_sums == num_phy), "Replication counts do not sum to num_phy."

    # Build phy2log (logical id per physical replica) and ranks
    n, m = logcnt.shape
    counts_flat = logcnt.reshape(-1)  # [n*m]
    comp_idx = torch.arange(n * m, device=device, dtype=i64)
    seq = torch.repeat_interleave(comp_idx, counts_flat)  # length n*num_phy
    phy2log = (seq % m).view(n, num_phy)

    # Replica ranks within each logical expert
    starts = counts_flat.cumsum(0) - counts_flat
    starts_seq = torch.repeat_interleave(starts, counts_flat)
    abs_pos = torch.arange(seq.numel(), device=device, dtype=i64)
    rank = (abs_pos - starts_seq).view(n, num_phy)

    return phy2log, rank, logcnt


@torch.no_grad()
def _per_gpu_load_from_mapping(
    weight: torch.Tensor,
    logcnt: torch.Tensor,
    pphy2log: torch.Tensor,
    num_gpus: int,
) -> torch.Tensor:
    """
    Compute per-GPU approximate loads from final mapping.

    Args:
        weight: [L, M]
        logcnt: [L, M]
        pphy2log: [L, E]
        num_gpus: G

    Returns:
        gpu_loads: [L, G]
    """
    L, E = pphy2log.shape
    per_gpu = E // num_gpus
    device = weight.device
    dtype = weight.dtype

    approx_phy_load = (weight / logcnt.clamp_min(1)).gather(-1, pphy2log).float()  # [L, E]
    gpu_ids = (torch.arange(E, device=device, dtype=torch.int64) // per_gpu).unsqueeze(0).expand(L, -1)
    gpu_loads = torch.zeros(L, num_gpus, dtype=dtype, device=device)
    gpu_loads.scatter_add_(1, gpu_ids, approx_phy_load)
    return gpu_loads


@torch.no_grad()
def _imbalance_score(gpu_loads: torch.Tensor) -> torch.Tensor:
    """
    Compute worst-layer imbalance ratio: max(load) / mean(load).
    gpu_loads: [L, G]
    Returns scalar tensor
    """
    eps = 1e-8
    max_per_layer = gpu_loads.max(dim=1).values
    mean_per_layer = gpu_loads.mean(dim=1).clamp_min(eps)
    ratio = max_per_layer / mean_per_layer
    return ratio.max()


@torch.no_grad()
def rebalance_experts_global(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Global-first strategy:
      1) Replicate globally (hybrid apportionment/minimax).
      2) Directly pack all physical experts to GPUs with exact equal counts per GPU.

    Args:
        weight: [L, num_logical_experts]
        num_physical_experts: total physical experts after replication
        num_nodes: number of nodes (for sanity, must divide num_gpus)
        num_gpus: total GPUs (multiple of num_nodes)

    Returns:
        pphy2log: [L, num_physical_experts] logical id per physical expert (GPU-major indexing)
        pphyrank: [L, num_physical_experts] replica rank within its logical expert
        logcnt:   [L, num_logical_experts] replica count per logical expert
    """
    L, num_logical = weight.shape
    assert num_physical_experts % num_gpus == 0, "num_physical_experts must be divisible by num_gpus."
    assert num_gpus % num_nodes == 0, "num_gpus must be divisible by num_nodes."
    per_gpu = num_physical_experts // num_gpus

    i64 = torch.int64
    device = weight.device

    # Step 1: global replication (hybrid)
    phy2log, phyrank, logcnt = replicate_experts(weight, num_physical_experts)  # [L, E], [L, E], [L, M]

    # Approximate per-physical load
    approx_phy_load = (weight / logcnt.clamp_min(1)).gather(-1, phy2log).float()  # [L, E]

    # Step 2: direct GPU packing (equal counts per GPU)
    gpu_pack_index, gpu_rank_in_pack = balanced_packing(approx_phy_load, num_gpus)  # [L, E]
    phy2pphy = gpu_pack_index * per_gpu + gpu_rank_in_pack                         # [L, E]
    pphy2phy = _inverse_perm_rows(phy2pphy)                                        # [L, E]

    # Materialize final logical ids and replica ranks in pphy order
    pphy2log = phy2log.gather(-1, pphy2phy)    # [L, E]
    pphyrank = phyrank.gather(-1, pphy2phy)    # [L, E]
    return pphy2log, pphyrank, logcnt


@torch.no_grad()
def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Two-level hierarchical balancing:
      1) Pack expert groups to nodes with equal group count per node.
      2) Replicate within each node to meet its replica budget.
      3) Pack physical experts to GPUs within each node with equal per-GPU counts.

    Args:
        weight: [num_layers, num_logical_experts]
        num_physical_experts: total physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of nodes
        num_gpus: total GPUs (multiple of num_nodes)

    Returns:
        pphy2log: [L, num_physical_experts] logical id per physical expert (global indexing)
        pphyrank: [L, num_physical_experts] replica rank within its logical expert
        logcnt:   [L, num_logical_experts] replica count per logical expert
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0, "num_logical_experts must be divisible by num_groups."
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0, "num_groups must be divisible by num_nodes."
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0, "num_gpus must be divisible by num_nodes."
    assert num_physical_experts % num_gpus == 0, "num_physical_experts must be divisible by num_gpus."
    phy_experts_per_gpu = num_physical_experts // num_gpus

    device = weight.device
    i64 = torch.int64

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, G]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=i64, device=device)).flatten(-2)
    mlog2log = _inverse_perm_rows(log2mlog)

    # Step 2: replicate within nodes (per-node budget)
    per_node_log = num_logical_experts // num_nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, per_node_log)  # [L*N, M/N]
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs within each node
    approx_phy_load = (tokens_per_mlog / mlogcnt.clamp_min(1)).gather(-1, phy2mlog)
    packs_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = balanced_packing(approx_phy_load, packs_per_node)  # [L*N, per_node]
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse_perm_rows(phy2pphy)

    # Map back to global logical ids and flatten node dimension
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [L*N, E_per_node]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(0, num_logical_experts, per_node_log, device=device).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)                     # [L, num_physical_experts]
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)  # [L, num_physical_experts]
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)    # [L, num_logical_experts]
    return pphy2log, pphyrank, logcnt


@torch.no_grad()
def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Default: Global-first replication + direct GPU packing.
    Fallback: If imbalance remains high and hierarchical constraints are valid,
              use hierarchical strategy and choose the better of the two.

    Args:
        weight: [layers, num_logical_experts]
        num_replicas: total physical experts (multiple of num_gpus)
        num_groups: number of expert groups
        num_nodes: number of nodes
        num_gpus: total number of GPUs

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, max_replica_count] (padded with -1)
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()

    # Strategy 1: Global-first
    phy2log_g, phyrank_g, logcnt_g = rebalance_experts_global(
        weight, num_replicas, num_nodes, num_gpus
    )

    # Evaluate imbalance for global mapping
    gpu_loads_g = _per_gpu_load_from_mapping(weight, logcnt_g, phy2log_g, num_gpus)
    score_g = _imbalance_score(gpu_loads_g)

    use_global = True

    # Strategy 2: Hierarchical (only if constraints allow and global imbalance is pronounced)
    can_hier = (
        (num_logical_experts % max(num_groups, 1) == 0)
        and (num_groups % max(num_nodes, 1) == 0)
        and (num_replicas % max(num_gpus, 1) == 0)
        and (num_gpus % max(num_nodes, 1) == 0)
    )

    # Threshold when to consider hierarchical alternative
    consider_hier = (score_g > 1.06) and can_hier

    if consider_hier:
        try:
            phy2log_h, phyrank_h, logcnt_h = rebalance_experts_hierarchical(
                weight, num_replicas, num_groups, num_nodes, num_gpus
            )
            gpu_loads_h = _per_gpu_load_from_mapping(weight, logcnt_h, phy2log_h, num_gpus)
            score_h = _imbalance_score(gpu_loads_h)

            # Choose mapping with better worst-layer imbalance; prefer global if tie
            if score_h + 1e-8 < score_g:
                phy2log, phyrank, logcnt = phy2log_h, phyrank_h, logcnt_h
                use_global = False
            else:
                phy2log, phyrank, logcnt = phy2log_g, phyrank_g, logcnt_g
        except AssertionError:
            # If hierarchical preconditions don't hold, fallback to global
            phy2log, phyrank, logcnt = phy2log_g, phyrank_g, logcnt_g
            use_global = True
    else:
        phy2log, phyrank, logcnt = phy2log_g, phyrank_g, logcnt_g

    # Build logical->physical (ragged) with -1 padding
    maxlogcnt = int(logcnt.max().item())
    device = weight.device
    i64 = torch.int64

    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=i64,
        device=device,
    )
    physical_ids = torch.arange(num_replicas, dtype=i64, device=device).expand(num_layers, -1)

    # Scatter physical indices into per-logical slots using replica rank as slot
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        physical_ids,
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END


from pathlib import Path
from typing import Any, Dict

class Solution:
    """GEPA-evolved solution for EPLB."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Extract code before Solution class definition
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}