# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using
Vectorized Marginal Gain Replication (VMGR) and Direct Thue-Morse Packing.

Key Innovations:
1. Direct Thue-Morse Permutation: Generates the packing gather indices directly
   from grid coordinates in O(N) time, bypassing the need for an inverse argsort.
2. Native Precision VMGR: computes replication gains in native precision to
   maximize memory bandwidth efficiency on GPUs.
3. Multiplicative Marginal Gains: Replaces division with multiplication by
   reciprocals in the replication kernel.
"""

import torch
import functools

@functools.lru_cache(maxsize=64)
def _get_packing_gather_perm(items_per_pack: int, num_packs: int,
                             device: torch.device) -> torch.Tensor:
    """
    Generate the gather permutation for Thue-Morse constrained packing directly.
    Returns indices P such that sorted_items[P] is the item at the k-th packed slot.

    Mapping Logic:
    Output Slot s -> (Pack p, Row r) -> Input Grid Column c -> Input Rank k
    """
    # 1. Generate Thue-Morse sequence for the rows
    # tm_seq[r] == 1 implies the row r is filled right-to-left
    if hasattr(torch.Tensor, "bitwise_count"):
        tm_seq = (torch.arange(items_per_pack, device=device).bitwise_count() & 1).bool()
    else:
        seq = torch.tensor([0], dtype=torch.bool, device=device)
        while seq.numel() < items_per_pack:
            seq = torch.cat([seq, ~seq])
        tm_seq = seq[:items_per_pack]

    # 2. Compute the permutation indices directly
    # Slot index s goes from 0 to (items_per_pack * num_packs) - 1
    # We interpret s as iterating through Packs first, then Rows (Output Order)
    # Slot s corresponds to Pack p and Row r in the packed output.
    # However, standard packing fills Pack 0, then Pack 1, etc.
    # So s = p * items_per_pack + r

    total_items = items_per_pack * num_packs
    # Use int32 for index math to save register/bandwidth, cast to int64 for gather
    slots = torch.arange(total_items, device=device, dtype=torch.int32)

    p = slots // items_per_pack
    r = slots % items_per_pack

    # Determine direction based on row r
    is_reversed = tm_seq[r]

    # Map (Pack p, Row r) to Column c in the sorted item grid.
    # If Row r is standard (0): Item at Col c goes to Pack c. So p = c.
    # If Row r is reversed (1): Item at Col c goes to Pack M-1-c. So p = M-1-c => c = M-1-p.
    c = torch.where(is_reversed, num_packs - 1 - p, p)

    # Calculate original Sorted Rank k
    # The sorted items are arranged in a grid of [items_per_pack, num_packs]
    # Rank k = Row r * num_packs + Column c
    perm = r * num_packs + c

    return perm.to(torch.int64)


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> torch.Tensor:
    """
    Pack items to packs using Thue-Morse constrained sorting strategy.
    Returns the permutation [layers, n] that maps PackedSlot -> OriginalItem.
    """
    num_layers, num_items = weight.shape
    device = weight.device

    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    if items_per_pack == 1:
        # Identity mapping if 1 item per pack (already balanced)
        return torch.arange(num_items, device=device).expand(num_layers, num_items)

    # Sort items descending: [num_layers, num_items]
    # indices maps SortedRank -> OriginalItem
    indices = weight.argsort(dim=-1, descending=True)

    # Get Slot -> SortedRank mapping (Cached & Optimized)
    # This P tells us: Which SortedRank sits at OutputSlot s?
    slot2rank = _get_packing_gather_perm(items_per_pack, num_packs, device)

    # Expand to batch dimension
    slot2rank = slot2rank.unsqueeze(0).expand(num_layers, -1)

    # Map Slot -> OriginalItem using a single gather
    # Output[s] = indices[slot2rank[s]]
    return indices.gather(1, slot2rank)


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas using Vectorized Marginal Gain.
    """
    num_layers, num_log = weight.shape
    num_redundant = num_phy - num_log
    device = weight.device

    # Pre-allocate output tensors
    phy2log = torch.empty(num_layers, num_phy, dtype=torch.int64, device=device)
    rank = torch.empty(num_layers, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)

    # Base initialization: First num_log experts map 1:1
    base_indices = torch.arange(num_log, dtype=torch.int64, device=device).expand(num_layers, num_log)
    phy2log[:, :num_log] = base_indices
    rank[:, :num_log] = 0

    if num_redundant == 0:
        return phy2log, rank, logcnt

    # Ensure we work in a floating point type, but prefer native precision (e.g. half)
    if weight.is_floating_point():
        dtype = weight.dtype
    else:
        dtype = torch.float32
        weight = weight.to(dtype)

    # Vectorized Greedy Selection
    divisors = torch.arange(1, num_redundant + 1, device=device, dtype=dtype)
    inv_divisors = torch.reciprocal(divisors)

    # Calculate marginal gains: weight * (1/divisors)
    # [L, N, 1] * [1, 1, R] -> [L, N, R]
    scores = weight.unsqueeze(-1) * inv_divisors

    # Flatten scores to select global top-k
    flat_scores = scores.flatten(1)

    # Select top K redundant slots across all experts
    # Operating on native precision (e.g. float16) reduces memory bandwidth
    topk_indices = torch.topk(flat_scores, k=num_redundant, dim=1).indices

    # Decode indices
    redundant_expert_ids = topk_indices // num_redundant
    redundant_ranks = (topk_indices % num_redundant) + 1

    # Fill redundant slots in output tensors
    phy2log[:, num_log:] = redundant_expert_ids
    rank[:, num_log:] = redundant_ranks

    # Update logcnt
    # Use expand to avoid allocating a ones tensor
    ones = torch.ones((), device=device, dtype=torch.int64).expand(redundant_expert_ids.shape)
    logcnt.scatter_add_(1, redundant_expert_ids, ones)

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing with efficient index mapping.
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: pack groups to nodes
    # Sum weights per group
    # Note: Using sum on float16 might risk overflow for very large counts,
    # but typical use cases handle this or inputs are scaled.
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    # group_perm: [layers, num_groups], Slot -> GroupID
    group_perm = balanced_packing(tokens_per_group, num_nodes)

    # Expand group permutation to item permutation (Slot -> LogicalItem)
    # mlog2log: [layers, num_logical_experts]
    mlog2log = (group_perm.unsqueeze(-1) * group_size +
                torch.arange(group_size, device=weight.device)).flatten(-2)

    # Step 2: construct redundant experts within nodes
    # Gather weights to packed order (mlog)
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # tokens_per_phy: [L*Nodes, PhyPerNode]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)

    # pphy2phy: [L*Nodes, PhyPerNode], Slot -> PhyIndex (index in phy2mlog)
    pphy2phy = balanced_packing(tokens_per_phy, num_gpus // num_nodes)

    # Map final physical slots (pphy) to mlog (Packed Logical)
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)

    # Adjust offsets for node blocks to recover global logical indices
    node_offsets = torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=weight.device,
    ).view(1, -1, 1)

    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets).flatten(-2)

    # Recover global logical IDs
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)

    # Recover logcnt in logical order
    # Scatter mlogcnt (which matches mlog2log order) back to logical order
    logcnt = torch.empty(num_layers, num_logical_experts, dtype=torch.int64, device=weight.device)
    logcnt.scatter_(1, mlog2log, mlogcnt.view(num_layers, -1))

    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.
    """
    num_layers, num_logical_experts = weight.shape

    # Removed global weight.float() cast to support native precision execution.

    # Delegate to hierarchical or flat rebalancing
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    # Dynamically determine max capacity to save memory and init time
    # .item() syncs, but savings on large tensors (N*R) outweigh overhead
    maxlogcnt = logcnt.max().item()

    # Create logical to physical map
    # Initialize with -1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    # Scatter physical indices to logical slots
    # Target index = logical_idx * max_capacity + rank_idx
    flat_indices = phy2log * maxlogcnt + phyrank

    src_values = torch.arange(num_replicas, dtype=torch.int64,
                              device=log2phy.device).expand(num_layers, -1)

    log2phy.view(num_layers, -1).scatter_(
        -1,
        flat_indices,
        src_values,
    )
    return phy2log, log2phy, logcnt
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_eplb(weight: torch.Tensor, num_replicas: int, num_groups: int,
             num_nodes: int, num_gpus: int):
    """Run the expert parallelism load balancer"""
    phy2log, log2phy, logcnt = rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts", "run_eplb"]


from pathlib import Path
from typing import Any, Dict

class Solution:
    """ShinkaEvolve-evolved solution for EPLB."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Extract code before Solution class definition
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}