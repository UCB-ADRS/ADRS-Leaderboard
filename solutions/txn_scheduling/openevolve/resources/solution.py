import time
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Improved: multi-start beam-guided greedy + small local search.
    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # EVOLVE-BLOCK-START
    # Global cache for cost evaluations across the entire search
    cache = {}

    def cost_of(seq):
        k = tuple(seq)
        c = cache.get(k)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cache[k] = c
        return c

    n = workload.num_txns
    if n == 0:
        return 0, []
    all_txns = list(range(n))

    # Pairwise precedence scoring: prefer orders that reduce 2-txn makespan
    def pairwise_scores():
        pref = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                cij = cost_of([i, j])
                cji = cost_of([j, i])
                margin = cji - cij  # >0 => i before j is better
                pref[i][j] = margin
                pref[j][i] = -margin
        scores = [0]*n
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j and pref[i][j] > 0:
                    s += pref[i][j]
            scores[i] = s
        return scores

    def greedy(sample_k=10):
        # Best-position insertion greedy seeded from low-cost singletons
        singletons = [(cost_of([i]), i) for i in all_txns]
        singletons.sort(key=lambda x: x[0])
        pool = [i for _, i in singletons[:min(4, n)]] if n > 0 else [0]
        start = random.choice(pool)
        seq = [start]
        remaining = [t for t in all_txns if t != start]
        while remaining:
            cand_txns = remaining if len(remaining) <= sample_k else random.sample(remaining, sample_k)
            best_c, best_t, best_pos = float("inf"), None, None
            m = len(seq)
            for t in cand_txns:
                for pos in range(m + 1):
                    cand_seq = seq[:pos] + [t] + seq[pos:]
                    c = cost_of(cand_seq)
                    if c < best_c or (c == best_c and random.random() < 0.5):
                        best_c, best_t, best_pos = c, t, pos
            if best_t is None:
                best_t = remaining[0]
                best_pos = len(seq)
            seq.insert(best_pos, best_t)
            remaining.remove(best_t)
        return cost_of(seq), seq

    def beam(width=3, sample_k=10):
        # Beam search seeded by best singletons; expand using sampled candidates
        starts = [i for _, i in sorted(((cost_of([i]), i) for i in all_txns), key=lambda x: x[0])[:min(width, n)]]
        if not starts:
            return 0, []
        beams = [([s], cost_of([s])) for s in starts]
        while beams and len(beams[0][0]) < n:
            candidates, seen = [], set()
            for seq, _ in beams:
                rem = [x for x in all_txns if x not in seq]
                if not rem:
                    candidates.append((seq, cost_of(seq)))
                    continue
                cand = rem if len(rem) <= sample_k else random.sample(rem, sample_k)
                for t in cand:
                    new_seq = seq + [t]
                    tp = tuple(new_seq)
                    if tp in seen:
                        continue
                    seen.add(tp)
                    candidates.append((new_seq, cost_of(new_seq)))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])
            beams = candidates[:width]
        best_seq, best_cost = min(beams, key=lambda x: x[1])
        return best_cost, best_seq

    def borda_seed(scores):
        # Order by pairwise score, then insert each at best position
        order = sorted(all_txns, key=lambda t: -scores[t])
        if not order:
            return greedy()
        seq = [order[0]]
        for t in order[1:]:
            m = len(seq)
            best_c, best_seq = float("inf"), None
            for pos in range(m + 1):
                cand = seq[:pos] + [t] + seq[pos:]
                c = cost_of(cand)
                if c < best_c:
                    best_c, best_seq = c, cand
            seq = best_seq
        return cost_of(seq), seq

    def improve_descent(seq, rounds=2):
        # Deterministic local descent: adjacent swaps + full relocate moves
        best_seq = seq[:]
        best_c = cost_of(best_seq)
        for _ in range(rounds):
            m = len(best_seq)
            improved = False
            # Adjacent swap pass with slight backtracking
            i = 0
            while i < m - 1:
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = cost_of(cand)
                if c < best_c:
                    best_c, best_seq = c, cand
                    improved = True
                    if i > 0:
                        i -= 1
                    continue
                i += 1
            # Full relocate pass (all i -> j)
            m = len(best_seq)
            for i in range(m):
                for j in range(m):
                    if j == i:
                        continue
                    cand = best_seq[:]
                    v = cand.pop(i)
                    cand.insert(j, v)
                    c = cost_of(cand)
                    if c < best_c:
                        best_c, best_seq = c, cand
                        improved = True
            if not improved:
                break
        return best_c, best_seq

    def local_improve(seq, budget=50):
        # Light randomized neighborhood search (swap/move)
        best_s = seq[:]
        best_c = cost_of(best_s)
        m = len(best_s)
        for _ in range(budget if m > 1 else 0):
            i = random.randrange(m)
            j = random.randrange(m)
            if i == j:
                continue
            cand = best_s[:]
            if random.random() < 0.5:
                cand[i], cand[j] = cand[j], cand[i]
            else:
                v = cand.pop(i)
                cand.insert(j, v)
            c = cost_of(cand)
            if c < best_c:
                best_c, best_s = c, cand
        return best_c, best_s

    best_cost = float("inf")
    best_seq = None
    attempts = max(6, int(num_seqs) if isinstance(num_seqs, int) else 6)

    # Build a strong heuristic seed once using pairwise preferences
    pw_scores = pairwise_scores() if n > 1 else [0]*n

    for i in range(attempts):
        mode = i % 3
        if mode == 0:
            c, s = greedy(sample_k=min(12, n))
        elif mode == 1:
            c, s = beam(width=min(4, n), sample_k=min(10, n))
        else:
            c, s = borda_seed(pw_scores)
        # Quick deterministic descent polish
        c, s = improve_descent(s, rounds=1)
        if c < best_cost:
            best_cost, best_seq = c, s

    # Final local refinement with a small random budget scaled by n
    c2, s2 = local_improve(best_seq, budget=max(40, min(70, 15 + n)))
    if c2 < best_cost:
        best_cost, best_seq = c2, s2

    # Final deterministic polish
    c3, s3 = improve_descent(best_seq, rounds=1)
    if c3 < best_cost:
        best_cost, best_seq = c3, s3

    return best_cost, best_seq
    # EVOLVE-BLOCK-END

def get_random_costs():
    start_time = time.time()
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3], time.time() - start_time


if __name__ == "__main__":
    makespan, schedule, time = get_random_costs()
    print(f"Makespan: {makespan}, Time: {time}")


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Dict, Any

class Solution:
    """OpenEvolve solution for txn_scheduling."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code (everything before the Solution class)
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}
