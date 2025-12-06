# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan schedule using regret-based best-insertion construction
    with memoized cost evaluation, followed by multi-neighborhood local refinement,
    iterated local search, large-neighborhood destroy/repair, and elite path relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (number of seeds, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Memoization to avoid repeated cost computations for the same sequence
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # ----------------------------
    # Parameters (adaptive)
    # ----------------------------
    elite_size = max(3, min(6, 2 + num_seqs // 3))
    seed_elite_singletons = max(2, min(6, int(num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    # Candidate sampling sizes
    k_txn_sample = min(16, max(8, 2 + int(1.5 * num_seqs)))  # txns per insertion
    k_pos_sample = min(10, max(6, 2 + int(1.2 * num_seqs)))  # positions per insertion
    rem_all_threshold = 14                                    # when few remain, consider all txns

    # Regret selection probability (exploration vs exploitation)
    regret_prob = 0.65

    # Small local beam for construction and diversity settings
    local_beam_width = max(3, min(6, int(num_seqs)))
    high_regret_quota_ratio = 0.3
    diversity_suffix_k = 2
    endgame_all_pos_threshold = max(6, min(12, num_seqs))     # when few remain, try all positions

    # Local search parameters
    ls_adj_rounds_max = 2
    two_opt_trials = min(35, max(12, n // 3))
    reinsertion_pos_factor = 1.0  # multiplier for k_pos_sample

    # Iterated local search (ILS) / perturbation
    ils_rounds = max(2, min(5, 1 + num_seqs // 4))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # Large Neighborhood Search (LNS)
    lns_iters = max(2, min(6, 2 + num_seqs // 3))
    destroy_frac_range = (0.08, 0.18)

    # ----------------------------
    # Helpers
    # ----------------------------
    def position_samples(seq_len, focus_idx=None, k_positions=None):
        """Sample insertion positions; include anchors and positions near focus_idx if provided.
        When the sequence is small, evaluate all positions for accuracy."""
        if seq_len <= 1:
            return [0, seq_len]
        k = k_positions if k_positions is not None else k_pos_sample
        # For small sequences, check all positions deterministically
        if seq_len <= 12:
            return list(range(seq_len + 1))
        pos_set = {0, seq_len, seq_len // 2}
        # Bias positions near the focus index (if provided)
        if focus_idx is not None:
            for d in [-3, -2, -1, 0, 1, 2, 3]:
                p = focus_idx + d
                if 0 <= p <= seq_len:
                    pos_set.add(p)
        # Add random internal positions
        for _ in range(min(k, seq_len + 1)):
            pos_set.add(random.randint(0, seq_len))
        return sorted(pos_set)

    def evaluate_best_two_positions(base_seq, t, pos_list):
        """Return (best_cost, best_pos, second_best_cost) for inserting t into base_seq over pos_list."""
        best = (float('inf'), None)
        second = float('inf')
        for p in pos_list:
            cand = base_seq[:p] + [t] + base_seq[p:]
            c = seq_cost(cand)
            if c < best[0]:
                second = best[0]
                best = (c, p)
            elif c < second:
                second = c
        return best[0], best[1], second

    # Memoized best-two insertion per (sequence, txn, use_all_pos) with deterministic stratified positions
    best_two_cache = {}
    def best_two_insertion(base_seq, t, use_all_pos=False, focus_idx=None, k_positions=None):
        """
        Compute (best_cost, best_pos, second_best_cost) for inserting txn t into base_seq.
        - If use_all_pos or seq small, evaluate all positions exactly.
        - Else, deterministically sample anchors + a few interior positions (seeded by sequence suffix and t).
        Results are cached by (tuple(base_seq), t, use_all_pos) for reuse across beam and LNS phases.
        """
        key = (tuple(base_seq), t, bool(use_all_pos))
        if key in best_two_cache:
            return best_two_cache[key]
        L = len(base_seq)
        if use_all_pos or L <= 12:
            pos_list = list(range(L + 1))
        else:
            # Deterministic sampling: anchors + near focus + seeded interior
            pos_set = {0, L, L // 2, (L * 1) // 4, (L * 3) // 4}
            if focus_idx is not None:
                for d in (-3, -2, -1, 0, 1, 2, 3):
                    p = focus_idx + d
                    if 0 <= p <= L:
                        pos_set.add(p)
            cap = k_positions if k_positions is not None else k_pos_sample
            seed = (tuple(base_seq[-min(10, L):]), t, L)
            rng = random.Random(hash(seed) & 0xffffffff)
            for _ in range(min(cap, L + 1)):
                pos_set.add(rng.randint(0, L))
            pos_list = sorted(pos_set)
        res = evaluate_best_two_positions(base_seq, t, pos_list)
        best_two_cache[key] = res
        return res

    # ----------------------------
    # Endgame exact completion (branch-and-bound over last K txns)
    # ----------------------------
    endgame_enum_K = max(6, min(9, endgame_all_pos_threshold + 3))
    endgame_memo = {}
    def endgame_optimal_completion(prefix_seq, rem_set):
        """
        Given a prefix sequence and a small remaining set of txns (|rem_set| <= K),
        enumerate txn orders with best-position insertion to find the minimal-cost completion.
        Uses branch-and-bound with the current best bound and partial-cost pruning.
        Returns (best_cost, best_seq).
        """
        key0 = (tuple(prefix_seq), frozenset(rem_set))
        if key0 in endgame_memo:
            return endgame_memo[key0]

        best_c = float('inf')
        best_s = None

        def dfs(seq, rem):
            nonlocal best_c, best_s
            # Prune by current partial cost
            c_prefix = seq_cost(seq)
            if c_prefix >= best_c:
                return
            if not rem:
                if c_prefix < best_c:
                    best_c = c_prefix
                    best_s = seq[:]
                return
            # Order candidates by high regret first, then by lower best insertion cost
            order = []
            for t in rem:
                b, p, s2 = best_two_insertion(seq, t, use_all_pos=True)
                regret = (s2 - b) if s2 < float('inf') else 0.0
                order.append((-regret, b, t, p))
            order.sort()
            for _, bcost, t, p in order:
                if bcost >= best_c:
                    continue
                new_seq = seq[:p] + [t] + seq[p:]
                new_rem = rem.copy()
                new_rem.remove(t)
                dfs(new_seq, new_rem)

        dfs(prefix_seq[:], set(rem_set))
        if best_s is None:
            # Fallback: deterministic append to avoid None
            seq_complete = prefix_seq[:] + sorted(list(rem_set))
            best_c = seq_cost(seq_complete)
            best_s = seq_complete
        endgame_memo[key0] = (best_c, best_s)
        return best_c, best_s

    def regret_insertion_build(seed_t=None):
        """Construct a schedule using regret-guided insertion with a small beam and diversity."""
        if seed_t is None:
            seed_t = random.randint(0, n - 1)

        # Beam holds tuples: (seq, rem_set, cost)
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        # Add a second element by prepend/append across beam
        expansions = []
        for seq, rem, _ in beam:
            if not rem:
                expansions.append((seq, rem, seq_cost(seq), 0.0))
                continue
            cand_txns = list(rem)
            if len(cand_txns) > k_txn_sample:
                cand_txns = random.sample(cand_txns, k_txn_sample)
            for t in cand_txns:
                # Try prepend and append for early shaping
                for new_seq in ([t] + seq, seq + [t]):
                    c = seq_cost(new_seq)
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    expansions.append((new_seq, new_rem, c, 0.0))
        # Select initial beam by cost-diversity
        expansions.sort(key=lambda x: x[2])
        next_beam = []
        seen_sig = set()
        for s, r, c, _ in expansions:
            sig = tuple(s[-2:]) if len(s) >= 2 else tuple(s)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            next_beam.append((s, r, c))
            if len(next_beam) >= local_beam_width:
                break
        if next_beam:
            beam = next_beam

        while True:
            # If all states are complete, stop
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            seen_seqs = set()
            for seq, rem, _ in beam:
                if not rem:
                    expansions.append((seq, rem, seq_cost(seq), 0.0, None))
                    continue

                # Candidate transactions
                if len(rem) <= rem_all_threshold:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                # Determine position policy (all vs sampled)
                use_all_pos = (len(rem) <= endgame_all_pos_threshold) or (len(seq) <= 18)
                pos_list_all = list(range(len(seq) + 1)) if use_all_pos else position_samples(len(seq))

                # For each candidate transaction, evaluate best and second-best positions
                for t in cand_txns:
                    best_c, best_p, second_c = best_two_insertion(seq, t, use_all_pos=use_all_pos)
                    new_seq = seq[:best_p] + [t] + seq[best_p:]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    regret = second_c - best_c if second_c < float('inf') else 0.0
                    key = tuple(new_seq)
                    if key in seen_seqs:
                        continue
                    seen_seqs.add(key)
                    expansions.append((new_seq, new_rem, best_c, regret, t))

            if not expansions:
                break

            # Rank expansions
            by_cost = sorted(expansions, key=lambda x: (x[2], -x[3]))
            by_regret = sorted(expansions, key=lambda x: (-x[3], x[2]))

            # Select next beam with diversity and regret quota
            base_quota = max(1, int(local_beam_width * (1.0 - high_regret_quota_ratio)))
            regret_quota = max(0, local_beam_width - base_quota)
            next_beam = []
            seen_sig = set()

            # Helper to add with diversity (suffix-k signature)
            def try_add(entry):
                s, r, c, reg, _t = entry
                sig = tuple(s[-diversity_suffix_k:]) if len(s) >= diversity_suffix_k else tuple(s)
                if sig in seen_sig:
                    return False
                seen_sig.add(sig)
                next_beam.append((s, r, c))
                return True

            # Fill cost-first
            i = 0
            while len(next_beam) < base_quota and i < len(by_cost):
                try_add(by_cost[i])
                i += 1
            # Fill regret quota
            j = 0
            while len(next_beam) < base_quota + regret_quota and j < len(by_regret):
                try_add(by_regret[j])
                j += 1
            # If still short, fill by cost
            k = 0
            while len(next_beam) < local_beam_width and k < len(by_cost):
                try_add(by_cost[k])
                k += 1

            if not next_beam:
                # Fallback: take top-k by cost
                next_beam = [(s, r, c) for s, r, c, _, _ in by_cost[:local_beam_width]]

            beam = next_beam

        # Choose best complete (use exact endgame when few remain)
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if not rem:
                # Already complete
                c = cost
                s = seq
            else:
                if len(rem) <= endgame_enum_K:
                    c, s = endgame_optimal_completion(seq, rem)
                else:
                    # append rest deterministically then evaluate
                    seq_complete = seq[:] + sorted(list(rem))
                    c = seq_cost(seq_complete)
                    s = seq_complete
            if c < best_cost:
                best_cost = c
                best_seq = s

        return best_seq

    # Local refinement: VND with adjacent swaps, Or-opt (1,2,3), and sampled 2-opt
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)
        improved_outer = True

        adj_rounds = 0

        def two_opt_sample_pass(cur_seq, cur_cost):
            tried = 0
            while tried < two_opt_trials:
                i = random.randint(0, n - 2)
                j = random.randint(i + 2, n - 1) if i + 2 < n else None
                tried += 1
                if j is None:
                    continue
                cand = cur_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    return True, c, cand
            return False, cur_cost, cur_seq

        def try_reinsertion(cur_seq, cur_cost):
            k_positions = max(6, int(reinsertion_pos_factor * k_pos_sample))
            for i in range(n):
                item = cur_seq[i]
                base = cur_seq[:i] + cur_seq[i + 1:]
                # Endgame: exhaustive positions when sequence small
                if len(base) <= 20:
                    positions = list(range(len(base) + 1))
                else:
                    positions = position_samples(len(base), focus_idx=i, k_positions=k_positions)
                for p in positions:
                    cand = base[:p] + [item] + base[p:]
                    if cand == cur_seq:
                        continue
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return True, c, cand
            return False, cur_cost, cur_seq

        def try_or_opt(cur_seq, cur_cost, k):
            L = len(cur_seq)
            for i in range(L - k + 1):
                block = cur_seq[i:i + k]
                base = cur_seq[:i] + cur_seq[i + k:]
                # Endgame: exhaustive positions when small; otherwise sample near i
                if len(base) <= 20:
                    positions = list(range(len(base) + 1))
                else:
                    positions = position_samples(len(base), focus_idx=i, k_positions=k_pos_sample)
                for p in positions:
                    # Skip no-op reinsertion at same position
                    if p == i:
                        continue
                    cand = base[:p] + block + base[p:]
                    if cand == cur_seq:
                        continue
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return True, c, cand
            return False, cur_cost, cur_seq

        while improved_outer:
            improved_outer = False

            # Adjacent swap pass (limited rounds)
            if adj_rounds < ls_adj_rounds_max:
                improved = True
                while improved:
                    improved = False
                    for i in range(n - 1):
                        cand = best_seq[:]
                        cand[i], cand[i + 1] = cand[i + 1], cand[i]
                        c = seq_cost(cand)
                        if c < best_cost:
                            best_cost = c
                            best_seq = cand
                            improved = True
                            improved_outer = True
                            break
                adj_rounds += 1
                if improved_outer:
                    continue

            # Or-opt k=1 (single reinsertion)
            changed, new_cost, new_seq = try_reinsertion(best_seq, best_cost)
            if changed:
                best_seq, best_cost = new_seq, new_cost
                improved_outer = True
                continue

            # Or-opt k=2
            changed, new_cost, new_seq = try_or_opt(best_seq, best_cost, 2)
            if changed:
                best_seq, best_cost = new_seq, new_cost
                improved_outer = True
                continue

            # Or-opt k=3
            changed, new_cost, new_seq = try_or_opt(best_seq, best_cost, 3)
            if changed:
                best_seq, best_cost = new_seq, new_cost
                improved_outer = True
                continue

            # Sampled 2-opt pass (non-adjacent swaps)
            changed, new_cost, new_seq = two_opt_sample_pass(best_seq, best_cost)
            if changed:
                best_seq = new_seq
                best_cost = new_cost
                improved_outer = True

        return best_cost, best_seq

    # Perturbation methods for ILS
    def perturb(seq):
        s = seq[:]
        mode = random.random()
        if mode < 0.5:
            # Random swaps
            for _ in range(perturb_swap_count):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                s[i], s[j] = s[j], s[i]
        else:
            # Block shuffle
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                del s[start:start + perturb_block_len]
                insert_pos = random.randint(0, len(s))
                s = s[:insert_pos] + block + s[insert_pos:]
        return s

    # LNS destroy-and-repair
    def lns_attempt(seq):
        cur = seq[:]
        # Choose destroy size
        frac = random.uniform(*destroy_frac_range)
        m = max(4, min(n // 2, int(frac * n)))
        # Mix of random removal and contiguous block removal
        if random.random() < 0.5:
            remove_idxs = sorted(random.sample(range(n), m))
        else:
            start = random.randint(0, n - m)
            remove_idxs = list(range(start, start + m))
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in set(remove_idxs)]
        # Repair using regret insertion
        seq_rep = remaining[:]
        rem_set = removed[:]
        # Exact endgame rebuild when small
        if len(rem_set) <= endgame_enum_K:
            c_end, s_end = endgame_optimal_completion(seq_rep, set(rem_set))
            c_rep, s_rep = local_refine(s_end)
            return c_rep, s_rep

        while rem_set:
            if len(rem_set) > k_txn_sample:
                cand_txns = random.sample(rem_set, k_txn_sample)
            else:
                cand_txns = rem_set[:]

            best_overall = (float('inf'), None, None)  # cost, txn, new_seq
            best_by_regret = (float('-inf'), None, None)

            for t in cand_txns:
                use_all = (len(rem_set) <= endgame_all_pos_threshold) or (len(seq_rep) <= 18)
                best_c, best_p, second_c = best_two_insertion(seq_rep, t, use_all_pos=use_all)
                # Track pure best
                if best_c < best_overall[0]:
                    new_seq = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                    best_overall = (best_c, t, new_seq)
                # Track regret
                regret = second_c - best_c if second_c < float('inf') else 0.0
                if regret > best_by_regret[0]:
                    new_seq_r = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                    best_by_regret = (regret, t, new_seq_r)

            pick_regret = (random.random() < regret_prob)
            chosen = best_by_regret if pick_regret and best_by_regret[1] is not None else best_overall
            if chosen[1] is None:
                t = random.choice(rem_set)
                seq_rep = seq_rep + [t]
                rem_set.remove(t)
            else:
                seq_rep = chosen[2]
                rem_set.remove(chosen[1])

        c_rep, s_rep = local_refine(seq_rep)
        return c_rep, s_rep

    # Path Relinking: move current solution towards target elite by aligning positions
    def path_relink(source_seq, target_seq, max_moves=12):
        pos_in_target = {t: i for i, t in enumerate(target_seq)}
        s = source_seq[:]
        best_c = seq_cost(s)
        best_s = s[:]
        moves = 0
        # Choose items with largest position displacement
        displacement = [(abs(i - pos_in_target[s[i]]), i) for i in range(n)]
        displacement.sort(reverse=True)
        for _, idx in displacement:
            if moves >= max_moves:
                break
            item = s[idx]
            desired = pos_in_target[item]
            if desired == idx:
                continue
            base = s[:idx] + s[idx + 1:]
            # Insert at desired (bounded within current length)
            desired = max(0, min(desired, len(base)))
            cand = base[:desired] + [item] + base[desired:]
            c = seq_cost(cand)
            if c < best_c:
                best_c = c
                best_s = cand[:]
                s = cand
                moves += 1
        if best_c < float('inf'):
            return best_c, best_s
        return seq_cost(source_seq), source_seq

    # ----------------------------
    # Seeding
    # ----------------------------
    # Evaluate all singletons; pick elite seeds + some random distinct seeds
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and seed_random_additional > 0:
        extra = random.sample(remaining, min(seed_random_additional, len(remaining)))
        seed_txns.extend(extra)

    # Build schedules from seeds and keep elite pool
    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # keep unique sequences by suffix of length 2 to add diversity
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            if s1:
                sig = tuple(s1[-2:]) if len(s1) >= 2 else (s1[-1],)
            else:
                sig = (None,)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    for seed in seed_txns:
        seq0 = regret_insertion_build(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)

    # If no elite (shouldn't happen), fallback
    if not elite:
        base = all_txns[:]
        random.shuffle(base)
        elite = [(seq_cost(base), base)]

    best_overall_cost, best_overall_seq = elite[0]

    # ----------------------------
    # Iterated Local Search (ILS)
    # ----------------------------
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        pert = perturb(cur_seq)
        c2, s2 = local_refine(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # ----------------------------
    # LNS destroy-and-repair
    # ----------------------------
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)

    # ----------------------------
    # Path Relinking among elites
    # ----------------------------
    if len(elite) >= 2:
        base_cost, base_seq = best_overall_cost, best_overall_seq
        # Try relinking with a few elite partners
        partners = elite[1:min(len(elite), elite_size)]
        for c_t, s_t in partners:
            pr_c, pr_s = path_relink(base_seq, s_t, max_moves=max(8, min(12, n // 8)))
            if pr_c < best_overall_cost:
                best_overall_cost, best_overall_seq = pr_c, pr_s

    return best_overall_cost, best_overall_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)

    total_makespan = cost1 + cost2 + cost3
    schedules = [schedule1, schedule2, schedule3]
    execution_time = time.time() - start_time

    return total_makespan, schedules, execution_time


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling():
    """Run the transaction scheduling algorithm for all workloads"""
    total_makespan, schedules, execution_time = get_random_costs()
    return total_makespan, schedules, execution_time


if __name__ == "__main__":
    total_makespan, schedules, execution_time = run_scheduling()
    print(f"Total makespan: {total_makespan}, Execution time: {execution_time:.4f}s")
    print(f"Individual workload costs: {[workload.get_opt_seq_cost(schedule) for workload, schedule in zip([Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)], schedules)]}")


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Dict, Any

class Solution:
    """Shinka solution for txn_scheduling."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code (everything before the Solution class)
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}