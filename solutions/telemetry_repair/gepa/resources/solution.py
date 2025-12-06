# Your complete improved program here - must be a full, runnable Python program
from typing import Dict, Any, Tuple, List, Optional
import math

# ============================================================
# Overview (brief, no internal chain-of-thought)
# ============================================================
# This implementation repairs telemetry by:
#  - Pairing interfaces into bidirectional links via local/remote metadata.
#  - Enforcing link symmetry (A.tx ≈ B.rx and B.tx ≈ A.rx) with a small tolerance.
#  - Using redundancy across links to compute per-router trust and soft directional bias hints.
#  - Detecting likely counter tx/rx swaps before repair.
#  - Applying multiplicative fusion (geometric mean) or selecting the more credible side, guided by trust and bias.
#  - Dynamically reconciling interface_status with observed activity and ensuring pair-consistent status.
#  - Producing calibrated confidences (0-1) that increase with corroboration and decrease with uncertainty.
#
# We accept an external 'topology' parameter for API compatibility, but we prioritize
# per-interface metadata (local_router/remote_router), which is more robust in practice.

# ============================================================
# Utility helpers
# ============================================================

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def rel_diff(a: float, b: float, base_floor: float = 1.0) -> float:
    """
    Relative difference based on the larger magnitude and a protective floor.
    This keeps jitter at low traffic from exploding.
    """
    denom = max(abs(a), abs(b), base_floor)
    return abs(a - b) / denom

def approx_equal(a: float, b: float, rel_tol: float = 0.02, abs_tol: float = 0.0) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))

def near_zero(x: float, ztol: float = 1e-6) -> bool:
    return abs(x) <= ztol

def _normalize_status(s: Any) -> str:
    """
    Normalize interface status to 'up' or 'down'. Treat unknown/mixed admin states conservatively.
    """
    ss = str(s).strip().lower()
    if ss in ('up', 'up/up', 'active'):
        return 'up'
    if 'down' in ss or ss in ('admin_down', 'shutdown', 'inactive'):
        return 'down'
    # Fallback: assume up unless it clearly says down
    return 'up'

def _activity_detected(values: List[Optional[float]], thres: float = 0.05) -> bool:
    """
    Detect meaningful traffic activity on a link using a small absolute threshold (Mbps).
    Avoid interpreting minute noise as real activity.
    """
    for v in values:
        if v is not None and safe_float(v, 0.0) > thres:
            return True
    return False

def _safe_log_ratio(num: float, den: float, floor: float = 1.0) -> float:
    """
    Stable log ratio for bias estimation. Uses a positive floor to limit
    extreme values in low-traffic situations.
    """
    n = max(num, floor)
    d = max(den, floor)
    return math.log(n / d)

def _weighted_geometric_mean(values: List[float], weights: List[float]) -> float:
    """
    Compute a weighted geometric mean in a numerically stable way.
    """
    wsum = sum(max(0.0, w) for w in weights)
    if wsum <= 0.0:
        # Fallback to simple geometric mean
        prod = 1.0
        n = 0
        for v in values:
            vv = max(v, 1e-9)
            prod *= vv
            n += 1
        return prod ** (1.0 / max(1, n))
    acc = 0.0
    for v, w in zip(values, weights):
        acc += max(0.0, w) * math.log(max(v, 1e-9))
    return math.exp(acc / wsum)

# ============================================================
# Required compatibility helpers
# ============================================================

def convert_interfaces_to_csv_row(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compatibility conversion used by some evaluators. Produces a flat row-like dictionary
    with per-interface rates and simple per-router aggregates. This function does not
    influence the repair logic.

    Note: We keep nested dict values for clarity; evaluators that ignore this function
    won't be affected.
    """
    csv_row: Dict[str, Any] = {
        'timestamp': '2024/01/01 12:00 UTC',
        'telemetry_perturbed_type': 'unknown',
        'input_perturbed_type': 'unknown',
        'true_detect_inconsistent': False
        # Per-interface columns follow below
    }

    router_termination: Dict[str, float] = {}
    router_origination: Dict[str, float] = {}

    for interface_id, data in telemetry.items():
        if '_to_' not in interface_id:
            continue
        parts = interface_id.split('_to_')
        if len(parts) != 2:
            continue
        source, dest = parts

        egress_col = f"low_{source}_egress_to_{dest}"
        tx_rate = safe_float(data.get('tx_rate', 0.0), 0.0)
        csv_row[egress_col] = {
            'ground_truth': tx_rate,
            'perturbed': tx_rate,
            'corrected': None,
            'confidence': None
        }

        ingress_col = f"low_{dest}_ingress_from_{source}"
        rx_rate = safe_float(data.get('rx_rate', 0.0), 0.0)
        csv_row[ingress_col] = {
            'ground_truth': rx_rate,
            'perturbed': rx_rate,
            'corrected': None,
            'confidence': None
        }

        router_origination[source] = router_origination.get(source, 0.0) + tx_rate
        router_termination[dest] = router_termination.get(dest, 0.0) + rx_rate

    all_routers = set(router_termination.keys()) | set(router_origination.keys())
    for router in all_routers:
        term_rate = router_termination.get(router, 0.0)
        csv_row[f"low_{router}_termination"] = {
            'ground_truth': term_rate,
            'perturbed': term_rate,
            'corrected': None,
            'confidence': None
        }

        orig_rate = router_origination.get(router, 0.0)
        csv_row[f"low_{router}_origination"] = {
            'ground_truth': orig_rate,
            'perturbed': orig_rate,
            'corrected': None,
            'confidence': None
        }

    return csv_row


def build_topology_dict(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a simple undirected topology graph from per-interface metadata.

    Note: The core repair path relies on interface-level metadata (local_router/remote_router)
    rather than this summarized view because summarized maps can be stale in production.
    """
    routers = set()
    connections = set()

    for _, data in telemetry.items():
        a = data.get('local_router')
        b = data.get('remote_router')
        if a and b:
            routers.add(a)
            routers.add(b)
            connections.add(tuple(sorted((a, b))))

    router_list = sorted(list(routers))
    router_to_id = {r: i for i, r in enumerate(router_list)}

    topology = {
        'nodes': [{'id': router_to_id[r], 'name': r} for r in router_list],
        'links': [{'source': router_to_id[a], 'target': router_to_id[b]} for a, b in connections]
    }
    return topology

# ============================================================
# Pair construction helper
# ============================================================

def _build_pairs(telemetry: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Optional[str]]]:
    """
    Group interfaces into unordered router pairs, keeping the directional interface IDs.
    """
    pairs: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    for iface_id, data in telemetry.items():
        lr = data.get('local_router')
        rr = data.get('remote_router')
        if not lr or not rr or '_to_' not in iface_id:
            continue
        pair = tuple(sorted([str(lr), str(rr)]))
        if pair not in pairs:
            pairs[pair] = {'A': pair[0], 'B': pair[1], 'A_to_B': None, 'B_to_A': None}
        if lr == pair[0] and rr == pair[1]:
            pairs[pair]['A_to_B'] = iface_id
        elif lr == pair[1] and rr == pair[0]:
            pairs[pair]['B_to_A'] = iface_id
    return pairs

# ============================================================
# Router trust and balance hints
# ============================================================

def _compute_router_trust(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute a router trust score using:
      - Mean link symmetry residuals (A.tx vs B.rx and B.tx vs A.rx) across its links.
      - Coverage (number of corroborated directions).
      - Soft balance check (sum local tx vs sum peer rx where both sides present).

    Lower residuals and higher coverage -> higher trust.
    Trust is mapped into [0.5, 1.0].
    """
    pair_map = _build_pairs(telemetry)

    per_router_errs: Dict[str, List[float]] = {}
    per_router_tx_sum: Dict[str, float] = {}
    per_router_peer_rx_sum: Dict[str, float] = {}
    per_router_cov: Dict[str, int] = {}

    for _, info in pair_map.items():
        A, B = info['A'], info['B']
        ab = info['A_to_B']
        ba = info['B_to_A']

        if ab is not None and ba is not None:
            a_tx = safe_float(telemetry[ab].get('tx_rate', 0.0), 0.0)
            b_rx = safe_float(telemetry[ba].get('rx_rate', 0.0), 0.0)
            b_tx = safe_float(telemetry[ba].get('tx_rate', 0.0), 0.0)
            a_rx = safe_float(telemetry[ab].get('rx_rate', 0.0), 0.0)

            a_stat = _normalize_status(telemetry[ab].get('interface_status', 'up'))
            b_stat = _normalize_status(telemetry[ba].get('interface_status', 'up'))

            # Only use up/up links as valid evidence for trust
            if a_stat == 'up' and b_stat == 'up':
                d1 = rel_diff(a_tx, b_rx, base_floor=1.0)
                d2 = rel_diff(b_tx, a_rx, base_floor=1.0)
                per_router_errs.setdefault(A, []).append(d1)
                per_router_errs.setdefault(B, []).append(d2)
                per_router_cov[A] = per_router_cov.get(A, 0) + 1
                per_router_cov[B] = per_router_cov.get(B, 0) + 1
                per_router_tx_sum[A] = per_router_tx_sum.get(A, 0.0) + a_tx
                per_router_peer_rx_sum[A] = per_router_peer_rx_sum.get(A, 0.0) + b_rx
                per_router_tx_sum[B] = per_router_tx_sum.get(B, 0.0) + b_tx
                per_router_peer_rx_sum[B] = per_router_peer_rx_sum.get(B, 0.0) + a_rx

    trust: Dict[str, float] = {}
    routers = set(list(per_router_cov.keys()) + list(per_router_errs.keys()))
    for r in routers:
        errs = per_router_errs.get(r, [])
        cov = per_router_cov.get(r, 0)
        if len(errs) == 0:
            mean_err = 0.15  # neutral prior if no corroboration
        else:
            mean_err = sum(errs) / len(errs)

        # Soft balance error based on tx sum vs peer rx sum
        txs = per_router_tx_sum.get(r, 0.0)
        prx = per_router_peer_rx_sum.get(r, 0.0)
        bal_err = rel_diff(txs, prx, base_floor=1.0)

        # Map errors into [0,1]: lower error -> higher score
        sym_score = 1.0 / (1.0 + mean_err)  # [~0,1]
        bal_score = 1.0 / (1.0 + bal_err)   # [~0,1]

        # Coverage boost: more corroborated directions -> more trust
        cov_boost = clamp(0.8 + 0.04 * min(cov, 6), 0.8, 1.04)

        # Combine and map into [0.5, 1.0]
        raw = (0.65 * sym_score + 0.35 * bal_score) * cov_boost
        trust[r] = clamp(0.5 + 0.5 * clamp(raw, 0.0, 1.2), 0.5, 1.0)

    return trust


def _compute_router_balance_hints(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Balance hints capture persistent directional bias at each router by comparing:
      - Sum of peer_rx vs my_tx (outgoing bias)
      - Sum of peer_tx vs my_rx (incoming bias)
    We aggregate log-ratios across corroborated up/up links to produce a soft hint.

    Returned per-router fields:
      - out_bias: exp(mean log(peer_rx / my_tx))   (>1 => my_tx tends to be lower than peer_rx)
      - in_bias:  exp(mean log(peer_tx / my_rx))   (>1 => my_rx tends to be lower than peer_tx)
      - strength: number of corroborated comparisons used (for weighting)
    """
    pair_map = _build_pairs(telemetry)

    out_logs: Dict[str, List[float]] = {}
    in_logs: Dict[str, List[float]] = {}

    for _, info in pair_map.items():
        A, B = info['A'], info['B']
        ab = info['A_to_B']
        ba = info['B_to_A']
        if ab is None or ba is None:
            continue

        a_stat = _normalize_status(telemetry[ab].get('interface_status', 'up'))
        b_stat = _normalize_status(telemetry[ba].get('interface_status', 'up'))
        if a_stat != 'up' or b_stat != 'up':
            continue

        a_tx = safe_float(telemetry[ab].get('tx_rate', 0.0), 0.0)
        a_rx = safe_float(telemetry[ab].get('rx_rate', 0.0), 0.0)
        b_tx = safe_float(telemetry[ba].get('tx_rate', 0.0), 0.0)
        b_rx = safe_float(telemetry[ba].get('rx_rate', 0.0), 0.0)

        # Router A biases
        out_logs.setdefault(A, []).append(_safe_log_ratio(b_rx, a_tx, floor=1.0))
        in_logs.setdefault(A, []).append(_safe_log_ratio(b_tx, a_rx, floor=1.0))
        # Router B biases
        out_logs.setdefault(B, []).append(_safe_log_ratio(a_rx, b_tx, floor=1.0))
        in_logs.setdefault(B, []).append(_safe_log_ratio(a_tx, b_rx, floor=1.0))

    hints: Dict[str, Dict[str, float]] = {}
    routers = set(list(out_logs.keys()) + list(in_logs.keys()))
    for r in routers:
        ologs = out_logs.get(r, [])
        ilogs = in_logs.get(r, [])
        o_mean = sum(ologs) / len(ologs) if ologs else 0.0
        i_mean = sum(ilogs) / len(ilogs) if ilogs else 0.0
        out_bias = math.exp(o_mean)
        in_bias = math.exp(i_mean)
        strength = float(len(ologs) + len(ilogs)) / 2.0  # average samples per direction
        # Clamp bias to avoid extreme influence from limited evidence
        hints[r] = {
            'out_bias': clamp(out_bias, 0.7, 1.3),
            'in_bias': clamp(in_bias, 0.7, 1.3),
            'strength': strength
        }
    return hints

# ============================================================
# Pre-pass: detect and mark counter swaps
# ============================================================

def _detect_counter_swaps(telemetry: Dict[str, Dict[str, Any]],
                          REL_TOL: float = 0.02,
                          ACTIVITY_THRES: float = 0.05) -> Dict[str, bool]:
    """
    Detect likely tx/rx swapped counters on an interface by comparing total symmetry residuals
    across the two interfaces of a link. We only act when both directions exist and there is
    meaningful traffic.

    Returns a map iface_id -> True if we should swap tx and rx for that interface in repair.
    """
    pairs = _build_pairs(telemetry)
    swap_flags: Dict[str, bool] = {}

    for _, info in pairs.items():
        ab = info['A_to_B']
        ba = info['B_to_A']
        if not ab or not ba:
            continue

        a_stat = _normalize_status(telemetry[ab].get('interface_status', 'up'))
        b_stat = _normalize_status(telemetry[ba].get('interface_status', 'up'))
        # Only consider up/up links for swap inference
        if a_stat != 'up' or b_stat != 'up':
            continue

        a_tx = max(0.0, safe_float(telemetry[ab].get('tx_rate', 0.0), 0.0))
        a_rx = max(0.0, safe_float(telemetry[ab].get('rx_rate', 0.0), 0.0))
        b_tx = max(0.0, safe_float(telemetry[ba].get('tx_rate', 0.0), 0.0))
        b_rx = max(0.0, safe_float(telemetry[ba].get('rx_rate', 0.0), 0.0))

        # Require some activity to avoid spurious swaps on noise
        if not _activity_detected([a_tx, a_rx, b_tx, b_rx], thres=ACTIVITY_THRES):
            continue

        # Avoid swap inference if all values are tiny (noise domain)
        if max(a_tx, a_rx, b_tx, b_rx) < 0.5 * ACTIVITY_THRES:
            continue

        # Residuals under different hypotheses
        r_norm = rel_diff(a_tx, b_rx, base_floor=1.0) + rel_diff(b_tx, a_rx, base_floor=1.0)
        r_swap_a = rel_diff(a_rx, b_rx, base_floor=1.0) + rel_diff(b_tx, a_tx, base_floor=1.0)
        r_swap_b = rel_diff(a_tx, b_tx, base_floor=1.0) + rel_diff(b_rx, a_rx, base_floor=1.0)
        r_swap_both = rel_diff(a_rx, b_tx, base_floor=1.0) + rel_diff(b_rx, a_tx, base_floor=1.0)

        # Decide the best hypothesis
        cand_vals = {
            'norm': r_norm,
            'swap_a': r_swap_a,
            'swap_b': r_swap_b,
            'swap_both': r_swap_both
        }
        best_label = min(cand_vals, key=lambda k: cand_vals[k])
        best_val = cand_vals[best_label]

        # Strong improvement needed to trigger a swap
        improvement_ratio = best_val / (r_norm + 1e-9)

        # Avoid triggering swaps when normal is already reasonably small
        if r_norm <= 0.2:
            continue

        # Conservative thresholds to reduce false positives
        if best_label == 'swap_a' and improvement_ratio < 0.65 and best_val < 0.6:
            swap_flags[ab] = True
        elif best_label == 'swap_b' and improvement_ratio < 0.65 and best_val < 0.6:
            swap_flags[ba] = True
        elif best_label == 'swap_both' and improvement_ratio < 0.65 and best_val < 0.6:
            swap_flags[ab] = True
            swap_flags[ba] = True

    return swap_flags

# ============================================================
# Core repair logic
# ============================================================

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Hodor: Three-step validation and repair:
      1) Signal collection:
         - Group interfaces into bidirectional links using local/remote metadata.
      2) Signal hardening using redundancy:
         - Enforce symmetry A.tx ~= B.rx and B.tx ~= A.rx with ~2% tolerance.
         - Use router trust (redundancy across links) and balance hints to arbitrate disagreements.
         - Apply strong heuristics for zero-vs-nonzero conflicts and detect tx/rx swaps.
         - When fusing disagreeing readings, prefer multiplicative (geometric mean) fusion, weighted by credibility.
      3) Dynamic checking:
         - If either side claims 'down' but clear activity is present, correct to 'up'.
         - If any side claims 'down' without activity, or if all four counters are ~0,
           force the pair 'down' and zero traffic.

    We accept a 'topology' map for API compatibility but rely on interface-level metadata
    (local_router/remote_router) because summarized topologies can be stale in production.

    Returns a per-interface repair report:
      - tx_rate: (original, repaired, confidence)
      - rx_rate: (original, repaired, confidence)
      - interface_status: (original, repaired, confidence)
      - metadata fields propagated unchanged
    """
    # Thresholds
    REL_TOL = 0.02         # ~2% jitter tolerance for symmetry
    ZERO_TOL = 1e-6
    ACTIVITY_THRES = 0.05  # Mbps considered as meaningful traffic

    # Group into bidirectional links by unordered (routerA, routerB)
    pairs = _build_pairs(telemetry)

    # Detect potential per-interface tx/rx swaps for robust comparison
    swap_flags = _detect_counter_swaps(telemetry, REL_TOL=REL_TOL, ACTIVITY_THRES=ACTIVITY_THRES)

    # Compute router trust from observed discrepancies and soft conservation
    router_trust = _compute_router_trust(telemetry)
    # Compute router balance hints (directional bias)
    balance_hints = _compute_router_balance_hints(telemetry)

    pair_repairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for pair_key, info in pairs.items():
        A, B = info['A'], info['B']
        ab = info['A_to_B']
        ba = info['B_to_A']

        # Collect signals (non-negative), applying swap corrections if flagged
        if ab:
            a_stat = _normalize_status(telemetry[ab].get('interface_status', 'up'))
            a_tx_raw = safe_float(telemetry[ab].get('tx_rate'), 0.0)
            a_rx_raw = safe_float(telemetry[ab].get('rx_rate'), 0.0)
            if swap_flags.get(ab, False):
                a_tx_raw, a_rx_raw = a_rx_raw, a_tx_raw
            a_tx = max(0.0, a_tx_raw)
            a_rx = max(0.0, a_rx_raw)
        else:
            a_stat = 'up'
            a_tx = None
            a_rx = None

        if ba:
            b_stat = _normalize_status(telemetry[ba].get('interface_status', 'up'))
            b_tx_raw = safe_float(telemetry[ba].get('tx_rate'), 0.0)
            b_rx_raw = safe_float(telemetry[ba].get('rx_rate'), 0.0)
            if swap_flags.get(ba, False):
                b_tx_raw, b_rx_raw = b_rx_raw, b_tx_raw
            b_tx = max(0.0, b_tx_raw)
            b_rx = max(0.0, b_rx_raw)
        else:
            b_stat = 'up'
            b_tx = None
            b_rx = None

        # Detect activity on the pair
        activity = _activity_detected([a_tx, a_rx, b_tx, b_rx], thres=ACTIVITY_THRES)

        # Dynamic status pre-check
        force_down = False
        down_reason = ""
        status_override_up = False

        if (a_stat == 'down' or b_stat == 'down') and activity:
            # Counters show activity while one side reports down -> correct to up
            status_override_up = True
        elif a_stat == 'down' or b_stat == 'down':
            # No activity and some side reported down -> trust the down state
            force_down = True
            down_reason = "reported_down"
        else:
            # If both present and all counters are zero-ish, it's effectively down
            if ab and ba:
                if near_zero(a_tx or 0.0, ZERO_TOL) and near_zero(a_rx or 0.0, ZERO_TOL) and \
                   near_zero(b_tx or 0.0, ZERO_TOL) and near_zero(b_rx or 0.0, ZERO_TOL):
                    force_down = True
                    down_reason = "all_zero"
            else:
                # If only one interface is present and has no activity, softly mark as down
                if not activity:
                    force_down = True
                    down_reason = "single_zero"

        # Defaults for the pair repair decision
        repaired = {
            'status': 'up',
            'status_conf': 0.9,
            'AB_value': None,     # repaired A->B flow (None means keep original A.tx)
            'AB_conf': 0.6,
            'AB_changed': False,
            'BA_value': None,     # repaired B->A flow (None means keep original B.tx)
            'BA_conf': 0.6,
            'BA_changed': False
        }

        # Calibrate status and confidence
        if status_override_up:
            repaired['status'] = 'up'
            repaired['status_conf'] = 0.84
        elif force_down:
            repaired['status'] = 'down'
            if down_reason == "reported_down":
                if a_stat == 'down' and b_stat == 'down':
                    repaired['status_conf'] = 0.985
                else:
                    repaired['status_conf'] = 0.88
            elif down_reason == "all_zero":
                repaired['status_conf'] = 0.91
            else:  # "single_zero"
                repaired['status_conf'] = 0.86
        else:
            if a_stat == 'up' and b_stat == 'up':
                repaired['status'] = 'up'
                repaired['status_conf'] = 0.955
            else:
                repaired['status'] = 'up'
                repaired['status_conf'] = 0.78

        # If down, force counters to zero with confidence derived from status
        if repaired['status'] == 'down':
            repaired['AB_value'] = 0.0
            repaired['BA_value'] = 0.0
            repaired['AB_conf'] = clamp(repaired['status_conf'] - 0.05, 0.5, 1.0)
            repaired['BA_conf'] = clamp(repaired['status_conf'] - 0.05, 0.5, 1.0)
            repaired['AB_changed'] = True
            repaired['BA_changed'] = True
            pair_repairs[pair_key] = repaired
            continue

        # Otherwise, harden signals per direction using symmetry, trust, and balance hints
        trust_A = router_trust.get(A, 0.7)
        trust_B = router_trust.get(B, 0.7)

        hint_A = balance_hints.get(A, {'out_bias': 1.0, 'in_bias': 1.0, 'strength': 0.0})
        hint_B = balance_hints.get(B, {'out_bias': 1.0, 'in_bias': 1.0, 'strength': 0.0})

        # Whether we saw a prior swap on either side for this pair (slight confidence penalty)
        pair_has_swap = (ab in swap_flags) or (ba in swap_flags)

        # Helper for direction repair with heuristics and hints
        def repair_direction(x_tx: Optional[float], y_rx: Optional[float],
                             trust_x: float, trust_y: float,
                             out_bias_x: float, in_bias_y: float,
                             strength_x: float, strength_y: float,
                             has_peer: bool,
                             swap_involved: bool) -> Tuple[Optional[float], float, bool]:
            """
            Repair a single direction (X -> Y) using:
              - If both present and within tolerance: keep unchanged, high confidence.
              - If both present and mismatch:
                    - Zero-vs-nonzero: choose the active reading.
                    - Else compute a preference score from trust difference and balance hints.
                      When not decisive, fuse multiplicatively using a weighted geometric mean.
              - If only one signal present: keep it with confidence influenced by local trust
                and reduced if we cannot corroborate with the peer.

            Returns: (repaired_value_or_None_for_keep, confidence, changed_flag)
            """
            # Both available
            if x_tx is not None and y_rx is not None:
                xt = max(0.0, x_tx)
                yr = max(0.0, y_rx)
                if approx_equal(xt, yr, rel_tol=REL_TOL, abs_tol=0.0):
                    d = rel_diff(xt, yr, base_floor=1.0)
                    # Slightly higher confidence if volumes are moderate/high and routers are trusted
                    vol = max(xt, yr)
                    vol_bonus = 0.03 if vol >= 10.0 else (0.01 if vol >= 1.0 else 0.0)
                    trust_bonus = 0.02 * (0.5 * (trust_x + trust_y) - 0.5)
                    conf = clamp(0.93 + 0.05 * (1.0 - d) + vol_bonus + trust_bonus, 0.9, 0.99)
                    if swap_involved:
                        conf = clamp(conf - 0.01, 0.0, 1.0)
                    return (None, conf, False)

                # Strong zero-vs-nonzero heuristic
                if xt <= ACTIVITY_THRES and yr > 5 * ACTIVITY_THRES:
                    d = rel_diff(xt, yr, base_floor=1.0)
                    conf = clamp(0.83 + 0.06 * (1.0 - d), 0.76, 0.93)
                    if swap_involved:
                        conf = clamp(conf - 0.02, 0.0, 1.0)
                    return (yr, conf, True)
                if yr <= ACTIVITY_THRES and xt > 5 * ACTIVITY_THRES:
                    d = rel_diff(xt, yr, base_floor=1.0)
                    conf = clamp(0.83 + 0.06 * (1.0 - d), 0.76, 0.93)
                    if swap_involved:
                        conf = clamp(conf - 0.02, 0.0, 1.0)
                    return (xt, conf, True)

                # Preference from trust and bias hints
                # Positive preference -> favor peer_rx (Y)
                # Negative preference -> favor local_tx (X)
                log_out = math.log(max(out_bias_x, 1e-9))
                log_in = math.log(max(in_bias_y, 1e-9))
                # Bias gains softly limited by evidence strength
                strength_gain = clamp((strength_x + strength_y) / 6.0, 0.0, 1.0)
                k = 0.25 * strength_gain  # bias weight
                p = (trust_y - trust_x) + k * (log_out + log_in)

                # More decisive threshold to better arbitrate clear disagreements
                decisive = abs(p) > 0.10
                # Additional decisive condition when discrepancy is large and trust gap is notable
                if not decisive:
                    if rel_diff(xt, yr, base_floor=1.0) > 0.5 and abs(trust_y - trust_x) > 0.2:
                        decisive = True
                        p = (trust_y - trust_x)  # rely on trust gap

                if decisive:
                    chosen = yr if p > 0 else xt
                    trust_boost = max(trust_x, trust_y)
                    d = rel_diff(xt, yr, base_floor=1.0)
                    redundancy = 1.0  # both present
                    base_conf = 0.74 * redundancy
                    penalty = clamp(d / 0.6, 0.0, 0.6)
                    trust_factor = 0.6 + 0.4 * trust_boost
                    vol = max(xt, yr)
                    vol_uplift = 0.03 if vol >= 10.0 else (0.01 if vol >= 1.0 else 0.0)
                    swap_pen = 0.02 if swap_involved else 0.0
                    conf = clamp((base_conf * trust_factor) - (0.38 * penalty) + vol_uplift - swap_pen, 0.3, 0.95)
                    return (max(0.0, chosen), conf, True)

                # Otherwise do a weighted multiplicative fusion
                base_wx = 0.5 + 0.8 * (trust_x - 0.5)  # in [0.1..0.9]
                base_wy = 0.5 + 0.8 * (trust_y - 0.5)
                # Bias adjustments:
                # If out_bias_x > 1 (my_tx low vs peer_rx), nudge weight to peer_rx
                base_wy += 0.15 * clamp(log_out, -0.3, 0.3)
                # If in_bias_y > 1 (my_rx low vs peer_tx), reduce trust in y_rx
                base_wy += 0.15 * clamp(-log_in, -0.3, 0.3)
                # Ensure positivity
                wx = clamp(base_wx, 0.05, 1.5)
                wy = clamp(base_wy, 0.05, 1.5)

                fused = _weighted_geometric_mean([xt, yr], [wx, wy])

                d = rel_diff(xt, yr, base_floor=1.0)
                redundancy = 1.0
                base_conf = 0.76 * redundancy
                penalty = clamp(d / 0.7, 0.0, 0.6)
                trust_factor = 0.58 + 0.42 * (0.5 * (trust_x + trust_y))
                vol = max(xt, yr)
                vol_uplift = 0.03 if vol >= 10.0 else (0.01 if vol >= 1.0 else 0.0)
                conf = clamp((base_conf * trust_factor) - (0.35 * penalty) + vol_uplift, 0.35, 0.96)
                if swap_involved:
                    conf = clamp(conf - 0.01, 0.0, 1.0)
                return (max(0.0, fused), conf, True)

            # Only one present -> keep it, confidence based on the owner's trust
            if x_tx is not None:
                conf = clamp(0.6 + 0.35 * (trust_x - 0.5), 0.55, 0.87 if has_peer else 0.8)
                if swap_involved:
                    conf = clamp(conf - 0.01, 0.0, 1.0)
                return (None, conf, False)
            if y_rx is not None:
                conf = clamp(0.6 + 0.35 * (trust_y - 0.5), 0.55, 0.87 if has_peer else 0.8)
                if swap_involved:
                    conf = clamp(conf - 0.01, 0.0, 1.0)
                return (None, conf, False)

            # None present -> fallback to zero with low confidence
            return (0.0, 0.5, True)

        # A -> B direction uses A.tx and B.rx
        ab_val, ab_conf, ab_changed = repair_direction(
            a_tx if ab else None,
            b_rx if ba else None,
            trust_A, trust_B,
            hint_A.get('out_bias', 1.0), hint_B.get('in_bias', 1.0),
            hint_A.get('strength', 0.0), hint_B.get('strength', 0.0),
            has_peer=(ab is not None and ba is not None),
            swap_involved=pair_has_swap
        )
        repaired['AB_value'] = ab_val
        repaired['AB_conf'] = ab_conf
        repaired['AB_changed'] = ab_changed

        # B -> A direction uses B.tx and A.rx
        ba_val, ba_conf, ba_changed = repair_direction(
            b_tx if ba else None,
            a_rx if ab else None,
            trust_B, trust_A,
            hint_B.get('out_bias', 1.0), hint_A.get('in_bias', 1.0),
            hint_B.get('strength', 0.0), hint_A.get('strength', 0.0),
            has_peer=(ab is not None and ba is not None),
            swap_involved=pair_has_swap
        )
        repaired['BA_value'] = ba_val
        repaired['BA_conf'] = ba_conf
        repaired['BA_changed'] = ba_changed

        # Confidence uplift when both directions closely match their peers
        if (ab is not None and ba is not None and
            a_tx is not None and b_rx is not None and b_tx is not None and a_rx is not None):
            ab_match = approx_equal(safe_float(a_tx), safe_float(b_rx), rel_tol=REL_TOL)
            ba_match = approx_equal(safe_float(b_tx), safe_float(a_rx), rel_tol=REL_TOL)
            if ab_match and ba_match:
                repaired['AB_conf'] = clamp(repaired['AB_conf'] + 0.03, 0.0, 1.0)
                repaired['BA_conf'] = clamp(repaired['BA_conf'] + 0.03, 0.0, 1.0)

        pair_repairs[pair_key] = repaired

    # Build final result per interface, keeping metadata and applying pair repairs
    result: Dict[str, Dict[str, Tuple]] = {}

    for iface_id, data in telemetry.items():
        local = data.get('local_router')
        remote = data.get('remote_router')
        original_status = _normalize_status(data.get('interface_status', 'up'))
        # Apply swap if flagged so "original" and "repaired" are coherent
        o_tx = safe_float(data.get('tx_rate', 0.0), 0.0)
        o_rx = safe_float(data.get('rx_rate', 0.0), 0.0)
        if swap_flags.get(iface_id, False):
            o_tx, o_rx = o_rx, o_tx

        original_tx = max(0.0, o_tx)
        original_rx = max(0.0, o_rx)

        out: Dict[str, Any] = {}

        if local and remote and '_to_' in iface_id:
            pair_key = tuple(sorted([str(local), str(remote)]))
            pr = pair_repairs.get(pair_key)

            if pr is not None:
                # Shared status application
                repaired_status = pr['status']
                status_conf = pr['status_conf']

                # Directional application ensures coherent A->B and B->A views
                if str(local) == pair_key[0] and str(remote) == pair_key[1]:
                    # This interface is A->B
                    if pr['AB_value'] is None:
                        repaired_tx = original_tx
                        conf_tx = pr['AB_conf']
                    else:
                        repaired_tx = pr['AB_value']
                        conf_tx = pr['AB_conf']

                    # Its received traffic equals the opposite direction's repaired value
                    if pr['BA_value'] is None:
                        repaired_rx = original_rx
                        conf_rx = pr['BA_conf']
                    else:
                        repaired_rx = pr['BA_value']
                        conf_rx = pr['BA_conf']
                else:
                    # This interface is B->A
                    if pr['BA_value'] is None:
                        repaired_tx = original_tx
                        conf_tx = pr['BA_conf']
                    else:
                        repaired_tx = pr['BA_value']
                        conf_tx = pr['BA_conf']

                    if pr['AB_value'] is None:
                        repaired_rx = original_rx
                        conf_rx = pr['AB_conf']
                    else:
                        repaired_rx = pr['AB_value']
                        conf_rx = pr['AB_conf']
            else:
                # No pair info (rare), keep unchanged with moderate confidence
                repaired_status = original_status
                status_conf = 0.8
                repaired_tx = original_tx
                repaired_rx = original_rx
                conf_tx = 0.62
                conf_rx = 0.62
        else:
            # Missing metadata; be conservative and avoid cross-link inference
            repaired_status = original_status
            status_conf = 0.72
            repaired_tx = original_tx
            repaired_rx = original_rx
            conf_tx = 0.6
            conf_rx = 0.6

        # Final guard: an interface that is down cannot carry traffic
        if repaired_status == 'down':
            repaired_tx = 0.0
            repaired_rx = 0.0
            if original_status == 'down':
                status_conf = max(status_conf, 0.95)

        out['tx_rate'] = (float(original_tx), float(max(0.0, repaired_tx)), float(clamp(conf_tx, 0.0, 1.0)))
        out['rx_rate'] = (float(original_rx), float(max(0.0, repaired_rx)), float(clamp(conf_rx, 0.0, 1.0)))
        out['interface_status'] = (original_status, repaired_status, float(clamp(status_conf, 0.0, 1.0)))

        # Propagate metadata unchanged
        out['connected_to'] = data.get('connected_to')
        out['local_router'] = local
        out['remote_router'] = remote

        result[iface_id] = out

    return result


def run_repair(telemetry: Dict[str, Dict[str, Any]],
               topology: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Entry point expected by the evaluator.

    We intentionally do not rely on the provided 'topology' mapping to perform repairs.
    Instead, we trust the richer local metadata in telemetry (local_router/remote_router)
    to pair interfaces. This design is resilient to stale or incomplete summarized maps,
    while still honoring link symmetry and per-router flow conservation signals extracted
    directly from the paired interfaces.
    """
    return repair_network_telemetry(telemetry, topology)


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Dict

class Solution:
    """GEPA solution for telemetry_repair."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code (everything before the Solution class)
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}