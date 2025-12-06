# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """

    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    # Magnitude-aware tolerance and absolute guard to avoid over-correcting tiny flows
    LOW_RATE_CUTOFF = 10.0  # Mbps
    ABS_GUARD = 0.5         # Mbps; require this absolute delta to trigger a repair
    QUIET_EPS = 0.1         # Mbps; consider as "no traffic" for asymmetry handling

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    result = {}

    # Build peer mapping
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        peers[if_id] = peer if peer in telemetry else None

    # Plan pairwise adjustments to avoid inconsistent per-side changes
    # field_value_adjust[(if_id, 'tx'|'rx')] = new_value
    field_value_adjust: Dict[Tuple[str, str], float] = {}
    # field_conf_assign[(if_id, dir)] = confidence if we actively changed value
    field_conf_assign: Dict[Tuple[str, str], float] = {}
    # field_conf_floor provides high-confidence floors for in-tolerance agreements
    field_conf_floor: Dict[Tuple[str, str], float] = {}
    # multiplicative confidence scalers for asymmetric traffic evidence
    field_conf_scale: Dict[Tuple[str, str], float] = {}

    visited_pairs = set()

    # Helper for relative difference
    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    # Compute pair-level per-direction adjustments
    for a_id, a_data in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        # process each undirected link once
        key = tuple(sorted([a_id, b_id]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        b_data = telemetry[b_id]

        sa = a_data.get('interface_status', 'unknown')
        sb = b_data.get('interface_status', 'unknown')

        a_tx = float(a_data.get('tx_rate', 0.0))
        a_rx = float(a_data.get('rx_rate', 0.0))
        b_tx = float(b_data.get('tx_rate', 0.0))
        b_rx = float(b_data.get('rx_rate', 0.0))

        # Activity-based trust weights (bias consensus toward stronger signal)
        act_a = max(a_tx, a_rx)
        act_b = max(b_tx, b_rx)
        denom_act = max(1e-9, act_a + act_b)
        w_a = act_a / denom_act
        w_b = act_b / denom_act

        # Only attempt counter fusion if both are not explicitly down; down logic handled later
        if sa != 'down' and sb != 'down':
            # Direction a->b: a_tx vs b_rx
            abs_ab = abs(a_tx - b_rx)
            max_ab = max(1.0, a_tx, b_rx)
            diff_ab = abs_ab / max_ab
            tol_ab = 0.05 if max(a_tx, b_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            # Low-rate–aware guards and convergence shaping
            low_band_ab = max(a_tx, b_rx) < LOW_RATE_CUTOFF
            abs_guard_ab = 0.3 if low_band_ab else ABS_GUARD
            full_mult_ab = 1.6 if low_band_ab else 2.0
            exp_ab = 1.2 if low_band_ab else 1.0

            if diff_ab > tol_ab and abs_ab > abs_guard_ab:
                # Trust-weighted consensus using activity as reliability proxy
                consensus_ab = w_a * a_tx + w_b * b_rx
                if diff_ab <= full_mult_ab * tol_ab:
                    # Partial averaging near threshold, accelerated on low-rate links
                    k_raw = (diff_ab - tol_ab) / max(tol_ab, 1e-9)
                    k = max(0.0, min(1.0, k_raw ** exp_ab))
                    new_a_tx = a_tx * (1.0 - k) + consensus_ab * k
                    new_b_rx = b_rx * (1.0 - k) + consensus_ab * k
                else:
                    # Clear violation: converge fully to consensus
                    new_a_tx = consensus_ab
                    new_b_rx = consensus_ab
                field_value_adjust[(a_id, 'tx')] = new_a_tx
                field_value_adjust[(b_id, 'rx')] = new_b_rx
                # Confidence aligned with disagreement magnitude
                conf_ab = clamp(1.0 - diff_ab)
                field_conf_assign[(a_id, 'tx')] = conf_ab
                field_conf_assign[(b_id, 'rx')] = conf_ab
            else:
                # Within tolerance: set magnitude-aware confidence floors
                if max(a_tx, b_rx) >= 10.0:
                    floor = 0.99 if diff_ab <= 0.005 else 0.98
                else:
                    floor = 0.97
                field_conf_floor[(a_id, 'tx')] = max(field_conf_floor.get((a_id, 'tx'), 0.0), floor)
                field_conf_floor[(b_id, 'rx')] = max(field_conf_floor.get((b_id, 'rx'), 0.0), floor)
                # Harmonize very strong agreements with geometric-mean floor
                if max(a_tx, b_rx) >= 10.0 and diff_ab <= 0.005:
                    fa = field_conf_floor.get((a_id, 'tx'), floor)
                    fb = field_conf_floor.get((b_id, 'rx'), floor)
                    gm = (max(1e-9, fa) * max(1e-9, fb)) ** 0.5
                    field_conf_floor[(a_id, 'tx')] = max(field_conf_floor.get((a_id, 'tx'), 0.0), gm)
                    field_conf_floor[(b_id, 'rx')] = max(field_conf_floor.get((b_id, 'rx'), 0.0), gm)

            # Direction b->a: b_tx vs a_rx
            abs_ba = abs(b_tx - a_rx)
            max_ba = max(1.0, b_tx, a_rx)
            diff_ba = abs_ba / max_ba
            tol_ba = 0.05 if max(b_tx, a_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            # Low-rate–aware guards and convergence shaping
            low_band_ba = max(b_tx, a_rx) < LOW_RATE_CUTOFF
            abs_guard_ba = 0.3 if low_band_ba else ABS_GUARD
            full_mult_ba = 1.6 if low_band_ba else 2.0
            exp_ba = 1.2 if low_band_ba else 1.0

            if diff_ba > tol_ba and abs_ba > abs_guard_ba:
                consensus_ba = w_b * b_tx + w_a * a_rx
                if diff_ba <= full_mult_ba * tol_ba:
                    k_raw = (diff_ba - tol_ba) / max(tol_ba, 1e-9)
                    k = max(0.0, min(1.0, k_raw ** exp_ba))
                    new_b_tx = b_tx * (1.0 - k) + consensus_ba * k
                    new_a_rx = a_rx * (1.0 - k) + consensus_ba * k
                else:
                    new_b_tx = consensus_ba
                    new_a_rx = consensus_ba
                field_value_adjust[(b_id, 'tx')] = new_b_tx
                field_value_adjust[(a_id, 'rx')] = new_a_rx
                conf_ba = clamp(1.0 - diff_ba)
                field_conf_assign[(b_id, 'tx')] = conf_ba
                field_conf_assign[(a_id, 'rx')] = conf_ba
            else:
                if max(b_tx, a_rx) >= 10.0:
                    floor = 0.99 if diff_ba <= 0.005 else 0.98
                else:
                    floor = 0.97
                field_conf_floor[(b_id, 'tx')] = max(field_conf_floor.get((b_id, 'tx'), 0.0), floor)
                field_conf_floor[(a_id, 'rx')] = max(field_conf_floor.get((a_id, 'rx'), 0.0), floor)
                # Harmonize very strong agreements with geometric-mean floor
                if max(b_tx, a_rx) >= 10.0 and diff_ba <= 0.005:
                    fb = field_conf_floor.get((b_id, 'tx'), floor)
                    fa = field_conf_floor.get((a_id, 'rx'), floor)
                    gm = (max(1e-9, fa) * max(1e-9, fb)) ** 0.5
                    field_conf_floor[(b_id, 'tx')] = max(field_conf_floor.get((b_id, 'tx'), 0.0), gm)
                    field_conf_floor[(a_id, 'rx')] = max(field_conf_floor.get((a_id, 'rx'), 0.0), gm)

            # Asymmetric confidence when only one side shows traffic (directional)
            if a_tx > QUIET_EPS and b_rx <= QUIET_EPS:
                field_conf_scale[(b_id, 'rx')] = min(field_conf_scale.get((b_id, 'rx'), 1.0), 0.88)
            if b_rx > QUIET_EPS and a_tx <= QUIET_EPS:
                field_conf_scale[(a_id, 'tx')] = min(field_conf_scale.get((a_id, 'tx'), 1.0), 0.88)
            if b_tx > QUIET_EPS and a_rx <= QUIET_EPS:
                field_conf_scale[(a_id, 'rx')] = min(field_conf_scale.get((a_id, 'rx'), 1.0), 0.88)
            if a_rx > QUIET_EPS and b_tx <= QUIET_EPS:
                field_conf_scale[(b_id, 'tx')] = min(field_conf_scale.get((b_id, 'tx'), 1.0), 0.88)

    # Second pass: repair using planned adjustments and assign calibrated confidences
    for interface_id, data in telemetry.items():
        repaired_data = {}

        interface_status = data.get('interface_status', 'unknown')
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Default: identity repair with high initial confidence; floors/penalties adjust later
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_confidence = 1.0
        tx_confidence = 1.0
        status_confidence = 1.0

        # Peer snapshot for redundancy use
        peer_status = None
        if connected_to and connected_to in telemetry:
            peer = telemetry[connected_to]
            peer_status = peer.get('interface_status', 'unknown')

        # Enforce interface consistency: if either side is down, set effective down and zero rates
        if interface_status == 'down' or (peer_status == 'down' if peer_status is not None else False):
            both_down = (interface_status == 'down' and (peer_status == 'down' if peer_status is not None else False))
            repaired_status = 'down'
            repaired_rx = 0.0
            repaired_tx = 0.0
            status_confidence = 0.95 if both_down else 0.7
            rx_confidence = status_confidence
            tx_confidence = status_confidence
        else:
            # Apply pairwise planned adjustments if available
            if (interface_id, 'rx') in field_value_adjust:
                repaired_rx = float(field_value_adjust[(interface_id, 'rx')])
                rx_confidence = field_conf_assign.get((interface_id, 'rx'), rx_confidence)
            if (interface_id, 'tx') in field_value_adjust:
                repaired_tx = float(field_value_adjust[(interface_id, 'tx')])
                tx_confidence = field_conf_assign.get((interface_id, 'tx'), tx_confidence)

            # If no peer redundancy, slightly lower confidence to reflect uncertainty
            if not connected_to or connected_to not in telemetry:
                rx_confidence = min(rx_confidence, 0.92)
                tx_confidence = min(tx_confidence, 0.92)

            # If statuses differ (but neither side is down), reduce status confidence moderately
            if connected_to and connected_to in telemetry:
                peer_status = telemetry[connected_to].get('interface_status', 'unknown')
                if interface_status != peer_status:
                    status_confidence = 0.6

            # Apply confidence floors for in-tolerance agreements
            rx_floor = field_conf_floor.get((interface_id, 'rx'))
            tx_floor = field_conf_floor.get((interface_id, 'tx'))
            if rx_floor is not None:
                rx_confidence = max(rx_confidence, rx_floor)
            if tx_floor is not None:
                tx_confidence = max(tx_confidence, tx_floor)

            # Apply asymmetric confidence scaling if present
            if (interface_id, 'rx') in field_conf_scale:
                rx_confidence = clamp(rx_confidence * field_conf_scale[(interface_id, 'rx')])
            if (interface_id, 'tx') in field_conf_scale:
                tx_confidence = clamp(tx_confidence * field_conf_scale[(interface_id, 'tx')])

        # Store repaired values with confidence scores
        repaired_data['rx_rate'] = (rx_rate, repaired_rx, clamp(rx_confidence))
        repaired_data['tx_rate'] = (tx_rate, repaired_tx, clamp(tx_confidence))
        repaired_data['interface_status'] = (interface_status, repaired_status, clamp(status_confidence))

        # Copy metadata unchanged
        repaired_data['connected_to'] = connected_to
        repaired_data['local_router'] = data.get('local_router')
        repaired_data['remote_router'] = data.get('remote_router')

        result[interface_id] = repaired_data

    # Router-level flow conservation: micro-adjustments on dominating dangling interfaces
    # Build router->interfaces mapping (prefer topology, augment with telemetry hints)
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        if lr:
            router_ifaces.setdefault(lr, [])
            if if_id not in router_ifaces[lr]:
                router_ifaces[lr].append(if_id)
        rr = d.get('remote_router')
        if rr and rr not in router_ifaces:
            router_ifaces[rr] = []

    # Compute initial per-router sums and residuals to target micro-adjustments
    router_sums: Dict[str, Tuple[float, float]] = {}
    for r, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in result:
                sum_tx += float(result[if_id]['tx_rate'][1])
                sum_rx += float(result[if_id]['rx_rate'][1])
        router_sums[r] = (sum_tx, sum_rx)

    # Apply tightly scoped micro-adjustments only on dominating dangling interfaces
    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        imbalance = sum_tx - sum_rx
        abs_imb = abs(imbalance)
        if abs_imb <= 0.0:
            continue  # already balanced

        denom = max(1.0, sum_tx, sum_rx)
        resid_frac = abs_imb / denom
        # Require sufficient residual to attempt micro-adjustments
        if resid_frac < 0.03:
            continue

        # Candidate interfaces: unpaired, up, with non-trivial traffic
        candidates = []
        for if_id in if_list:
            if if_id not in result:
                continue
            # unpaired if connected_to missing or not in telemetry
            connected_to = result[if_id].get('connected_to')
            is_unpaired = not connected_to or connected_to not in telemetry
            status = result[if_id]['interface_status'][1]
            if is_unpaired and status == 'up':
                txv = float(result[if_id]['tx_rate'][1])
                rxv = float(result[if_id]['rx_rate'][1])
                if max(txv, rxv) < LOW_RATE_CUTOFF:
                    continue  # avoid nudging tiny flows
                contrib = abs(txv - rxv)
                candidates.append((contrib, if_id, txv, rxv))

        if not candidates:
            continue

        # Pick dominating candidate
        candidates.sort(reverse=True)
        top_contrib, top_if, txv, rxv = candidates[0]
        if top_contrib < 0.5 * abs_imb:
            continue  # not dominating enough

        # Compute nudge magnitude capped at 2%
        alpha = min(0.02, 0.5 * resid_frac)
        if alpha <= 0.0:
            continue

        # Helper to compute router residual for a single router from current result snapshot
        def compute_router_resid_local(router_id: str) -> float:
            stx, srx = 0.0, 0.0
            for iid in router_ifaces.get(router_id, []):
                if iid in result:
                    stx += float(result[iid]['tx_rate'][1])
                    srx += float(result[iid]['rx_rate'][1])
            return abs(stx - srx) / max(1.0, stx, srx)

        # Adjust only the larger counter toward reducing router imbalance, tentatively
        orx, rrx, rc = result[top_if]['rx_rate']
        otx, rtx, tc = result[top_if]['tx_rate']
        pre_internal = abs(rtx - rrx) / max(1.0, max(rtx, rrx))

        if imbalance > 0:
            # sum_tx > sum_rx: decrease tx or increase rx
            if rtx >= rrx:
                new_tx = rtx * (1.0 - alpha)
                new_rx = rrx
                adjust_kind = 'tx'
            else:
                new_rx = rrx * (1.0 + alpha)
                new_tx = rtx
                adjust_kind = 'rx'
        else:
            # sum_tx < sum_rx: decrease rx or increase tx
            if rrx >= rtx:
                new_rx = rrx * (1.0 - alpha)
                new_tx = rtx
                adjust_kind = 'rx'
            else:
                new_tx = rtx * (1.0 + alpha)
                new_rx = rrx
                adjust_kind = 'tx'

        # Apply tentatively
        if adjust_kind == 'tx':
            result[top_if]['tx_rate'] = (otx, new_tx, tc)
        else:
            result[top_if]['rx_rate'] = (orx, new_rx, rc)

        resid_after = compute_router_resid_local(r)
        post_internal = abs(new_tx - new_rx) / max(1.0, max(new_tx, new_rx))

        # Commit only if improves residual >=8% and internal imbalance doesn't worsen by >3% relative
        if resid_after <= (1.0 - 0.08) * resid_frac and post_internal <= pre_internal + 0.03:
            # Reduce confidence to reflect heuristic adjustment
            if adjust_kind == 'tx':
                new_conf = min(tc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['tx_rate'] = (otx, new_tx, clamp(new_conf))
            else:
                new_conf = min(rc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['rx_rate'] = (orx, new_rx, clamp(new_conf))
        else:
            # Revert
            result[top_if]['tx_rate'] = (otx, rtx, tc)
            result[top_if]['rx_rate'] = (orx, rrx, rc)

    # Recompute per-router residual mismatch from possibly adjusted values
    router_resid: Dict[str, float] = {}
    router_sum_tx: Dict[str, float] = {}
    router_sum_rx: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in result:
                sum_tx += float(result[if_id]['tx_rate'][1])
                sum_rx += float(result[if_id]['rx_rate'][1])
        denom = max(1.0, sum_tx, sum_rx)
        router_resid[r] = abs(sum_tx - sum_rx) / denom
        router_sum_tx[r] = sum_tx
        router_sum_rx[r] = sum_rx

    # Apply tri-axis confidence composition:
    # - direction-aware router penalties,
    # - link symmetry fit after repair,
    # - correction magnitude; plus magnitude-aware floors and asymmetric traffic-evidence shaping.
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        rr = d.get('remote_router')
        orx, rrx, rc = result[if_id]['rx_rate']
        otx, rtx, tc = result[if_id]['tx_rate']
        ost, rst, sc = result[if_id]['interface_status']

        resid_local = router_resid.get(lr, 0.0)
        resid_remote = router_resid.get(rr, 0.0)
        # Share-aware, direction-coupled penalties
        sum_tx_local = router_sum_tx.get(lr, 0.0)
        sum_rx_local = router_sum_rx.get(lr, 0.0)
        tx_share = rtx / max(1.0, sum_tx_local)
        rx_share = rrx / max(1.0, sum_rx_local)
        penalty_tx = clamp(1.0 - ((0.6 + 0.2 * tx_share) * resid_local + (0.4 - 0.2 * tx_share) * resid_remote), 0.5, 1.0)
        penalty_rx = clamp(1.0 - ((0.6 + 0.2 * rx_share) * resid_local + (0.4 - 0.2 * rx_share) * resid_remote), 0.5, 1.0)

        # If repaired status is down, keep zeros and only scale confidences by router penalties
        if rst == 'down':
            result[if_id]['rx_rate'] = (orx, 0.0, clamp(rc * penalty_rx))
            result[if_id]['tx_rate'] = (otx, 0.0, clamp(tc * penalty_tx))
            status_scale = 0.85 + 0.15 * min(penalty_tx, penalty_rx)
            result[if_id]['interface_status'] = (ost, rst, clamp(sc * status_scale))
            continue

        # Peer repaired values for symmetry assessment
        peer_id = d.get('connected_to')
        peer_exists = (peer_id in result) if peer_id else False
        if peer_exists:
            _, peer_rrx, _ = result[peer_id]['rx_rate']
            _, peer_rtx, _ = result[peer_id]['tx_rate']
        else:
            peer_rrx = None
            peer_rtx = None

        # Symmetry fit after repair
        if peer_exists and peer_rrx is not None and peer_rtx is not None:
            diff_tx = rel_diff(rtx, peer_rrx)  # my_tx vs their_rx
            diff_rx = rel_diff(rrx, peer_rtx)  # my_rx vs their_tx
            c_sym_tx = clamp(1.0 - diff_tx)
            c_sym_rx = clamp(1.0 - diff_rx)

            # Magnitude-aware thresholds and floors
            th_tx = 0.05 if max(rtx, peer_rrx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            th_rx = 0.05 if max(rrx, peer_rtx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            if max(rtx, peer_rrx) >= 10.0 and diff_tx <= 0.005:
                floor_tx = 0.99
            elif diff_tx <= th_tx:
                floor_tx = 0.98 if max(rtx, peer_rrx) >= 10.0 else 0.97
            else:
                floor_tx = 0.0
            if max(rrx, peer_rtx) >= 10.0 and diff_rx <= 0.005:
                floor_rx = 0.99
            elif diff_rx <= th_rx:
                floor_rx = 0.98 if max(rrx, peer_rtx) >= 10.0 else 0.97
            else:
                floor_rx = 0.0
        else:
            # No redundancy; use conservative defaults
            c_sym_tx = 0.9
            c_sym_rx = 0.9
            floor_tx = 0.0
            floor_rx = 0.0

        # Correction magnitude component (bigger changes => lower confidence)
        m_tx = abs(rtx - otx) / max(1.0, rtx, otx)
        m_rx = abs(rrx - orx) / max(1.0, rrx, orx)
        c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
        c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

        # Compose final confidences
        conf_tx_new = clamp(0.45 * penalty_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
        conf_rx_new = clamp(0.45 * penalty_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)

        # Apply magnitude-aware floors
        conf_tx_new = max(conf_tx_new, floor_tx)
        conf_rx_new = max(conf_rx_new, floor_rx)

        # Asymmetric traffic-evidence shaping: penalize silent side if peer shows traffic
        if peer_exists and peer_rrx is not None and rtx <= QUIET_EPS and peer_rrx > QUIET_EPS:
            conf_tx_new = clamp(conf_tx_new * 0.88)
        if peer_exists and peer_rtx is not None and rrx <= QUIET_EPS and peer_rtx > QUIET_EPS:
            conf_rx_new = clamp(conf_rx_new * 0.88)

        result[if_id]['tx_rate'] = (otx, rtx, conf_tx_new)
        result[if_id]['rx_rate'] = (orx, rrx, conf_rx_new)

        # Status confidence shaping: residual-based scaling + mild alignment with per-direction confidences
        status_scale = 0.85 + 0.15 * min(penalty_tx, penalty_rx)
        status_conf_new = clamp(sc * status_scale)
        status_conf_new = clamp(status_conf_new * (0.85 + 0.15 * min(conf_tx_new, conf_rx_new)))
        result[if_id]['interface_status'] = (ost, rst, status_conf_new)

    return result

# EVOLVE-BLOCK-END


def run_repair(telemetry: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Main entry point that will be called by the evaluator.

    Args:
        telemetry: Network interface telemetry data
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary containing repaired results with confidence scores
    """
    return repair_network_telemetry(telemetry, topology)


if __name__ == "__main__":
    # Simple test case
    test_telemetry = {
        'if1_to_if2': {
            'interface_status': 'up',
            'rx_rate': 100.0,
            'tx_rate': 95.0,
            'connected_to': 'if2_to_if1',
            'local_router': 'router1',
            'remote_router': 'router2'
        },
        'if2_to_if1': {
            'interface_status': 'up',
            'rx_rate': 95.0,  # Should match if1's TX
            'tx_rate': 100.0,  # Should match if1's RX
            'connected_to': 'if1_to_if2',
            'local_router': 'router2',
            'remote_router': 'router1'
        }
    }

    test_topology = {
        'router1': ['if1_to_if2'],
        'router2': ['if2_to_if1']
    }

    result = run_repair(test_telemetry, test_topology)

    print("Repair results:")
    for if_id, data in result.items():
        print(f"\n{if_id}:")
        print(f"  RX: {data['rx_rate']}")
        print(f"  TX: {data['tx_rate']}")
        print(f"  Status: {data['interface_status']}")


# Solution class wrapper for evaluator compatibility
from pathlib import Path

class Solution:
    """Shinka solution for telemetry_repair."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code (everything before the Solution class)
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}