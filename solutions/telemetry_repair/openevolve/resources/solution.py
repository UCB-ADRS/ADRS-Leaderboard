"""
Simple network interface telemetry repair. 

Takes interface telemetry data and then detects and repairs any inconsistencies.
Returns the same data structure with repairs and confidence scores for each measurement.
"""
# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting internal inconsistencies.
    
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
        Dictionary with same structure but values become tuples of:
        (original_value, repaired_value, confidence_score)
    """
    
    result = {}

    # Pre-calculate Router Health
    router_health = {}
    for rid, ifaces in topology.items():
        sin = sum(telemetry[i].get('rx_rate', 0) for i in ifaces if i in telemetry)
        sout = sum(telemetry[i].get('tx_rate', 0) for i in ifaces if i in telemetry)
        total = sin + sout
        router_health[rid] = 1.0 if total < 1.0 else max(0.0, 1.0 - abs(sin - sout) / total)
    
    def fuse(v1, v2, h1=0.5, h2=0.5):
        diff = abs(v1 - v2)
        avg = (v1 + v2) / 2.0
        if diff <= 0.05 * avg + 0.05: return avg, 1.0
        if v1 < 0.1 and v2 > 0.1: return v2, 0.9
        if v2 < 0.1 and v1 > 0.1: return v1, 0.9
        # Trust healthier router if significant health difference
        if h1 > h2 + 0.3: return v1, 0.8
        if h2 > h1 + 0.3: return v2, 0.8
        return avg, max(0.1, 1.0 - (diff / (avg + 1.0)))

    # Pass 1: Link Symmetry & Status Repair
    intermediate = {}
    for iface_id, data in telemetry.items():
        target_id = data.get('connected_to')
        target_data = telemetry.get(target_id) if target_id else None
        
        raw_stat = data.get('interface_status', 'down')
        raw_tx, raw_rx = data.get('tx_rate', 0.0), data.get('rx_rate', 0.0)
        
        # Defaults
        new_stat, conf_stat = raw_stat, 0.5
        new_tx, conf_tx = raw_tx, 0.5
        new_rx, conf_rx = raw_rx, 0.5
        
        if target_data:
            rem_stat = target_data.get('interface_status', 'down')
            rem_rx, rem_tx = target_data.get('rx_rate', 0.0), target_data.get('tx_rate', 0.0)
            
            # Repair Status
            traffic = max(raw_tx, raw_rx, rem_tx, rem_rx)
            if raw_stat == rem_stat:
                if raw_stat == 'down' and traffic > 0.5:
                    new_stat, conf_stat = 'up', 0.9
                else:
                    new_stat, conf_stat = raw_stat, 1.0
            else:
                new_stat = "up" if traffic > 0.1 else "down"
                conf_stat = 0.8
            
            # Repair Rates (Link Symmetry)
            if new_stat == 'down':
                new_tx, conf_tx = 0.0, 1.0
                new_rx, conf_rx = 0.0, 1.0
            else:
                h1 = router_health.get(data.get('local_router'), 0.5)
                h2 = router_health.get(data.get('remote_router'), 0.5)
                new_tx, conf_tx = fuse(raw_tx, rem_rx, h1, h2)
                new_rx, conf_rx = fuse(raw_rx, rem_tx, h1, h2)
        
        intermediate[iface_id] = {
            'status': (raw_stat, new_stat, conf_stat),
            'tx': (raw_tx, new_tx, conf_tx),
            'rx': (raw_rx, new_rx, conf_rx),
            'data': data
        }

    # Pass 2: Flow Conservation (Router Level)
    for router_id, ifaces in topology.items():
        # Only analyze if we have full visibility of the router
        r_ifaces = [i for i in ifaces if i in intermediate]
        if len(r_ifaces) != len(ifaces) or not r_ifaces: continue
        
        sum_in = sum(intermediate[i]['rx'][1] for i in r_ifaces)
        sum_out = sum(intermediate[i]['tx'][1] for i in r_ifaces)
        diff = sum_in - sum_out
        avg_flow = (sum_in + sum_out) / 2.0
        
        # Threshold: 5% + 1Mbps noise floor
        threshold = 0.05 * avg_flow + 1.0
        
        if abs(diff) <= threshold:
            # Balanced: Boost confidence
            for i in r_ifaces:
                for k in ['tx', 'rx']:
                    orig, curr, conf = intermediate[i][k]
                    if conf < 1.0:
                        intermediate[i][k] = (orig, curr, min(1.0, conf + 0.2))
        else:
            # Imbalanced: Try to repair stuck zero counters
            target = 'tx' if diff > 0 else 'rx'
            suspects = [i for i in r_ifaces if intermediate[i][target][1] < 0.1]
            
            if len(suspects) == 1:
                idx = suspects[0]
                orig, curr, conf = intermediate[idx][target]
                new_val = curr + abs(diff)
                intermediate[idx][target] = (orig, new_val, 0.9)
                
                s_orig, s_curr, s_conf = intermediate[idx]['status']
                if s_curr == 'down':
                    intermediate[idx]['status'] = (s_orig, 'up', 0.9)
                
                # Propagate repair to neighbor
                nid = intermediate[idx]['data'].get('connected_to')
                if nid and nid in intermediate:
                    n_target = 'rx' if target == 'tx' else 'tx'
                    n_orig, n_curr, n_conf = intermediate[nid][n_target]
                    if n_conf < 0.9 or n_curr < 0.1:
                        intermediate[nid][n_target] = (n_orig, new_val, 0.9)
                        ns_orig, ns_curr, ns_conf = intermediate[nid]['status']
                        if ns_curr == 'down':
                            intermediate[nid]['status'] = (ns_orig, 'up', 0.9)

    # Reconstruct result
    for iface_id, info in intermediate.items():
        res = {}
        res['interface_status'] = info['status']
        res['tx_rate'] = info['tx']
        res['rx_rate'] = info['rx']
        for k in ['connected_to', 'local_router', 'remote_router']:
            res[k] = info['data'].get(k)
        result[iface_id] = res
    
    
    return result

# EVOLVE-BLOCK-END


def run_repair(telemetry: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Main entry point that will be called by the OpenEvolve evaluator.
    
    Args:
        telemetry: Network interface telemetry data
        topology: Dictionary where key is router_id and value contains a list of interface_ids
    Returns:
        Dictionary containing validated results and summary stats
    """
    
    # Run repair.
    repaired_interfaces = repair_network_telemetry(telemetry, topology)
    
    return repaired_interfaces


# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Dict

class Solution:
    """OpenEvolve solution for telemetry_repair."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        # Read this file and return the code (everything before the Solution class)
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split('\n')
        end_idx = next(i for i, line in enumerate(lines) if 'class Solution:' in line)
        program_code = '\n'.join(lines[:end_idx])
        return {"code": program_code}