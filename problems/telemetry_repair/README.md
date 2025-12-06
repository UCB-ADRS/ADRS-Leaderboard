Telemetry Repair Problem
========================

Problem Setting
---------------
Repair corrupted network telemetry data by detecting and fixing inconsistencies. Given interface telemetry data from a network topology, identify measurements that are incorrect (due to hardware failures, measurement errors, configuration mistakes, or timing issues) and repair them while providing calibrated confidence scores.

Network telemetry data can become corrupted due to:
- Hardware failures causing counter drops (zeroed values)
- Measurement errors leading to scaled values
- Correlated failures affecting entire routers
- Timing issues between measurements

The challenge is to validate and repair this data by exploiting inherent relationships in network topology - for example, the receive rate on one interface should approximately match the transmit rate on the connected interface.

Target
------
- **Primary**: Maximize counter repair accuracy (75% weight)
- **Secondary**: Provide well-calibrated confidence scores (20% weight)
- **Tertiary**: Repair interface status correctly (5% weight)

API Specification
-----------------
Implement a `Solution` class that returns a repair algorithm:

```python
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/algorithm.py"}
        """
        pass
```

Your algorithm code must implement:

```python
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and fixing inconsistencies.
    
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
    """
    pass

def run_repair(telemetry: Dict[str, Dict[str, Any]], 
               topology: Dict[str, List[str]]) -> Dict[str, Any]:
    """Main entry point called by the evaluator."""
    return repair_network_telemetry(telemetry, topology)
```

Scoring (0-100)
---------------
```
weighted_accuracy = counter_repair × 0.75 + status_repair × 0.05 + confidence_calibration × 0.20

# Piecewise linear scaling
baseline = 0.80, knee = 0.82, optimal = 1.0

if weighted_accuracy <= knee:
    score = ((weighted_accuracy - baseline) / (knee - baseline)) × 50
else:
    score = 50 + ((weighted_accuracy - knee) / (optimal - knee)) × 50
score = max(0, min(100, score))
```

**0-Point Baseline (Passthrough)**:
- Returns input data unchanged (no repair attempted)
- Since ~80% of measurements are unperturbed, doing nothing appears deceptively accurate
- This is the naive baseline that any useful algorithm should beat

**100-Point Upper Bound (Perfect Repair)**:
- Every repaired value exactly matches ground truth
- Perfect confidence calibration (high when correct, low when uncertain)
- Requires oracle knowledge of ground truth - practically unreachable

Data Characteristics
--------------------
**Input Interface Data**:
```python
interfaces = {
    'if1': {
        'interface_status': 'up',      # 'up' or 'down'
        'rx_rate': 100.0,              # Receive rate in Mbps
        'tx_rate': 95.0,               # Transmit rate in Mbps  
        'connected_to': 'if2',         # ID of connected interface
        'local_router': 'router1',     # Router this interface belongs to
        'remote_router': 'router2'     # Router on other end
    }
}
```

**Output Format**:
```python
repaired = {
    'if1': {
        'interface_status': ('up', 'up', 0.95),      # (original, repaired, confidence)
        'rx_rate': (100.0, 100.0, 0.9),              # (original, repaired, confidence)
        'tx_rate': (95.0, 95.0, 0.9),                # (original, repaired, confidence)
        'connected_to': 'if2',                       # Metadata unchanged
        'local_router': 'router1',
        'remote_router': 'router2'
    }
}
```

Implementation Notes
--------------------
- Use topology to find relationships between interfaces at the same router
- Connected interfaces should have consistent rates: `my_tx ≈ their_rx`
- Key validation principles:
  - **Link Symmetry**: `my_tx_rate ≈ their_rx_rate` for connected interfaces
  - **Flow Conservation**: Traffic into a router ≈ traffic out
  - **Interface Consistency**: Status should match across connected pairs
- Confidence should reflect actual repair quality
- Handle both isolated errors and correlated failures
- Consider greedy repair, topology-aware inference, or iterative refinement approaches

Evaluation Details
------------------
- **Counter Repair Accuracy**: Measures how close repaired rx/tx rates are to ground truth
  - Error = |repaired - ground_truth| / ground_truth
  - Accuracy = 1.0 - average_error

- **Status Repair Accuracy**: Fraction of interface statuses correctly repaired

- **Confidence Calibration**: How well confidence scores reflect actual repair quality
  - Good repair (>80% accuracy) + high confidence (>0.7) = best score
  - Poor repair + high confidence = large penalty (overconfidence is dangerous)
  - Good repair + low confidence = moderate penalty (underconfidence is wasteful)

- **Test Scenarios**: Algorithm tested against multiple perturbation types:
  1. Random zeroing of counters (20% of interfaces)
  2. Correlated zeroing (entire router, 20% chance)
  3. Random scaling up/down (20% of interfaces)
  4. Correlated scaling (entire router, 20% chance)

- 30 test scenarios evaluated per run
- Time limit: 60 seconds per evaluation
