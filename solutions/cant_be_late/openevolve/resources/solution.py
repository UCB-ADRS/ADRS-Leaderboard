# EVOLVE-BLOCK START
"""
Minimal greedy seed (non‑UP):
- Prefer SPOT when available
- Otherwise wait (NONE)
- Enforce hard safety lines with tick‑aligned checks; equality is unsafe
  • If need1d ≥ left_ticks → use ON_DEMAND (point of no return)
  • If in 2d zone and no productive SPOT → use ON_DEMAND

No comparisons to any baseline/average progress; uses only tick variables.
"""

import math
import collections
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class GreedySafetyStrategy(Strategy):
    NAME = "adaptive_regime_v4"

    def __init__(self, args):
        super().__init__(args)
        self.history = collections.deque()
        self.od_entry_tick = None
        self.sealed = False
        self.spot_consecutive = 0

    def reset(self, env, task):
        super().reset(env, task)
        gap = env.gap_seconds
        # Dynamic window ~1 hour
        win_len = max(10, int(3600 / gap))
        self.history = collections.deque(maxlen=win_len)
        self.od_entry_tick = None
        self.sealed = False
        self.spot_consecutive = 0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        gap = env.gap_seconds
        tick = env.tick

        # 1. Update Statistics
        self.history.append(has_spot)
        if has_spot:
            self.spot_consecutive += 1
        else:
            self.spot_consecutive = 0
        
        if len(self.history) > 0:
            alpha = sum(self.history) / len(self.history)
            # Compute L
            max_l = 0
            curr = 0
            for x in self.history:
                if x:
                    curr += 1
                else:
                    max_l = max(max_l, curr)
                    curr = 0
            L = max(max_l, curr)
        else:
            alpha = 1.0 if has_spot else 0.0
            L = 1 if has_spot else 0

        # 2. Tick-aligned Bookkeeping
        work_left = self.task_duration - sum(self.task_done_time)
        if work_left <= 1e-9:
            return ClusterType.NONE

        left_ticks = max(0, math.floor((self.deadline - env.elapsed_seconds) / gap))
        od_ticks = math.ceil(self.restart_overhead / gap)
        
        need1d = math.ceil((work_left + self.restart_overhead) / gap)
        need2d = math.ceil((work_left + 2 * self.restart_overhead) / gap)

        # 3. Regime Parameters
        # Scarce: Low availability OR short runs. Threshold 0.4 aligns with better historical perf.
        is_scarce = (alpha < 0.4) or (L < 2.0 * od_ticks)

        if is_scarce:
            # Conservative: Seal early, Hard to exit OD, Force OD if preempted
            m_seal = 3 * od_ticks
            m_exit = 5 * od_ticks
            m_wait = 100 * od_ticks
        else:
            # Rich: Seal late, Moderate exit, Preserve buffer in NONE
            # m_wait > m_exit ensures we preserve enough buffer to exit OD later
            m_seal = 1 * od_ticks
            m_exit = 2 * od_ticks
            m_wait = 4 * od_ticks

        min_od_dwell = 2 * od_ticks

        # Track OD entry
        if last_cluster_type == ClusterType.ON_DEMAND:
            if self.od_entry_tick is None:
                self.od_entry_tick = tick
        else:
            self.od_entry_tick = None

        # 4. Decision Logic
        
        # A. Critical Safety (1d)
        if need1d >= left_ticks:
            return ClusterType.ON_DEMAND

        # B. Tail Sealing
        if not self.sealed:
            if left_ticks <= need1d + m_seal:
                self.sealed = True
        
        if self.sealed:
            # Thaw logic: strict requirements to unlock
            if not is_scarce and has_spot and left_ticks > need2d + m_exit:
                self.sealed = False
            else:
                return ClusterType.ON_DEMAND

        # C. 2d Zone
        if need2d >= left_ticks:
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # D. Normal Operation
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Dwell
            if (self.od_entry_tick is not None) and (tick - self.od_entry_tick < min_od_dwell):
                return ClusterType.ON_DEMAND
            
            # Exit Guards
            if left_ticks <= need2d + m_exit:
                return ClusterType.ON_DEMAND
            
            # Check stability (debounce)
            if self.spot_consecutive < od_ticks:
                return ClusterType.ON_DEMAND
            
            # Profitability: L must be substantial
            # If we have massive slack, we can be riskier.
            has_excess_slack = left_ticks > need2d + 8 * od_ticks
            if not has_excess_slack and L < 1.5 * od_ticks:
                return ClusterType.ON_DEMAND
                
            return ClusterType.SPOT

        elif last_cluster_type == ClusterType.SPOT:
            if has_spot:
                return ClusterType.SPOT
            else:
                # Preempted
                if left_ticks > need2d + m_wait:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND

        else: # NONE
            if has_spot:
                return ClusterType.SPOT
            else:
                if left_ticks > need2d + m_wait:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

# EVOLVE-BLOCK END

# Solution class wrapper for evaluator compatibility
from pathlib import Path
from typing import Any, Dict

class Solution:
    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split("\n")
        end_idx = next(i for i, line in enumerate(lines) if "class Solution:" in line)
        program_code = "\n".join(lines[:end_idx])
        return {"code": program_code}

