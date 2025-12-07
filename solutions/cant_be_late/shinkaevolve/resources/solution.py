# EVOLVE-BLOCK-START
"""
Adaptive Brake Strategy:
- Improves upon Gradient Buffer by adding second-order momentum (Acceleration) and variance-adaptive limits.
- Acceleration Brake: Detects "negative acceleration" (worsening trend) to apply penalties before EMA crossover.
- Variance-Adaptive Seal: Scales the maximum safety cushion based on market variance (sigma). 
  - Stable markets (low sigma) -> Smaller cushion (4x overhead) to save opportunity cost.
  - Choppy markets (high sigma) -> Larger cushion (up to 8x overhead) to bridge frequent gaps.
- Polynomial Gradient Entry: Maintains the quadratic risk/position check for safe SPOT entry.
- Strict Floors: Removes sticky logic in the danger zone for guaranteed safety.
"""

import math
from collections import deque
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class AdaptiveBrakeStrategy(Strategy):
    NAME = "adaptive_brake_strategy"

    def __init__(self, args):
        super().__init__(args)
        self.history_len = 60
        self.history = deque(maxlen=self.history_len)

        # EMA State
        self.ema_short = None
        self.ema_long = None
        self.prev_trend = 0.0

        # Counters
        self.od_dwell_ticks = 0

    def reset(self, env, task):
        super().reset(env, task)
        self.history.clear()
        self.ema_short = None
        self.ema_long = None
        self.prev_trend = 0.0
        self.od_dwell_ticks = 0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        gap = env.gap_seconds

        # ----------------------------------------------------------------------
        # 1. Update Observations
        # ----------------------------------------------------------------------
        val = 1.0 if has_spot else 0.0
        self.history.append(has_spot)

        # EMA Calculation
        if self.ema_short is None:
            self.ema_short = val
            self.ema_long = val
            self.prev_trend = 0.0
        else:
            # Short Window ~12 ticks -> alpha = 0.154
            alpha_s = 0.154
            # Long Window ~60 ticks -> alpha = 0.033
            alpha_l = 0.033

            self.ema_short = alpha_s * val + (1.0 - alpha_s) * self.ema_short
            self.ema_long = alpha_l * val + (1.0 - alpha_l) * self.ema_long

        # Jitter Metric
        jitter = 0.0
        history_list = list(self.history)
        if len(history_list) > 1:
            transitions = 0
            for i in range(1, len(history_list)):
                if history_list[i] != history_list[i-1]:
                    transitions += 1
            jitter = transitions / (len(history_list) - 1)

        # L_max (Stability)
        l_max = 0
        curr_run = 0
        for x in self.history:
            if x:
                curr_run += 1
                l_max = max(l_max, curr_run)
            else:
                curr_run = 0

        # Dwell Counter
        if env.cluster_type == ClusterType.ON_DEMAND:
            self.od_dwell_ticks += 1
        else:
            self.od_dwell_ticks = 0

        # ----------------------------------------------------------------------
        # 2. Risk Calculation (with Acceleration Brake)
        # ----------------------------------------------------------------------
        overhead_ticks = math.ceil(self.restart_overhead / gap)

        # Base Risk
        risk = 1.0 - self.ema_long

        # Trend & Acceleration
        trend = self.ema_short - self.ema_long
        accel = trend - self.prev_trend
        self.prev_trend = trend

        # Trend Penalty / Quality-Gated Bonus
        if trend < 0:
            risk += abs(trend) * 2.0
        else:
            quality_factor = min(1.0, l_max / max(1.0, 2.0 * overhead_ticks))
            risk -= trend * 0.5 * quality_factor

        # Acceleration Brake: Penalize negative second derivative (worsening trend)
        if accel < 0:
            risk += abs(accel) * 2.0

        # Jitter Penalty
        risk += jitter * 0.5

        # Stability Floor Penalty
        if l_max < 2 * overhead_ticks:
            risk += 0.2

        # Flash Crash Override
        if len(self.history) >= 3:
            recent = list(self.history)[-3:]
            if sum(recent) == 0:
                risk = max(risk, 2.0)

        # Clamp Risk
        risk = max(0.0, min(2.0, risk))

        # ----------------------------------------------------------------------
        # 3. Variance-Adaptive Geometry
        # ----------------------------------------------------------------------
        work_left = self.task_duration - sum(self.task_done_time)
        if work_left <= 1e-9:
            return ClusterType.NONE

        left_ticks = max(0, math.floor((self.deadline - env.elapsed_seconds) / gap))
        work_ticks = math.ceil(work_left / gap)

        # Safety Floors
        need1d = work_ticks + overhead_ticks
        need2d = work_ticks + 2 * overhead_ticks

        # Variance-Adaptive Seal
        # sigma ranges [0, 0.5] for Bernoulli distribution
        sigma = math.sqrt(max(0.0, self.ema_long * (1.0 - self.ema_long)))
        
        # Scale max_seal: Base 4 overheads + up to 4 more based on variance (Total 8)
        # High variance (choppy) -> larger buffer. Low variance (stable) -> smaller buffer.
        base_seal_ticks = 4.0 * overhead_ticks
        var_seal_ticks = 8.0 * overhead_ticks * sigma
        max_seal = math.ceil(base_seal_ticks + var_seal_ticks)
        
        seal_cushion = math.ceil(risk * max_seal)
        seal_line = need2d + seal_cushion

        # ----------------------------------------------------------------------
        # 4. Decision Logic
        # ----------------------------------------------------------------------

        # A. Danger Zone (<= need2d)
        if left_ticks <= need2d:
            # Strict safety floor (no sticky logic)
            return ClusterType.ON_DEMAND

        # B. Warning Zone (need2d < left <= seal_line)
        if left_ticks <= seal_line:
            # 1. No SPOT -> OD (Waiting burns buffer)
            if not has_spot:
                return ClusterType.ON_DEMAND

            # 2. Greedy Persistence (Already on SPOT)
            if env.cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT

            # 3. Entry Logic (Polynomial Gradient)
            if seal_cushion > 0:
                dist = left_ticks - need2d
                pos = dist / seal_cushion # 0.0 (danger) to 1.0 (safe)
                
                # Quadratic boundary for safe entry
                if risk < (pos ** 2):
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                return ClusterType.SPOT

        # C. Safe Zone (left > seal_line)
        if env.cluster_type == ClusterType.ON_DEMAND:
            # 1. Min Dwell (Risk-scaled)
            dwell_req = math.ceil(overhead_ticks * (0.5 + risk))
            if self.od_dwell_ticks < dwell_req:
                return ClusterType.ON_DEMAND

            # 2. Hysteresis (Thaw Buffer)
            thaw_line = seal_line + math.ceil(overhead_ticks)
            if left_ticks <= thaw_line:
                return ClusterType.ON_DEMAND

            # 3. Profitability Check
            if l_max < overhead_ticks:
                return ClusterType.ON_DEMAND

            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

        if env.cluster_type == ClusterType.SPOT:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
# EVOLVE-BLOCK-END

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
