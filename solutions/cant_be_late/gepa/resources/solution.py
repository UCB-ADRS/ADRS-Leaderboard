import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class ImprovedEvolveStrategy(Strategy):
    NAME = 'gepa_cbl_strategy_v1'
    
    def __init__(self, args):
        super().__init__(args)
        # Hyperparameters
        self.safety_factor = 1.1  # Multiplier for task duration to account for potential bad luck/overhead
        self.min_slack_threshold = 0.5 # Hours. If slack is less than this, don't risk waiting for spot.
    
    def reset(self, env, task):
        super().reset(env, task)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        
        # 1. Task Completion Check
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        
        # 2. Time Calculations
        elapsed_time = env.elapsed_seconds
        remaining_time_until_deadline = self.deadline - elapsed_time
        
        # Calculate the "Point of No Return" (PONR).
        # This is the moment where we MUST start running continuously to finish.
        # We include the restart overhead because switching to OD takes time.
        time_required_to_finish = remaining_task_time + self.restart_overhead
        
        # Calculate Slack: How much extra time do we have beyond the work needed?
        slack = remaining_time_until_deadline - time_required_to_finish
        
        # 3. CRITICAL DEADLINE CHECK (The "Panic" Line)
        # If we are at or past the point where we need to run continuously to finish,
        # we must use ON_DEMAND immediately. 
        # We use a tiny epsilon for float comparison safety.
        if slack <= 0.05: # 3 minutes buffer
            return ClusterType.ON_DEMAND

        # 4. Spot Utilization Strategy
        if has_spot:
            # If we are currently on SPOT, keep using it.
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            
            # If we were waiting or on OD (unlikely), decide whether to switch to SPOT.
            # We switch to SPOT if we have enough slack to absorb a potential failure.
            # If slack is very tight (but not negative), switching to SPOT is risky
            # because if it dies in 10 mins, we pay overhead again and lose ground.
            # However, if we are currently NONE (waiting), and SPOT is available, take it.
            return ClusterType.SPOT

        # 5. Spot Unavailability Strategy (Waiting vs OD)
        else:
            # Spot is NOT available.
            
            # If we were already running ON_DEMAND, should we keep running it?
            # Yes, unless we have a massive amount of slack that allows us to pause 
            # and wait for SPOT to come back cheaper.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # If we have huge slack (e.g., > 2x the remaining work), we might pause 
                # to save money. But usually, once on OD, it's safer to stick until SPOT returns.
                # Here we stick to OD to avoid restart overheads of toggling.
                return ClusterType.ON_DEMAND

            # If we are waiting (NONE) or were on SPOT (and it just died):
            # We have slack (checked in step 3).
            # Should we burn money on OD now to "get ahead", or wait for SPOT?
            
            # Strategy: Wait for SPOT as long as the slack is "comfortable".
            # If slack drops below a threshold, start OD to ensure we don't get 
            # backed into a corner where we have 0 slack and forced OD at the end.
            
            # If slack is small (e.g., < 20% of the remaining work), start OD now.
            # This smoothes out the cost. It's better to run a bit of OD now and 
            # catch a Spot later, than to wait, lose all slack, and run pure OD later.
            slack_ratio = slack / remaining_task_time
            
            if slack_ratio < 0.2: 
                return ClusterType.ON_DEMAND
            else:
                # We have plenty of time. Wait for SPOT to become available.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)


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
