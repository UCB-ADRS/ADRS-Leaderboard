from pathlib import Path
from typing import Any, Dict

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType as SkyClusterType

import enum

class LocalClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

def strategy_step(ctx, last_cluster_type: LocalClusterType, has_spot: bool) -> LocalClusterType:
    # --- 1. State Extraction ---
    gap = ctx.env.gap_seconds
    elapsed = ctx.env.elapsed_seconds
    deadline = ctx.deadline
    overhead = ctx.restart_overhead
    total_work = ctx.task_duration
    work_done = sum(ctx.task_done_time)
    work_remaining = total_work - work_done
    time_remaining = deadline - elapsed

    # --- 2. Task Completion Check ---
    if work_remaining <= 0:
        return LocalClusterType.NONE

    # --- 3. Safety Critical: The "Point of No Return" ---
    # This is the most important check to prevent Deadline Misses (Score 0).
    # We must ensure that even if we switch to ON_DEMAND *right now*, we can finish.
    # Time needed = Work + Overhead (for the switch/restart).
    # We subtract a small buffer (gap) to account for the current tick's processing time.
    if time_remaining <= (work_remaining + overhead + gap):
        # We are in immediate danger.
        # If we are currently on SPOT and it's still available, staying is free and fastest.
        if has_spot and last_cluster_type == LocalClusterType.SPOT:
            return LocalClusterType.SPOT
        # Otherwise, we need the guarantee of ON_DEMAND.
        return LocalClusterType.ON_DEMAND

    # --- 4. Optimization Logic (Safe Zone) ---
    
    # Scenario A: SPOT is available this tick
    if has_spot:
        # If we are currently paying for ON_DEMAND, can we afford to switch to SPOT?
        if last_cluster_type == LocalClusterType.ON_DEMAND:
            # Calculate if switching is safe.
            # We lose one tick (gap) and pay overhead to switch.
            # We need: time_remaining > (work_remaining + overhead + gap)
            # Since we passed the check in step 3, we know time_remaining > (work_remaining + overhead).
            # We just need to ensure we aren't cutting it so close that the gap makes us miss.
            # A strict check: Ensure we have at least the gap + overhead buffer left.
            if time_remaining > (work_remaining + overhead + gap):
                return LocalClusterType.SPOT
            else:
                # Too risky to switch; stay on ON_DEMAND
                return LocalClusterType.ON_DEMAND
        
        # If we are on SPOT or NONE, just use SPOT (it's cheap and available)
        return LocalClusterType.SPOT

    # Scenario B: SPOT is NOT available
    else:
        # Hysteresis: If we are currently running ON_DEMAND, keep it running.
        # Stopping ON_DEMAND to wait for SPOT is risky (might miss deadline when switching back)
        # and wastes money (we paid for the instance this tick anyway).
        if last_cluster_type == LocalClusterType.ON_DEMAND:
            return LocalClusterType.ON_DEMAND

        # If we are currently on SPOT (preempted) or NONE:
        # Should we wait for SPOT to return (NONE) or switch to ON_DEMAND now?
        
        # Calculate a "Comfortable Buffer".
        # We wait if we have enough time to afford waiting + eventual switch overhead.
        # Buffer = (Work + Overhead) + (Safety Margin).
        # Safety Margin = 2 gaps (one for waiting, one for switching) + 10% of work.
        safety_margin = (2 * gap) + (work_remaining * 0.1)
        buffer_threshold = work_remaining + overhead + safety_margin

        if time_remaining > buffer_threshold:
            # We have plenty of time. Wait for cheaper SPOT.
            return LocalClusterType.NONE
        else:
            # Buffer is tight. Switch to ON_DEMAND now to be safe.
            return LocalClusterType.ON_DEMAND


class LEVISingleStrategy(Strategy):
    NAME = "LEVI_single"

    def _step(self, last_cluster_type: SkyClusterType, has_spot: bool) -> SkyClusterType:
        local_last = LocalClusterType[last_cluster_type.name]
        out = strategy_step(self, local_last, has_spot)
        return SkyClusterType[out.name]

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)


class Solution:
    """LEVI submission for cant_be_late."""

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        code = Path(__file__).read_text(encoding="utf-8")
        lines = code.split("\n")
        end_idx = next(i for i, line in enumerate(lines) if line.startswith("class Solution:"))
        return {"code": "\n".join(lines[:end_idx]).rstrip()}
