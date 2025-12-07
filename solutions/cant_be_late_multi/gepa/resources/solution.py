# EVOLVE-BLOCK-START

import configargparse
import logging
import typing

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env, task

logger = logging.getLogger(__name__)

class EvolutionaryStrategy(MultiRegionStrategy):
    NAME = 'evolutionary_cost_deadline_balancer_v4_scan'

    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        # State that must not assume env/task availability here
        self.initialized: bool = False
        # region_cache[i] = {'has_spot': Optional[bool], 'last_checked': float, 'succ': int, 'fail': int}
        self.region_cache: typing.Dict[int, typing.Dict[str, typing.Any]] = {}
        self._last_seen_time: float = -1.0
        # Cooldown to avoid flapping from ON_DEMAND to SPOT too aggressively
        self._od_sticky_until: float = -1.0
        # Track idle waiting streak (returning NONE) when hunting for spot
        self._idle_wait_streak: int = 0

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.region_cache.clear()
        for i in range(self.env.get_num_regions()):
            self.region_cache[i] = {
                'has_spot': None,
                'last_checked': -1.0,
                'succ': 0,
                'fail': 0,
            }
        self.initialized = True
        self._last_seen_time = -1.0
        self._od_sticky_until = -1.0
        self._idle_wait_streak = 0
        logger.info(f"{self.NAME} strategy reset with {self.env.get_num_regions()} regions.")

    def _remaining_work(self) -> float:
        # Remaining compute time (seconds)
        return max(0.0, self.task_duration - sum(self.task_done_time))

    def _slack(self) -> float:
        # Time buffer between deadline and remaining work + pending restart overhead
        time_left = max(0.0, self.deadline - self.env.elapsed_seconds)
        remaining = self._remaining_work() + self.remaining_restart_overhead
        return time_left - remaining

    def _update_cache(self, region_idx: int, has_spot: bool):
        entry = self.region_cache.get(region_idx, None)
        now = self.env.elapsed_seconds
        if entry is None:
            self.region_cache[region_idx] = {
                'has_spot': has_spot,
                'last_checked': now,
                'succ': 1 if has_spot else 0,
                'fail': 0 if has_spot else 1,
            }
        else:
            entry['has_spot'] = has_spot
            entry['last_checked'] = now
            if has_spot:
                entry['succ'] = entry.get('succ', 0) + 1
            else:
                entry['fail'] = entry.get('fail', 0) + 1

    def _region_order_for_scan(self, current_region_idx: int) -> typing.List[int]:
        n = self.env.get_num_regions()
        # Score regions:
        # Priority 1: cached has_spot True, prefer most recently checked
        # Priority 2: unknown (never checked)
        # Priority 3: by success rate and staleness
        true_list = []
        unknown_list = []
        scored_list = []
        now = self.env.elapsed_seconds
        for i in range(n):
            if i == current_region_idx:
                continue
            entry = self.region_cache[i]
            if entry['has_spot'] is True:
                true_list.append((i, entry['last_checked']))
            elif entry['last_checked'] < 0:
                unknown_list.append(i)
            else:
                succ = entry.get('succ', 0)
                fail = entry.get('fail', 0)
                total = succ + fail
                sr = succ / total if total > 0 else 0.0
                staleness = now - entry['last_checked']
                scored_list.append((i, sr, staleness))
        true_list.sort(key=lambda x: x[1], reverse=True)
        scored_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
        ordered = [i for i, _ in true_list] + unknown_list + [i for i, _, _ in scored_list]
        # Place current region at the very front so we consider it first (already evaluated via arg)
        return [current_region_idx] + ordered

    def _scan_for_spot(self, has_spot_current: bool) -> typing.Tuple[bool, int]:
        """
        Scan regions at the current timestamp to find any with spot available.
        Leaves env in the found region if successful; otherwise restores original region.
        Returns (found, region_idx_if_found_or_current).
        """
        orig_idx = self.env.get_current_region()
        # Quick win: if current region has spot, no scan needed.
        if has_spot_current:
            return True, orig_idx

        order = self._region_order_for_scan(orig_idx)
        found_idx = None

        # Check current region already evaluated; start from next in order
        for idx in order:
            if idx == orig_idx:
                # already know it's False at this timestamp
                continue
            # Move to candidate and check spot immediately for this tick
            self.env.switch_region(idx)
            q = self.env.spot_available()
            self._update_cache(idx, q)
            if q:
                found_idx = idx
                break

        if found_idx is not None:
            return True, found_idx

        # Restore original region if not committing to a new region
        if self.env.get_current_region() != orig_idx:
            self.env.switch_region(orig_idx)
        return False, orig_idx

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized:
            return ClusterType.NONE

        now = self.env.elapsed_seconds
        if self._last_seen_time != now:
            self._last_seen_time = now
            # reset idle streak once per tick if work progressed or time advanced
            # (will be incremented only when we return NONE below)
            pass

        current_region_idx = self.env.get_current_region()

        # Update spot cache for current region from provided signal
        self._update_cache(current_region_idx, has_spot)

        remaining_work = self._remaining_work()
        if remaining_work <= 0:
            return ClusterType.NONE

        slack = self._slack()
        # Urgency thresholds
        oh = max(self.restart_overhead, 1e-6)
        urgent_threshold = oh  # when slack <= OH, avoid hunting and choose OD
        switch_from_od_min_slack = 0.5 * oh  # slack needed to switch from OD -> SPOT
        wait_ok_slack = 2.0 * oh            # slack needed to afford a wait tick

        # If time is extremely tight, immediately choose ON_DEMAND to avoid override
        if slack <= 0:
            self._od_sticky_until = max(self._od_sticky_until, now + oh)
            self._idle_wait_streak = 0
            return ClusterType.ON_DEMAND

        # If spot is available in current region, prefer SPOT aggressively unless OD is sticky and slack tiny
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND and now < self._od_sticky_until and slack < switch_from_od_min_slack:
                return ClusterType.ON_DEMAND
            # Commit to SPOT
            self._idle_wait_streak = 0
            return ClusterType.SPOT

        # No spot in current region: optionally scan all regions at this timestamp
        # Only scan if we can potentially act on the result safely given slack
        can_scan = True  # scanning is non-committal; we restore region if not switching
        found, found_idx = (False, current_region_idx)
        if can_scan:
            found, found_idx = self._scan_for_spot(has_spot_current=False)

        if found:
            # If we are on OD, ensure we have minimal slack to afford the switch
            if last_cluster_type == ClusterType.ON_DEMAND and slack < switch_from_od_min_slack and now < self._od_sticky_until:
                # revert to original region if scan moved us
                if self.env.get_current_region() != current_region_idx:
                    self.env.switch_region(current_region_idx)
                return ClusterType.ON_DEMAND
            self._idle_wait_streak = 0
            return ClusterType.SPOT

        # No regions have spot this tick
        # Decide between waiting (NONE), exploring later, or using ON_DEMAND
        if last_cluster_type == ClusterType.ON_DEMAND:
            # While on OD, continue OD unless we have ample slack to reconsider soon
            if slack >= 3.0 * oh and now >= self._od_sticky_until:
                # allow switching to SPOT if it appears next tick; continue OD for now
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND

        # If we were on SPOT (preempted) or haven't started yet
        if slack > wait_ok_slack:
            # Allow short idle waiting to avoid OD usage
            # Cap idle streak based on slack budget
            max_idle_ticks = min(5, int(slack // oh))  # small cap
            if self._idle_wait_streak < max_idle_ticks:
                self._idle_wait_streak += 1
                return ClusterType.NONE

        # Not enough slack to keep waiting: choose ON_DEMAND and set short sticky window
        self._od_sticky_until = max(self._od_sticky_until, now + 0.5 * oh)
        self._idle_wait_streak = 0
        return ClusterType.ON_DEMAND

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
