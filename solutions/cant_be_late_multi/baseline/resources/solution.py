from pathlib import Path
from typing import Dict, Any


class Solution:
    """Baseline submission for cant_be_late_multi: returns the initial evolutionary strategy."""

    def __init__(self) -> None:
        # When COPIED to execution_env/solution_env/solution.py and run from there:
        # parents[0]=solution_env, [1]=execution_env, [2]=resources, [3]=cant_be_late_multi, [4]=problems, [5]=repo_root
        self._default_strategy_path = (
            Path(__file__).resolve()
            .parents[5]
            / "problems"
            / "cant_be_late_multi"
            / "resources"
            / "initial_program.py"
        )

    def solve(self, spec_path: str | None = None) -> Dict[str, Any]:
        try:
            code = self._default_strategy_path.read_text(encoding="utf-8")
            # Extract just the code between EVOLVE-BLOCK markers
            lines = code.split('\n')
            start_idx = -1
            end_idx = -1
            
            for i, line in enumerate(lines):
                if 'EVOLVE-BLOCK-START' in line:
                    start_idx = i + 1
                elif 'EVOLVE-BLOCK-END' in line:
                    end_idx = i
                    break
            
            if start_idx != -1 and end_idx != -1:
                strategy_code = '\n'.join(lines[start_idx:end_idx])
                return {"code": strategy_code}
            else:
                # Fallback: return the entire file
                return {"code": code}
        except FileNotFoundError:
            # Fallback minimal strategy if the reference file is missing.
            return {
                "code": (
                    "import configargparse\n"
                    "import json\n"
                    "import logging\n"
                    "import math\n"
                    "import typing\n\n"
                    "from sky_spot.strategies.multi_strategy import MultiRegionStrategy\n"
                    "from sky_spot.utils import ClusterType\n\n"
                    "if typing.TYPE_CHECKING:\n"
                    "    from sky_spot import env, task\n\n"
                    "logger = logging.getLogger(__name__)\n\n"
                    "class EvolutionaryStrategy(MultiRegionStrategy):\n"
                    '    NAME = \'evolutionary_robust_starter\'\n\n'
                    "    def __init__(self, args: configargparse.Namespace):\n"
                    "        super().__init__(args)\n"
                    "        self.initialized: bool = False\n"
                    "        self.region_cache: typing.Dict[int, typing.Dict[str, typing.Any]] = {}\n"
                    "        self.next_exploration_target_idx: int = 0\n\n"
                    "    def reset(self, env: 'env.Env', task: 'task.Task'):\n"
                    "        super().reset(env, task)\n"
                    "        for i in range(self.env.get_num_regions()):\n"
                    "            self.region_cache[i] = {'has_spot': None, 'last_checked': -1}\n"
                    "        self.initialized = True\n\n"
                    "    def _is_behind_schedule(self) -> bool:\n"
                    "        if not self.initialized:\n"
                    "            return False\n"
                    "        c_0 = self.task_duration\n"
                    "        c_t = self.task_duration - sum(self.task_done_time)\n"
                    "        t = self.env.elapsed_seconds\n"
                    "        r_0 = self.deadline\n"
                    "        if r_0 <= t:\n"
                    "            return True\n"
                    "        if t == 0:\n"
                    "            return False\n"
                    "        required_progress = t * (c_0 / r_0)\n"
                    "        actual_progress = c_0 - c_t\n"
                    "        return actual_progress < required_progress\n\n"
                    "    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:\n"
                    "        if not self.initialized or self.task_done:\n"
                    "            return ClusterType.NONE\n"
                    "        current_region_idx = self.env.get_current_region()\n"
                    "        num_regions = self.env.get_num_regions()\n"
                    "        self.region_cache[current_region_idx]['has_spot'] = has_spot\n"
                    "        self.region_cache[current_region_idx]['last_checked'] = self.env.elapsed_seconds\n"
                    "        if self._is_behind_schedule():\n"
                    "            if has_spot:\n"
                    "                return ClusterType.SPOT\n"
                    "            else:\n"
                    "                return ClusterType.ON_DEMAND\n"
                    "        else:\n"
                    "            if has_spot:\n"
                    "                return ClusterType.SPOT\n"
                    "            else:\n"
                    "                self.next_exploration_target_idx = (current_region_idx + 1) % num_regions\n"
                    "                self.env.switch_region(self.next_exploration_target_idx)\n"
                    "                return ClusterType.NONE\n"
                )
            }
