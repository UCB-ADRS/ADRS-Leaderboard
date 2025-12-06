"""
Baseline submission for the txn_scheduling problem.
"""

from pathlib import Path
from typing import Any, Dict, Optional


class Solution:
    """Baseline `Solution` implementation for txn_scheduling."""

    def __init__(self) -> None:
        # When copied to execution_env, the path is:
        # problems/txn_scheduling/resources/execution_env/solution_env/solution.py
        # So parents[2] is problems/txn_scheduling/resources/
        current = Path(__file__).resolve()
        
        # Try multiple possible locations
        possible_paths = [
            current.parents[2] / "initial_program.py",  # From execution_env
            current.parents[4] / "problems" / "txn_scheduling" / "resources" / "initial_program.py",  # From execution_env to repo root
            current.parents[2] / "problems" / "txn_scheduling" / "resources" / "initial_program.py",  # Original structure
        ]
        
        self._initial_program_path = None
        for path in possible_paths:
            if path.exists():
                self._initial_program_path = path
                break

    def solve(self, spec_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the scheduling program with a fixed seed for consistency.
        
        Args:
            spec_path: Path to submission spec (accepted for interface compatibility)
        
        Returns:
            Dict with 'code' key containing the program source
        """
        if self._initial_program_path and self._initial_program_path.exists():
            original_code = self._initial_program_path.read_text(encoding="utf-8")
            
            # Add a fixed seed at the start for deterministic results
            modified_code = original_code.replace(
                "def get_random_costs():",
                """def get_random_costs():
    random.seed(42)  # Fixed seed for reproducibility"""
            )
            return {"code": modified_code}
        else:
            # Fallback: return a trivial program that just returns baseline cost
            # This should NOT be used - it's a safety net
            fallback_code = (
                "import time, random\n\n"
                "def get_random_costs():\n"
                "    start = time.time()\n"
                "    random.seed(42)\n"
                "    # Trivial baseline: no transactions, zero cost.\n"
                "    return 0, [], time.time() - start\n\n"
                "if __name__ == '__main__':\n"
                "    makespan, schedule, dt = get_random_costs()\n"
                "    print(f'Makespan={makespan}, Time={dt}')\n"
            )
            return {"code": fallback_code}
