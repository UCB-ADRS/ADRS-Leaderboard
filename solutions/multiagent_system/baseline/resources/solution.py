"""Baseline delegating solution for multiagent_system.
Provides both:
- Solution.solve() returning the evolvable code block (matching other solution.py patterns)
- async run_multi_agent_task delegating to problem initial_program (required by evaluator)
"""
from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional

# Resolve path to problem initial_program.py
# From solution_env/solution.py, go up to problem directory:
# solution_env -> execution_env -> resources -> multiagent_system (problem dir)
PROBLEM_DIR = Path(__file__).resolve().parents[3]
PROBLEM_INITIAL = PROBLEM_DIR / 'resources' / 'initial_program.py'

# Only try to import if the file exists, otherwise use a fallback
if PROBLEM_INITIAL.exists():
    _spec = importlib.util.spec_from_file_location('initial_program_delegate', PROBLEM_INITIAL)
    if _spec and _spec.loader:
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
    else:
        _module = None
else:
    _module = None

class Solution:
    """Baseline submission for multiagent_system.
    Returns the code inside the EVOLVE-BLOCK markers of initial_program.py.
    Falls back to the entire file or a minimal stub if not found.
    """
    def __init__(self) -> None:
        self._initial_program_path = PROBLEM_INITIAL

    def solve(self, spec_path: Optional[str] = None) -> Dict[str, Any]:
        """Return evolvable code block as a dict with key 'code'.
        spec_path is accepted for interface consistency; unused in baseline.
        """
        try:
            code = self._initial_program_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            # Fallback minimal system (very small) if file missing
            return {"code": (
                "# Fallback minimal multi-agent system\n"
                "import asyncio\n\n"
                "async def run_multi_agent_task(idea: str, n_rounds: int = 1, log_file: str | None = None):\n"
                "    return f'Fallback trace for: {idea}'\n"
            )}
        # Extract EVOLVE block
        lines = code.split('\n')
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            if '# EVOLVE-BLOCK-START' in line:
                start_idx = i + 1
            elif '# EVOLVE-BLOCK-END' in line:
                end_idx = i
                break
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            evolve_code = '\n'.join(lines[start_idx:end_idx])
            return {"code": evolve_code}
        # Fallback: whole file
        return {"code": code}

async def run_multi_agent_task(idea: str, n_rounds: int = 4, log_file: Optional[str] = None):
    """Delegate to problem initial_program.run_multi_agent_task (required by evaluator)."""
    if _module and hasattr(_module, 'run_multi_agent_task'):
        return await _module.run_multi_agent_task(idea=idea, n_rounds=n_rounds, log_file=log_file)
    else:
        # Fallback if initial_program not available
        return f"Fallback trace for: {idea}"
