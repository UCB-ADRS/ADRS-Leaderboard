#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from resources.run_evaluator_real30 import evaluate_artifact

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "resources" / "submission_spec.json"
ARTIFACT_PATH = Path("./output_ans").resolve()


def load_solution_module(solution_path: Path) -> ModuleType:
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found at {solution_path}")
    spec = importlib.util.spec_from_file_location("submitted_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_artifact(result: Any, solution_path: Path) -> Path:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, dict):
        with ARTIFACT_PATH.open("w", encoding="utf-8") as fout:
            json.dump(result, fout)
        return ARTIFACT_PATH
    if isinstance(result, str):
        candidate = Path(result)
        if candidate.is_file():
            with ARTIFACT_PATH.open("w", encoding="utf-8") as fout:
                json.dump({"program_path": str(candidate.resolve())}, fout)
            return ARTIFACT_PATH
        with ARTIFACT_PATH.open("w", encoding="utf-8") as fout:
            fout.write(result)
        return ARTIFACT_PATH
    raise TypeError(
        "Solution.solve() must return a dict/path-string/code-string; got "
        f"{type(result)!r}."
    )


def evaluate(solution_path: Path, spec_path: Path) -> dict:
    module = load_solution_module(solution_path)
    if not hasattr(module, "Solution"):
        raise AttributeError("solution.py must define a 'Solution' class")
    SolutionCls = module.Solution  # type: ignore[attr-defined]
    solution_obj = SolutionCls()
    if not hasattr(solution_obj, "solve"):
        raise AttributeError("Solution class must define a 'solve' method")
    solve_fn = getattr(solution_obj, "solve")
    result = solve_fn(str(spec_path))
    artifact_path = materialize_artifact(result, solution_path)
    payload = evaluate_artifact(str(artifact_path))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cant-be-late solution module")
    parser.add_argument(
        "--solution",
        default="../../execution_env/solution_env/solution.py",
        help="Path to contestant solution.py",
    )
    parser.add_argument(
        "--spec",
        default=str(DEFAULT_SPEC),
        help="Path to submission spec (passed to Solution.solve)",
    )
    args = parser.parse_args()

    solution_path = Path(args.solution).resolve()
    spec_path = Path(args.spec).resolve()
    try:
        payload = evaluate(solution_path, spec_path)
    except Exception as e:
        print(json.dumps({"error": str(e), "score": 0}))
        raise
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
