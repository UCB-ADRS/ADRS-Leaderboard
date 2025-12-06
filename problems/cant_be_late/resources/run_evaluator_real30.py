#!/usr/bin/env python
import json
import os
import sys
from pathlib import Path



def evaluate_artifact(artifact_path: str) -> dict:
    """Evaluate a strategy artifact; return payload with score and metrics."""

    artifact_path = os.path.abspath(artifact_path)

    resources_dir = Path(__file__).resolve().parent
    evaluator_dir = resources_dir / "evaluator"
    sim_root = resources_dir / "cant-be-late-simulator"

    data_root = sim_root / "data"
    if not data_root.exists():
        raise RuntimeError(
            "Dataset not found. Please ensure real_traces.tar.gz has been extracted under "
            "resources/cant-be-late-simulator/data/."
        )

    sys.path.insert(0, str(evaluator_dir))
    sys.path.insert(0, str(resources_dir))
    try:
        import evaluator_real30  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import evaluator_real30: {e}") from e

    try:
        import sys as _sys  # noqa
        _sys.path.insert(0, str(sim_root))
        from sky_spot.utils import DEVICE_COSTS, COST_K  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import simulator pricing utils: {e}") from e

    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Error reading artifact {artifact_path}: {e}") from e

    program_path = None
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            if obj.get("program_path"):
                pp = obj["program_path"].strip()
                program_path = pp if os.path.isabs(pp) else os.path.abspath(pp)
            elif obj.get("code"):
                code = obj["code"]
                program_path = os.path.abspath("./solution_env/_submitted_program.py")
                os.makedirs(os.path.dirname(program_path), exist_ok=True)
                with open(program_path, "w", encoding="utf-8") as out:
                    out.write(code)
    except Exception:
        pass

    if program_path is None and os.path.exists(content):
        program_path = content if os.path.isabs(content) else os.path.abspath(content)

    if program_path is None:
        program_path = os.path.abspath("./solution_env/_submitted_program.py")
        os.makedirs(os.path.dirname(program_path), exist_ok=True)
        with open(program_path, "w", encoding="utf-8") as out:
            out.write(content)

    try:
        result = evaluator_real30.evaluate_stage2(program_path)
    except Exception as e:
        raise RuntimeError(f"Error running evaluator_real30: {e}") from e

    if isinstance(result, dict):
        metrics = result.get("metrics", {})
        artifacts = result.get("artifacts", {})
    else:
        metrics = getattr(result, "metrics", {})
        artifacts = getattr(result, "artifacts", {})

    avg_cost = float(metrics.get("avg_cost", 0.0))
    scen_json = artifacts.get("scenario_stats_json")

    if not scen_json:
        return {"score": 0, "avg_cost": avg_cost, "od_anchor": None, "spot_anchor": None}

    try:
        scenario_stats = json.loads(scen_json)
    except Exception as e:
        raise RuntimeError(f"Error parsing scenario_stats_json: {e}") from e

    total_weight = 0.0
    od_sum = 0.0
    spot_sum = 0.0

    for key, item in scenario_stats.items():
        env_path = item.get("env_path", "")
        duration = float(item.get("duration", 0))
        count = float(item.get("count", 0))
        if duration <= 0 or count <= 0 or not env_path:
            continue

        parts = env_path.split("_")
        device = None
        if len(parts) >= 3:
            device = f"{parts[-2]}_{parts[-1]}"
        if device not in DEVICE_COSTS:
            for cand in DEVICE_COSTS.keys():
                if cand in env_path:
                    device = cand
                    break
        od_price = DEVICE_COSTS.get(device)
        if od_price is None:
            continue
        spot_price = float(od_price) / float(COST_K)
        od_sum += float(od_price) * duration * count
        spot_sum += float(spot_price) * duration * count
        total_weight += count

    if total_weight <= 0 or od_sum <= 0:
        return {"score": 0, "avg_cost": avg_cost, "od_anchor": None, "spot_anchor": None}

    od_anchor = od_sum / total_weight
    spot_anchor = spot_sum / total_weight
    denom = (od_anchor - spot_anchor)
    if denom <= 1e-9:
        return {"score": 0, "avg_cost": avg_cost, "od_anchor": od_anchor, "spot_anchor": spot_anchor}

    norm = (od_anchor - avg_cost) / denom
    norm = max(0.0, min(1.0, norm))
    score = round(norm * 100)
    return {
        "score": score,
        "avg_cost": avg_cost,
        "od_anchor": od_anchor,
        "spot_anchor": spot_anchor,
        "scenario_count": total_weight,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: run_evaluator_real30.py <output_ans_file>", file=sys.stderr)
        sys.exit(1)

    try:
        payload = evaluate_artifact(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Score: 0")
        sys.exit(1)

    print(f"Score: {payload.get('score', 0)}")


if __name__ == "__main__":
    main()
