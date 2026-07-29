from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_residual_experiment(selection_file, evaluation_dir, output_dir):
    selection = json.loads(Path(selection_file).read_text(encoding="utf-8"))
    validation = selection["selected_metrics"]
    inference = pd.read_csv(Path(evaluation_dir) / "evaluation_summary.csv", encoding="utf-8-sig").iloc[0]
    rows = [{
        "attempt": 1, "status": "selected", "change": "Residual diffusion; softmax tau=0.1; dropout=0.1; clip q0.5-q99.5",
        "epochs": 200, "softmax_temperature": 0.1, "dropout": 0.1,
        "clip_quantiles": "0.5-99.5", "validation_MAE": validation["MAE"],
        "validation_MSE": validation["MSE"], "validation_RMSE": validation["RMSE"],
        "validation_coverage": validation["coverage_p10_p90"],
        "validation_interval_width": validation["mean_interval_width"],
        "quality_gate_passed": selection["quality_gate"]["passed"],
    }, {
        "attempt": 2, "status": "skipped", "change": "Temperature selection from 0.05/0.2",
        "skip_reason": "attempt 1 passed all validation gates",
    }, {
        "attempt": 3, "status": "skipped", "change": "dropout=0.05; clip q2.5-q97.5",
        "skip_reason": "attempt 1 passed all validation gates",
    }]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "attempts": rows, "selected_attempt": 1,
        "validation_quality_gate": selection["quality_gate"],
        "inference": {key: float(inference[key]) for key in (
            "MAE", "MSE", "RMSE", "coverage_p10_p90", "mean_interval_width"
        )},
        "inference_interval_width_below_2": bool(float(inference["mean_interval_width"]) < 2.0),
        "inference_is_final_holdout": True,
    }
    (output / "experiment_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
