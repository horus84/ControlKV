import json
import os
import glob
import pandas as pd
from typing import Dict, Any, List
from controlkv.metrics.statistics import compute_bootstrap_ci, compute_mcnemar_test

def aggregate_run_directories(runs_dir: str = "runs", output_dir: str = "results") -> Dict[str, Any]:
    """Aggregate metrics across all run directories in runs/ and compute stats."""
    os.makedirs(output_dir, exist_ok=True)
    
    run_folders = glob.glob(os.path.join(runs_dir, "*"))
    records = []

    for folder in run_folders:
        metrics_file = os.path.join(folder, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append(data)

    if not records:
        return {"num_runs": 0, "summary": {}}

    df = pd.DataFrame(records)
    summary_path = os.path.join(output_dir, "aggregated_summary.json")
    
    summary = {
        "num_runs": len(records),
        "models": df["model_identifier"].unique().tolist() if "model_identifier" in df else [],
        "cache_conditions": df["cache_condition"].unique().tolist() if "cache_condition" in df else [],
        "context_lengths": df["target_context_length"].unique().tolist() if "target_context_length" in df else [],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
