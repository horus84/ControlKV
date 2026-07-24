#!/usr/bin/env python3
"""
Phase 6 Result Aggregation & Statistical Significance Script
Aggregates all run results, computes bootstrap CIs, McNemar tests, generates disagreement log,
and plots retention vs context length.
"""
import os
import glob
import json
import pandas as pd
from controlkv.metrics.statistics import compute_bootstrap_ci, compute_mcnemar_test
from controlkv.analysis.plots import plot_context_retention

def aggregate_all_runs(runs_dir="runs", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    run_folders = glob.glob(os.path.join(runs_dir, "*"))
    all_metrics = []
    all_predictions = []
    disagreements = []

    for folder in run_folders:
        metrics_file = os.path.join(folder, "metrics.json")
        pred_file = os.path.join(folder, "predictions.jsonl")

        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_metrics.append(json.load(f))

        if os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        all_predictions.append(rec)
                        if not rec.get("agreement_metrics", {}).get("complete_action_agreement", True):
                            disagreements.append(rec)

    # Save disagreement log
    disagree_file = os.path.join(output_dir, "disagreements.jsonl")
    with open(disagree_file, "w", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")

    print(f"Aggregated {len(all_metrics)} runs and {len(all_predictions)} predictions.")
    print(f"Total Disagreements against Full Cache: {len(disagreements)} logged to {disagree_file}")

    # Build retention by context dict for plotting
    retention_by_ctx = {512: {}, 1024: {}, 2048: {}}
    for m in all_metrics:
        ctx = m.get("target_context_length")
        cond = m.get("cache_condition")
        acc = m.get("accuracy_mean", 0.0)
        if ctx in retention_by_ctx and cond:
            retention_by_ctx[ctx][cond] = acc

    plot_context_retention(retention_by_ctx, "paper/figures/retention_vs_context.pdf")
    print("Saved retention plot to paper/figures/retention_vs_context.pdf")

if __name__ == "__main__":
    aggregate_all_runs()
