import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def plot_context_retention(
    results_by_context: Dict[int, Dict[str, float]],
    output_path: str = "paper/figures/retention_vs_context.pdf"
):
    """Plot correctness retention rate across context lengths for cache conditions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    contexts = sorted(results_by_context.keys())
    conditions = ["full", "offloaded", "quanto_int4", "quanto_int2"]
    colors = {"full": "black", "offloaded": "blue", "quanto_int4": "orange", "quanto_int2": "red"}
    styles = {"full": "-", "offloaded": "--", "quanto_int4": "-.", "quanto_int2": ":"}

    for cond in conditions:
        vals = [results_by_context[ctx].get(cond, 0.0) * 100 for ctx in contexts]
        ax.plot(contexts, vals, label=cond, color=colors.get(cond, "gray"), linestyle=styles.get(cond, "-"), marker="o")

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Decision Retention Rate (%)")
    ax.set_title("Tool Decision Retention vs. Context Length")
    ax.set_xticks(contexts)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
