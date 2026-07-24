from typing import List, Dict, Any
from controlkv.metrics.statistics import compute_bootstrap_ci

def compute_group_confidence_intervals(
    scores_by_group: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for grouped metric scores."""
    results = {}
    for group_name, scores in scores_by_group.items():
        mean_val, ci_low, ci_high = compute_bootstrap_ci(scores)
        results[group_name] = {
            "mean": round(mean_val, 4),
            "ci_lower": round(ci_low, 4),
            "ci_upper": round(ci_high, 4)
        }
    return results
