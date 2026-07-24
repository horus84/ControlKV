import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats

def compute_bootstrap_ci(
    data: List[float],
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute 95% bootstrap confidence interval over example scores.

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    if not data:
        return 0.0, 0.0, 0.0

    arr = np.array(data, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) < 2 or np.all(arr == arr[0]):
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstraps)
    n = len(arr)

    for i in range(num_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, alpha / 2.0 * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))

    return mean_val, ci_lower, ci_upper

def compute_mcnemar_test(
    b_full: List[bool],
    b_cand: List[bool]
) -> Dict[str, Any]:
    """Compute McNemar's test for paired binary decisions.

    Contingency table:
                Cand Correct  Cand Incorrect
    Full Correct      b           c
    Full Incorrect    a           d

    b: Both correct
    c: Full correct, Cand incorrect (disagreement 1)
    a: Full incorrect, Cand correct (disagreement 2)
    d: Both incorrect
    """
    if len(b_full) != len(b_cand):
        raise ValueError("Inputs must have identical length")

    c = sum(1 for f, k in zip(b_full, b_cand) if f and not k)
    a = sum(1 for f, k in zip(b_full, b_cand) if not f and k)
    b = sum(1 for f, k in zip(b_full, b_cand) if f and k)
    d = sum(1 for f, k in zip(b_full, b_cand) if not f and not k)

    total_disagreements = a + c
    if total_disagreements == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # McNemar's test with continuity correction
        statistic = float((abs(a - c) - 1.0)**2 / (a + c)) if (a + c) > 0 else 0.0
        p_value = float(stats.distributions.chi2.sf(statistic, 1))

    return {
        "contingency_table": {"a_both_correct": b, "c_full_only": c, "b_cand_only": a, "d_both_incorrect": d},
        "statistic": statistic,
        "p_value": p_value,
        "significant_p05": p_value < 0.05
    }

def compute_median_latency(latencies: List[float]) -> float:
    """Compute median latency across repetitions."""
    if not latencies:
        return 0.0
    return float(np.median(latencies))
