import os, json, csv
from collections import defaultdict
from math import sqrt

def wilson_score_interval(successes, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = 1.96 # for 95%
    p = successes / n
    denominator = 1 + z**2/n
    centre = p + z**2 / (2*n)
    spread = z * sqrt((p*(1-p)/n) + (z**2/(4*n**2)))
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return max(0.0, lower), min(1.0, upper)

def main():
    res_file = "runs/phase3/agent_c_benchmark/benchmark_results.jsonl"
    if not os.path.exists(res_file):
        print("No benchmark results found.")
        return

    data = defaultdict(lambda: {"tool_total": 0, "tool_valid": 0, "tool_correct": 0, "ord_total": 0, "ord_coh": 0})
    
    with open(res_file, "r") as f:
        for line in f:
            d = json.loads(line)
            key = (d["model"], d["ctx"], d["cond"])
            if d["type"] == "tool":
                data[key]["tool_total"] += 1
                if d["valid"]: data[key]["tool_valid"] += 1
                if d["correct"]: data[key]["tool_correct"] += 1
            else:
                data[key]["ord_total"] += 1
                if d["rep"] < 0.2: data[key]["ord_coh"] += 1

    out_csv = "runs/phase3/agent_c_benchmark/benchmark_stats.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "context", "condition", 
            "tool_valid_rate", "tool_valid_lower", "tool_valid_upper",
            "tool_correct_rate", "tool_correct_lower", "tool_correct_upper",
            "ordinary_coherent_rate"
        ])
        writer.writeheader()
        
        for (model, ctx, cond), stats in data.items():
            t_tot = max(stats["tool_total"], 1)
            t_val = stats["tool_valid"]
            t_cor = stats["tool_correct"]
            o_tot = max(stats["ord_total"], 1)
            o_coh = stats["ord_coh"]
            
            vl, vu = wilson_score_interval(t_val, t_tot)
            cl, cu = wilson_score_interval(t_cor, t_tot)
            
            writer.writerow({
                "model": model, "context": ctx, "condition": cond,
                "tool_valid_rate": f"{t_val/t_tot:.3f}",
                "tool_valid_lower": f"{vl:.3f}", "tool_valid_upper": f"{vu:.3f}",
                "tool_correct_rate": f"{t_cor/t_tot:.3f}",
                "tool_correct_lower": f"{cl:.3f}", "tool_correct_upper": f"{cu:.3f}",
                "ordinary_coherent_rate": f"{o_coh/o_tot:.3f}"
            })
            
    print(f"Stats written to {out_csv}")

if __name__ == "__main__":
    main()
