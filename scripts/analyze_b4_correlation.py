import os, json, csv
from collections import defaultdict
import numpy as np

def main():
    res_file = "runs/phase3/agent_b_localization/b_ablation_results.jsonl"
    err_file = "runs/phase3/agent_b_localization/layer_errors.csv"
    
    if not os.path.exists(res_file) or not os.path.exists(err_file):
        print("Missing data files.")
        return

    # Load block ablation results
    results = defaultdict(lambda: {"valid": 0, "total": 0})
    with open(res_file, "r") as f:
        for line in f:
            d = json.loads(line)
            if d["type"] == "tool" and "block" in d["cond"]:
                results[d["cond"]]["total"] += 1
                if d["valid"]: results[d["cond"]]["valid"] += 1
                
    print("Block Ablation Degradation (lower valid = more degradation):")
    block_degradation = {}
    for cond, stats in results.items():
        rate = stats["valid"] / max(stats["total"], 1)
        print(f"  {cond}: {rate:.2f} valid rate")
        block_degradation[cond] = rate

    # Load layer errors
    layer_errors = []
    with open(err_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_errors.append({
                "layer": int(row["layer"]),
                "type": row["type"],
                "mae": float(row["mae"]),
                "mse": float(row["mse"])
            })
            
    if not layer_errors:
        print("No layer errors found.")
        pass

    # Aggregate layer errors by block
    # blocks: 0-6, 7-13, 14-20, 21-27
    b_sizes = [(0,6), (7,13), (14,20), (21,27)]
    block_mae = {}
    print("\nLayer Errors (MSE) by Block:")
    for start, end in b_sizes:
        errs = [e["mse"] for e in layer_errors if start <= e["layer"] <= end]
        avg_mse = sum(errs)/max(len(errs), 1)
        print(f"  block_{start}_{end}: {avg_mse:.6f}")
        cond = f"block_{start}_{end}_int4"
        block_mae[cond] = avg_mse
        
    print("\nK/V Ablation Results:")
    kv_res = defaultdict(lambda: {"valid": 0, "total": 0})
    with open(res_file, "r") as f:
        for line in f:
            d = json.loads(line)
            if d["type"] == "tool" and ("only" in d["cond"]):
                kv_res[d["cond"]]["total"] += 1
                if d["valid"]: kv_res[d["cond"]]["valid"] += 1
    
    for cond, stats in kv_res.items():
        rate = stats["valid"] / max(stats["total"], 1)
        print(f"  {cond}: {rate:.2f} valid rate")

if __name__ == "__main__":
    main()
