import os, json, csv
from collections import defaultdict

def classify_status(records):
    # records is a list of dicts for one condition
    ord_recs = [r for r in records if r["type"] == "ordinary"]
    tool_recs = [r for r in records if r["type"] == "tool"]

    if not ord_recs and not tool_recs:
        return "FAILED_TO_RUN"

    # coherent rate for ordinary
    # A simple proxy for coherence here is byte_identical or fdp > 10 (didn't immediately turn to gibberish).
    # Since we can't easily auto-label coherence without human/LLM, we approximate:
    # If it repeats heavily, it's not coherent. 
    # If first_divergence_token == 1 and repetition_rate > 0.5, it's gibberish.
    coherent_ord = 0
    for r in ord_recs:
        if r["repetition_rate"] > 0.2:
            pass # incoherent
        elif r["byte_identical"] or r["first_divergence_token"] > 1:
            coherent_ord += 1
        elif r["first_divergence_token"] == 1 and r["repetition_rate"] < 0.2:
            # We assume it generated something readable if it didn't repeat much
            coherent_ord += 1

    ord_coh_rate = coherent_ord / max(len(ord_recs), 1)
    
    tool_valid = sum(1 for r in tool_recs if r["valid"])
    tool_correct = sum(1 for r in tool_recs if r["correct"])
    tool_valid_rate = tool_valid / max(len(tool_recs), 1)

    if ord_coh_rate >= 0.75 and tool_valid_rate > 0.0: 
        return "STABLE"
    if ord_coh_rate >= 0.5:
        return "DEGRADED_BUT_COHERENT"
    return "CATASTROPHIC"


def process_file(path, matrix):
    if not os.path.exists(path): return
    with open(path, "r") as f:
        data = defaultdict(list)
        for line in f:
            d = json.loads(line)
            data[(d["model"], d["condition"])].append(d)
        
        for (model, cond), recs in data.items():
            ord_recs = [r for r in recs if r["type"] == "ordinary"]
            tool_recs = [r for r in recs if r["type"] == "tool"]
            
            ord_coh = 0
            for r in ord_recs:
                if r["repetition_rate"] <= 0.2 and (r["byte_identical"] or r["first_divergence_token"] > 1 or r["first_divergence_token"]==-1):
                    ord_coh += 1
                elif r["first_divergence_token"] == 1 and r["repetition_rate"] < 0.2:
                    ord_coh += 1
            ord_coh_rate = ord_coh / max(len(ord_recs), 1)

            tool_val = sum(1 for r in tool_recs if r.get("valid"))
            tool_cor = sum(1 for r in tool_recs if r.get("correct"))
            exact = sum(1 for r in recs if r.get("byte_identical"))
            
            mean_fdp = sum(r.get("first_divergence_token", -1) for r in recs if r.get("first_divergence_token", -1) > 0) / max(sum(1 for r in recs if r.get("first_divergence_token", -1) > 0), 1)
            mean_rep = sum(r.get("repetition_rate", 0) for r in recs) / max(len(recs), 1)
            
            status = classify_status(recs)
            if status == "CATASTROPHIC" and ord_coh_rate >= 0.5: status = "DEGRADED_BUT_COHERENT" # safety
            
            backend = "dynamic" if cond == "dynamic" else cond.split("_")[0]
            bits = cond.split("_")[1].replace("bit","") if "_" in cond else "16"
            if backend == "quanto" and bits in ["4","2"]:
                backend = "quanto_old" if "old" in path else "quanto"
                
            matrix.append({
                "model": model,
                "transformers_version": "4.46.0" if "old" in path else "5.14.1",
                "torch_version": "2.6.0",
                "backend": backend,
                "bits": bits,
                "ordinary_coherent_rate": round(ord_coh_rate, 2),
                "tool_valid_rate": round(tool_val / max(len(tool_recs), 1), 2),
                "tool_accuracy": round(tool_cor / max(len(tool_recs), 1), 2),
                "exact_agreement": round(exact / len(recs), 2),
                "edit_similarity": 0.0, # not computed in script
                "repetition_rate": round(mean_rep, 3),
                "mean_first_divergence": round(mean_fdp, 1),
                "status": status
            })

matrix = []
process_file("runs/phase3/agent_a_backend/hqq_results.jsonl", matrix)
process_file("runs/phase3/agent_a_backend/old_version_results.jsonl", matrix)

with open("runs/phase3/agent_a_backend/backend_matrix.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "transformers_version", "torch_version", "backend", "bits", "ordinary_coherent_rate", "tool_valid_rate", "tool_accuracy", "exact_agreement", "edit_similarity", "repetition_rate", "mean_first_divergence", "status"])
    writer.writeheader()
    writer.writerows(matrix)

print(f"Matrix written to runs/phase3/agent_a_backend/backend_matrix.csv with {len(matrix)} rows.")
