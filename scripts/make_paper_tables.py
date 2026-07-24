#!/usr/bin/env python3
"""
Phase 7 Paper Table Generator Script
Programmatically builds LaTeX tables from saved result JSON and CSV files.
Inserts [RESULT_PENDING] if results do not yet exist in results/ or runs/.
"""
import os
import json
import glob
import pandas as pd

def generate_main_results_table(output_path="paper/tables/main_results.tex"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    runs = glob.glob("runs/*/metrics.json")
    results_map = {}

    for r in runs:
        run_folder = os.path.dirname(r)
        with open(r, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # Read peak VRAM and logical KV from memory.csv if present
            mem_csv = os.path.join(run_folder, "memory.csv")
            if os.path.exists(mem_csv):
                try:
                    df_mem = pd.read_csv(mem_csv)
                    data["mean_peak_cuda_allocated_bytes"] = df_mem["peak_allocated_bytes"].mean()
                    data["mean_logical_kv_cache_bytes"] = df_mem["logical_kv_cache_bytes"].mean()
                except Exception:
                    pass

            key = (data.get("model_identifier", ""), data.get("cache_condition", ""), data.get("target_context_length", 512))
            results_map[key] = data

    latex_code = r"""\begin{table*}[t]
\centering
\small
\caption{Ground-truth tool decision accuracy (\%) and memory/latency trade-offs across KV-cache quantization conditions and context lengths.}
\label{tab:main_results}
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{Cache Condition} & \textbf{Ctx Len} & \textbf{Accuracy (95\% CI)} & \textbf{Token Agr (\%)} & \textbf{Peak VRAM (MB)} & \textbf{Logical KV (MB)} & \textbf{Latency (s)} \\
\midrule
"""

    models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    cache_conds = ["full", "offloaded", "quanto_int4", "quanto_int2"]
    contexts = [512, 1024, 2048]

    for model in models:
        model_short = model.split("/")[-1]
        for cond in cache_conds:
            for ctx in contexts:
                data = results_map.get((model, cond, ctx))
                if data and "accuracy_mean" in data:
                    acc = f"{data['accuracy_mean']*100:.1f}"
                    ci_low = f"{data['accuracy_ci_95'][0]*100:.1f}"
                    ci_high = f"{data['accuracy_ci_95'][1]*100:.1f}"
                    acc_str = f"{acc} [{ci_low}, {ci_high}]"
                    tok_agr = f"{data.get('valid_structured_rate', 0.0)*100:.1f}"
                    
                    alloc_bytes = data.get("mean_peak_cuda_allocated_bytes", data.get("peak_cuda_allocated_bytes", None))
                    peak_vram = f"{alloc_bytes / (1024**2):.1f}" if alloc_bytes is not None else "[PENDING]"
                    
                    kv_bytes = data.get("mean_logical_kv_cache_bytes", data.get("logical_kv_cache_bytes", None))
                    logical_kv = f"{kv_bytes / (1024**2):.1f}" if kv_bytes is not None else "[PENDING]"
                    
                    latency = f"{data.get('mean_latency_seconds', 0.0):.3f}"
                else:
                    acc_str = "[RESULT_PENDING]"
                    tok_agr = "[RESULT_PENDING]"
                    peak_vram = "[RESULT_PENDING]"
                    logical_kv = "[RESULT_PENDING]"
                    latency = "[RESULT_PENDING]"

                latex_code += f"{model_short} & {cond} & {ctx} & {acc_str} & {tok_agr} & {peak_vram} & {logical_kv} & {latency} \\\\\n"
            latex_code += r"\hline" + "\n"

    latex_code += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"Generated LaTeX table at {output_path}")

if __name__ == "__main__":
    generate_main_results_table()
