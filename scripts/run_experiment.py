#!/usr/bin/env python3
"""
Phase 5 Main Experiment Matrix Runner Script
Executes full experimental matrix across models, cache conditions, and context lengths.
Creates immutable run directories in runs/<timestamp>_<model>_<cache>_<context>/
"""
import os
import sys
import json
import csv
import time
import yaml  # type: ignore
import torch
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

from controlkv.utils.reproducibility import set_seed
from controlkv.utils.environment import capture_environment_metadata
from controlkv.models.loader import load_model_and_tokenizer
from controlkv.models.tool_template import format_qwen_tool_prompt
from controlkv.cache.factory import get_kv_cache
from controlkv.cache.accounting import compute_logical_kv_cache_bytes
from controlkv.benchmarks.context_scaling import scale_context_to_target
from controlkv.parsing.tool_calls import extract_tool_calls
from controlkv.metrics.actions import evaluate_ground_truth_correctness, evaluate_agreement_against_full_cache
from controlkv.metrics.surface import compute_surface_metrics
from controlkv.metrics.statistics import compute_bootstrap_ci
from controlkv.profiling.cuda_memory import CUDAMemoryTracker
from controlkv.profiling.timing import CUDATimer

def run_single_matrix_cell(
    model: Any,
    tokenizer: Any,
    subset_examples: Any,
    cache_condition: str,
    target_context_len: int,
    config: Dict[str, Any],
    timestamp_str: str,
    baseline_predictions_dict: Optional[Dict[str, Any]] = None
):
    model_name_clean = config["model_name_or_path"].split("/")[-1].replace(".", "")
    run_dir_name = f"{timestamp_str}_{model_name_clean}_{cache_condition}_ctx{target_context_len}"
    run_dir = os.path.join(config.get("save_dir", "runs"), run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Launching Run Cell: {run_dir_name}")
    print(f"Cache: {cache_condition} | Context: ~{target_context_len} tokens | Examples: {len(subset_examples)}")
    print(f"=======================================================")

    set_seed(config.get("seed", 42))
    memory_tracker = CUDAMemoryTracker(config["device"])
    timer = CUDATimer(config["device"])

    num_layers = getattr(model.config, "num_hidden_layers", 28)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    hidden_sz = getattr(model.config, "hidden_size", 1792)
    num_attn_heads = getattr(model.config, "num_attention_heads", 14)
    head_dim = hidden_sz // num_attn_heads

    predictions = []
    failures = []
    memory_records = []
    timing_records = []

    pred_jsonl_path = os.path.join(run_dir, "predictions.jsonl")
    fail_jsonl_path = os.path.join(run_dir, "failures.jsonl")
    mem_csv_path = os.path.join(run_dir, "memory.csv")
    time_csv_path = os.path.join(run_dir, "timing.csv")

    with open(pred_jsonl_path, "w", encoding="utf-8") as f_pred, \
         open(fail_jsonl_path, "w", encoding="utf-8") as f_fail, \
         open(mem_csv_path, "w", newline="", encoding="utf-8") as f_mem, \
         open(time_csv_path, "w", newline="", encoding="utf-8") as f_time:

        mem_writer = csv.writer(f_mem)
        mem_writer.writerow(["id", "peak_allocated_bytes", "peak_reserved_bytes", "logical_kv_cache_bytes"])
        
        time_writer = csv.writer(f_time)
        time_writer.writerow(["id", "total_latency_seconds", "generated_tokens", "tokens_per_second"])

        for idx, item in enumerate(subset_examples):
            item_id = item["id"]
            try:
                memory_tracker.reset()
                
                # 1. Base format
                base_prompt = format_qwen_tool_prompt(item["tools"], item["question"], tokenizer)
                
                # 2. Context scaling
                scaled_prompt, distractor_text, actual_ctx_len = scale_context_to_target(
                    base_prompt, target_context_len, tokenizer
                )

                input_ids = tokenizer.encode(scaled_prompt, return_tensors="pt").to(config["device"])
                
                cache_obj, cache_kwargs = get_kv_cache(cache_condition, model.config)

                # Warmup repetitions if requested
                warmup_runs = config.get("warmup_runs", 0) if idx == 0 else 0
                for _ in range(warmup_runs):
                    model.generate(
                        input_ids,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        **cache_kwargs
                    )
                    memory_tracker.reset()

                # Generation timing
                def gen_fn():
                    return model.generate(
                        input_ids,
                        max_new_tokens=config.get("max_new_tokens", 128),
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        **cache_kwargs
                    )

                gen_out, total_latency = timer.time_execution(gen_fn)
                output_tokens = gen_out[0][input_ids.shape[1]:].tolist()
                decoded_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                
                valid_struct, parsed_calls = extract_tool_calls(decoded_text)
                gt_metrics = evaluate_ground_truth_correctness(parsed_calls, item["ground_truth"])
                
                # Agreement against full cache if available
                agreement_metrics = {}
                if baseline_predictions_dict and item_id in baseline_predictions_dict:
                    full_parsed = baseline_predictions_dict[item_id]["parsed_calls"]
                    full_text = baseline_predictions_dict[item_id]["decoded_text"]
                    full_tokens = baseline_predictions_dict[item_id]["generated_token_ids"]
                    
                    agreement_metrics = evaluate_agreement_against_full_cache(parsed_calls, full_parsed)
                    surface_metrics = compute_surface_metrics(decoded_text, full_text, output_tokens, full_tokens)
                    agreement_metrics.update(surface_metrics)
                else:
                    agreement_metrics = {
                        "tool_name_agreement": True,
                        "arg_exact_agreement": True,
                        "complete_action_agreement": True,
                        "exact_output_string_agreement": True,
                        "normalized_edit_similarity": 1.0,
                        "exact_token_agreement": True,
                        "token_agreement_rate": 1.0
                    }

                mem_stats = memory_tracker.get_metrics()
                logical_bytes = compute_logical_kv_cache_bytes(
                    num_layers, num_kv_heads, head_dim, input_ids.shape[1] + len(output_tokens), cache_condition
                )
                
                tokens_per_sec = len(output_tokens) / total_latency if total_latency > 0 else 0.0

                record = {
                    "id": item_id,
                    "model_identifier": config["model_name_or_path"],
                    "cache_condition": cache_condition,
                    "target_context_length": target_context_len,
                    "actual_context_length": input_ids.shape[1],
                    "category": item["category"],
                    "tools": item["tools"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "generated_token_ids": output_tokens,
                    "decoded_text": decoded_text,
                    "valid_structured_output": valid_struct,
                    "parsed_calls": parsed_calls,
                    "ground_truth_correctness": gt_metrics,
                    "agreement_metrics": agreement_metrics,
                    "latency_seconds": total_latency,
                    "tokens_per_second": tokens_per_sec,
                    "peak_cuda_allocated_bytes": mem_stats["peak_allocated_bytes"],
                    "peak_cuda_reserved_bytes": mem_stats["peak_reserved_bytes"],
                    "logical_kv_cache_bytes": logical_bytes
                }

                predictions.append(record)
                f_pred.write(json.dumps(record) + "\n")
                f_pred.flush()

                mem_writer.writerow([item_id, mem_stats["peak_allocated_bytes"], mem_stats["peak_reserved_bytes"], logical_bytes])
                time_writer.writerow([item_id, total_latency, len(output_tokens), tokens_per_sec])

            except Exception as e:
                err_rec = {
                    "id": item_id,
                    "model_identifier": config["model_name_or_path"],
                    "cache_condition": cache_condition,
                    "context_length": target_context_len,
                    "error": str(e)
                }
                failures.append(err_rec)
                f_fail.write(json.dumps(err_rec) + "\n")
                f_fail.flush()

    # Save summary metadata and metrics
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(capture_environment_metadata(), f, indent=2)

    mean_alloc = sum(p["peak_cuda_allocated_bytes"] for p in predictions) / max(1, len(predictions))
    mean_res = sum(p["peak_cuda_reserved_bytes"] for p in predictions) / max(1, len(predictions))
    mean_kv = sum(p["logical_kv_cache_bytes"] for p in predictions) / max(1, len(predictions))

    summary_metrics = {
        "run_id": run_dir_name,
        "model_identifier": config["model_name_or_path"],
        "cache_condition": cache_condition,
        "target_context_length": target_context_len,
        "total_examples": len(subset_examples),
        "completed_examples": len(predictions),
        "failed_examples": len(failures),
        "accuracy_mean": acc_mean,
        "accuracy_ci_95": [acc_low, acc_high],
        "valid_structured_rate": sum(1 for p in predictions if p["valid_structured_output"]) / max(1, len(predictions)),
        "mean_latency_seconds": sum(p["latency_seconds"] for p in predictions) / max(1, len(predictions)),
        "mean_tokens_per_second": sum(p["tokens_per_second"] for p in predictions) / max(1, len(predictions)),
        "mean_peak_cuda_allocated_bytes": mean_alloc,
        "mean_peak_cuda_reserved_bytes": mean_res,
        "mean_logical_kv_cache_bytes": mean_kv
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Completed Cell {run_dir_name}: Accuracy = {acc_mean*100:.2f}% [{acc_low*100:.2f}%, {acc_high*100:.2f}%]")
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load 150-example subset
    with open("results/dataset_subset.json", "r", encoding="utf-8") as f:
        subset_manifest = json.load(f)
    examples = subset_manifest["examples"][:config.get("subset_size", 150)]

    print(f"=== Starting Main Experiment Runner for Model: {config['model_name_or_path']} ===")
    print(f"Total Examples: {len(examples)} | Cache Modes: {config['cache_conditions']} | Contexts: {config['context_lengths']}")

    model, tokenizer = load_model_and_tokenizer(
        config["model_name_or_path"],
        device=config["device"],
        torch_dtype=config["torch_dtype"]
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run matrix: full cache first for baselines
    for ctx_len in config["context_lengths"]:
        # 1. Full cache baseline
        full_preds = run_single_matrix_cell(
            model, tokenizer, examples, "full", ctx_len, config, timestamp_str
        )
        full_preds_dict = {p["id"]: p for p in full_preds}

        # 2. Other cache conditions against full baseline
        for cache_cond in config["cache_conditions"]:
            if cache_cond == "full":
                continue
            run_single_matrix_cell(
                model, tokenizer, examples, cache_cond, ctx_len, config, timestamp_str, full_preds_dict
            )

    print("\nMain Experiment Run Matrix Completed Successfully!")

if __name__ == "__main__":
    main()
