#!/usr/bin/env python3
"""
Phase 2 & Phase 3 Smoke Test Script
Executes 20 examples on Qwen2.5-1.5B-Instruct across DynamicCache, OffloadedCache,
Quanto int4, and Quanto int2. Verifies full-cache determinism and offloaded equivalence.
"""
import os
import sys
import json
import time
import yaml
import torch
import argparse
from datetime import datetime

from controlkv.utils.reproducibility import set_seed
from controlkv.utils.environment import capture_environment_metadata
from controlkv.models.loader import load_model_and_tokenizer
from controlkv.models.tool_template import format_qwen_tool_prompt
from controlkv.cache.factory import get_kv_cache
from controlkv.cache.accounting import compute_logical_kv_cache_bytes
from controlkv.parsing.tool_calls import extract_tool_calls
from controlkv.metrics.actions import evaluate_ground_truth_correctness, evaluate_agreement_against_full_cache
from controlkv.metrics.surface import compute_surface_metrics
from controlkv.profiling.cuda_memory import CUDAMemoryTracker
from controlkv.profiling.timing import CUDATimer

def run_smoke_pass(model, tokenizer, dataset, cache_condition, config):
    set_seed(config.get("seed", 42))
    memory_tracker = CUDAMemoryTracker(config["device"])
    timer = CUDATimer(config["device"])
    
    predictions = []
    failures = []
    
    num_layers = getattr(model.config, "num_hidden_layers", 28)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    head_dim = getattr(model.config, "hidden_size", 1792) // getattr(model.config, "num_attention_heads", 14)

    for item in dataset:
        try:
            memory_tracker.reset()
            prompt = format_qwen_tool_prompt(item["tools"], item["question"], tokenizer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config["device"])
            
            cache_obj, cache_kwargs = get_kv_cache(cache_condition, model.config)
            
            # Prefill & Decode timing
            def generate_fn():
                return model.generate(
                    input_ids,
                    max_new_tokens=config.get("max_new_tokens", 128),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    **cache_kwargs
                )
                
            gen_out, total_latency = timer.time_execution(generate_fn)
            output_tokens = gen_out[0][input_ids.shape[1]:].tolist()
            decoded_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            valid_struct, parsed_calls = extract_tool_calls(decoded_text)
            gt_metrics = evaluate_ground_truth_correctness(parsed_calls, item["ground_truth"])
            mem_stats = memory_tracker.get_metrics()
            logical_bytes = compute_logical_kv_cache_bytes(
                num_layers, num_kv_heads, head_dim, input_ids.shape[1] + len(output_tokens), cache_condition
            )

            rec = {
                "id": item["id"],
                "category": item["category"],
                "cache_condition": cache_condition,
                "input_tokens_count": input_ids.shape[1],
                "output_tokens_count": len(output_tokens),
                "generated_token_ids": output_tokens,
                "decoded_text": decoded_text,
                "valid_structured_output": valid_struct,
                "parsed_calls": parsed_calls,
                "ground_truth_correctness": gt_metrics,
                "latency_seconds": total_latency,
                "peak_cuda_allocated_bytes": mem_stats["peak_allocated_bytes"],
                "peak_cuda_reserved_bytes": mem_stats["peak_reserved_bytes"],
                "logical_kv_cache_bytes": logical_bytes
            }
            predictions.append(rec)
            
        except Exception as e:
            failures.append({
                "id": item.get("id", "unknown"),
                "cache_condition": cache_condition,
                "error": str(e)
            })

    return predictions, failures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"=== Starting Phase 2 & Phase 3 Smoke Test ===")
    print(f"Model: {config['model_name_or_path']}")
    print(f"Device: {config['device']}")

    # Load benchmark dataset
    with open("results/dataset_subset.json", "r", encoding="utf-8") as f:
        subset_manifest = json.load(f)
    dataset_20 = subset_manifest["examples"][:20]

    model, tokenizer = load_model_and_tokenizer(
        config["model_name_or_path"],
        device=config["device"],
        torch_dtype=config["torch_dtype"]
    )

    # Phase 2: DynamicCache Determinism Check (Run 1 vs Run 2)
    print("\n--- Running Phase 2: Full-Cache Run 1 ---")
    preds_full_1, fail_full_1 = run_smoke_pass(model, tokenizer, dataset_20, "full", config)
    
    print("--- Running Phase 2: Full-Cache Run 2 (Determinism Verification) ---")
    preds_full_2, fail_full_2 = run_smoke_pass(model, tokenizer, dataset_20, "full", config)

    # Assert token identity
    tokens_1 = [p["generated_token_ids"] for p in preds_full_1]
    tokens_2 = [p["generated_token_ids"] for p in preds_full_2]

    assert len(tokens_1) == len(tokens_2) == 20, "Full cache runs failed to generate 20 records"
    for i in range(20):
        if tokens_1[i] != tokens_2[i]:
            print(f"FATAL: Determinism check failed at example index {i} ({dataset_20[i]['id']})!")
            print(f"Run 1 tokens: {tokens_1[i]}")
            print(f"Run 2 tokens: {tokens_2[i]}")
            sys.exit(1)

    print("SUCCESS: Phase 2 Full-Cache Determinism Verification Passed! (20/20 Token-Identical)")

    # Phase 3: OffloadedCache Equivalence Check
    print("\n--- Running Phase 3: OffloadedCache Control Run ---")
    preds_offloaded, fail_off = run_smoke_pass(model, tokenizer, dataset_20, "offloaded", config)

    tokens_off = [p["generated_token_ids"] for p in preds_offloaded]
    for i in range(20):
        if tokens_1[i] != tokens_off[i]:
            print(f"FATAL: OffloadedCache output equivalence failed at example index {i} ({dataset_20[i]['id']})!")
            sys.exit(1)

    print("SUCCESS: Phase 3 OffloadedCache Lossless Control Equivalence Passed! (20/20 Token-Identical to DynamicCache)")

    # Quantized Caches (Quanto int4 & Quanto int2)
    print("\n--- Running Phase 3: QuantizedCache (Quanto int4) ---")
    preds_int4, fail_int4 = run_smoke_pass(model, tokenizer, dataset_20, "quanto_int4", config)

    print("--- Running Phase 3: QuantizedCache (Quanto int2) ---")
    preds_int2, fail_int2 = run_smoke_pass(model, tokenizer, dataset_20, "quanto_int2", config)

    # Save Smoke Test Run Artifacts
    run_dir = os.path.join("runs", "smoke_test")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(capture_environment_metadata(), f, indent=2)

    summary_metrics = {
        "determinism_pass": True,
        "offloaded_equivalence_pass": True,
        "full_run1_count": len(preds_full_1),
        "full_run2_count": len(preds_full_2),
        "offloaded_count": len(preds_offloaded),
        "int4_count": len(preds_int4),
        "int2_count": len(preds_int2),
        "failures_count": len(fail_full_1) + len(fail_off) + len(fail_int4) + len(fail_int2)
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\nSmoke Test Completed Successfully! All artifacts saved to {run_dir}/")

if __name__ == "__main__":
    main()
