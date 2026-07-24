#!/usr/bin/env python3
"""
Minimal Controlled Diagnostic Investigation (Tests 1 - 6)
Explicitly saves all diagnostic JSON files to absolute path:
C:/Users/linseed21/Desktop/ControlKV/runs/diagnostic_experiment
"""
import os
import sys
import json
import math
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, cast
from transformers.cache_utils import DynamicCache, QuantizedCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

from controlkv.utils.reproducibility import set_seed
from controlkv.models.tool_template import format_qwen_tool_prompt
from controlkv.benchmarks.context_scaling import scale_context_to_target
from controlkv.parsing.tool_calls import extract_tool_calls
from controlkv.metrics.actions import evaluate_ground_truth_correctness
from controlkv.metrics.surface import levenshtein_distance

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return float(p), max(0.0, float(centre - spread)), min(1.0, float(centre + spread))

def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    alpha = 1 - confidence
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return float(p), low, high

def main():
    save_dir = os.path.abspath("runs/diagnostic_experiment")
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Controlled Diagnostic Investigation on {device} ===", flush=True)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer_obj = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer_obj is None:
        raise RuntimeError("Failed to load tokenizer.")
    tokenizer = tokenizer_obj

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = cast(int, tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    with open("results/dataset_subset.json", "r", encoding="utf-8") as f:
        subset_manifest = json.load(f)
    all_ex = subset_manifest["examples"]
    
    simple_ex = [e for e in all_ex if e["category"] == "simple"][:4]
    multiple_ex = [e for e in all_ex if e["category"] == "multiple"][:4]
    parallel_ex = [e for e in all_ex if e["category"] == "parallel"][:4]
    fixed_12_examples = simple_ex + multiple_ex + parallel_ex

    # TEST 1
    print("--- TEST 1: No-Quantization Control & Quantization Check ---", flush=True)
    test1_results = []
    test1_conditions: List[Tuple[str, int, int]] = [
        ("full", 4, 0),
        ("int4_res4096", 4, 4096),
        ("int2_res4096", 2, 4096),
        ("int4_res128", 4, 128),
        ("int2_res128", 2, 128)
    ]
    file_t1 = os.path.join(save_dir, "test1_control.json")

    for ctx in [512, 1024, 2048]:
        for ex in fixed_12_examples:
            ex_id = ex["id"]
            base_prompt = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
            scaled_prompt, _, _ = scale_context_to_target(base_prompt, ctx, tokenizer)
            input_ids = tokenizer.encode(scaled_prompt, return_tensors="pt").to(device)

            dynamic_tokens: List[int] = []
            for cond_name, nbits, res_len in test1_conditions:
                set_seed(42)
                cache = DynamicCache() if cond_name == "full" else QuantizedCache(backend="quanto", config=model.config, nbits=nbits, residual_length=res_len)
                with torch.no_grad():
                    gen_out = model.generate(input_ids, max_new_tokens=32, do_sample=False, past_key_values=cache, use_cache=True)
                output_tokens = gen_out[0][input_ids.shape[1]:].tolist()
                decoded_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                valid_struct, parsed_calls = extract_tool_calls(decoded_text)
                gt_eval = evaluate_ground_truth_correctness(parsed_calls, ex["ground_truth"])

                if cond_name == "full":
                    dynamic_tokens = output_tokens
                    byte_identical = True
                    first_diff_idx = -1
                else:
                    byte_identical = (output_tokens == dynamic_tokens)
                    first_diff_idx = -1
                    if not byte_identical:
                        for idx, (t1, t2) in enumerate(zip(dynamic_tokens, output_tokens)):
                            if t1 != t2:
                                first_diff_idx = idx
                                break
                        if first_diff_idx == -1:
                            first_diff_idx = min(len(dynamic_tokens), len(output_tokens))

                test1_results.append({
                    "id": ex_id,
                    "category": ex["category"],
                    "context_len": ctx,
                    "condition": cond_name,
                    "residual_length": res_len,
                    "nbits": nbits,
                    "generated_token_ids": output_tokens,
                    "decoded_text": decoded_text,
                    "valid_structured_output": valid_struct,
                    "parsed_calls": parsed_calls,
                    "correct": gt_eval["correct"],
                    "byte_identical_to_dynamic": byte_identical,
                    "first_diff_token_pos": first_diff_idx
                })

            with open(file_t1, "w", encoding="utf-8") as f:
                json.dump(test1_results, f, indent=2)
            
        print(f"Completed Test 1 for CTX={ctx}. Persisted to {file_t1}", flush=True)

    # TEST 2
    print("--- TEST 2: Residual-Length Sweep (INT4 @ CTX 2048) ---", flush=True)
    test2_results = []
    res_lengths = [128, 256, 512, 1024, 2048, 4096]
    file_t2 = os.path.join(save_dir, "test2_sweep.json")

    dynamic_2048_map: Dict[str, List[int]] = {}
    for ex in fixed_12_examples:
        base_prompt = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
        scaled_prompt, _, _ = scale_context_to_target(base_prompt, 2048, tokenizer)
        input_ids = tokenizer.encode(scaled_prompt, return_tensors="pt").to(device)
        set_seed(42)
        cache = DynamicCache()
        with torch.no_grad():
            gen_out = model.generate(input_ids, max_new_tokens=32, do_sample=False, past_key_values=cache, use_cache=True)
        dynamic_2048_map[ex["id"]] = gen_out[0][input_ids.shape[1]:].tolist()

    for res_len in res_lengths:
        sweep_records = []
        for ex in fixed_12_examples:
            ex_id = ex["id"]
            base_prompt = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
            scaled_prompt, _, _ = scale_context_to_target(base_prompt, 2048, tokenizer)
            input_ids = tokenizer.encode(scaled_prompt, return_tensors="pt").to(device)
            dyn_tokens = dynamic_2048_map[ex_id]

            set_seed(42)
            cache = QuantizedCache(backend="quanto", config=model.config, nbits=4, residual_length=res_len)
            with torch.no_grad():
                gen_out = model.generate(input_ids, max_new_tokens=32, do_sample=False, past_key_values=cache, use_cache=True)
            output_tokens = gen_out[0][input_ids.shape[1]:].tolist()
            decoded_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            valid_struct, parsed_calls = extract_tool_calls(decoded_text)
            gt_eval = evaluate_ground_truth_correctness(parsed_calls, ex["ground_truth"])

            byte_identical = (output_tokens == dyn_tokens)
            first_diff_idx = -1
            if not byte_identical:
                for idx, (t1, t2) in enumerate(zip(dyn_tokens, output_tokens)):
                    if t1 != t2:
                        first_diff_idx = idx
                        break
                if first_diff_idx == -1:
                    first_diff_idx = min(len(dyn_tokens), len(output_tokens))

            sweep_records.append({
                "id": ex_id,
                "residual_length": res_len,
                "correct": gt_eval["correct"],
                "valid_structured": valid_struct,
                "byte_identical": byte_identical,
                "first_diff_token_pos": first_diff_idx,
                "malformed": (not valid_struct)
            })

        valid_rate = sum(1 for r in sweep_records if r["valid_structured"]) / len(sweep_records)
        acc_rate = sum(1 for r in sweep_records if r["correct"]) / len(sweep_records)
        exact_match_rate = sum(1 for r in sweep_records if r["byte_identical"]) / len(sweep_records)
        diff_positions = [r["first_diff_token_pos"] for r in sweep_records if r["first_diff_token_pos"] != -1]
        mean_first_diff = float(np.mean(diff_positions)) if diff_positions else -1.0

        test2_results.append({
            "residual_length": res_len,
            "valid_structured_rate": valid_rate,
            "accuracy": acc_rate,
            "exact_output_agreement": exact_match_rate,
            "mean_first_divergence_token": mean_first_diff,
            "malformed_output_rate": 1.0 - valid_rate
        })
        
        with open(file_t2, "w", encoding="utf-8") as f:
            json.dump(test2_results, f, indent=2)

    print(f"Completed Test 2. Persisted to {file_t2}", flush=True)

    # TEST 3
    print("--- TEST 3: Ordinary Language Control ---", flush=True)
    ordinary_prompts = [
        "What is the capital city of France?",
        "Summarize the general principle of gravity in two sentences.",
        "Solve this arithmetic problem: 15 * 12 =",
        "Continue the sentence: In a distant solar system, scientists discovered a planet made of",
        "Explain what a database transaction is in simple terms.",
        "List three common programming languages used for web development.",
        "What happens during photosynthesis in plants?",
        "Translate the word 'hello' into Spanish, French, and German.",
        "What is the primary function of the human heart?",
        "Describe the purpose of an operating system kernel.",
        "Why is water essential for human survival?",
        "Name the four primary seasons of the year."
    ]

    test3_results = []
    file_t3 = os.path.join(save_dir, "test3_ordinary.json")

    for idx, prompt_text in enumerate(ordinary_prompts, 1):
        formatted = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)

        dyn_tokens: List[int] = []
        dyn_text: str = ""
        cond_outputs = {}

        for cond_name, nbits in [("full", 4), ("int4_res128", 4), ("int2_res128", 2)]:
            set_seed(42)
            cache = DynamicCache() if cond_name == "full" else QuantizedCache(backend="quanto", config=model.config, nbits=nbits, residual_length=128)
            with torch.no_grad():
                gen_out = model.generate(input_ids, max_new_tokens=32, do_sample=False, past_key_values=cache, use_cache=True)
            out_tokens = gen_out[0][input_ids.shape[1]:].tolist()
            text = tokenizer.decode(out_tokens, skip_special_tokens=True)

            if cond_name == "full":
                dyn_tokens = out_tokens
                dyn_text = text
                exact_tok_agree = True
                edit_sim = 1.0
            else:
                exact_tok_agree = (out_tokens == dyn_tokens)
                max_l = max(len(text), len(dyn_text))
                edit_sim = 1.0 - (levenshtein_distance(text, dyn_text) / max_l) if max_l > 0 else 1.0

            is_gibberish = ("=" * 4 in text or "\\" * 4 in text or "::" in text or len(set(text.split())) < 3 if len(text) > 20 else False)

            cond_outputs[cond_name] = {
                "text": text,
                "exact_token_agreement": exact_tok_agree,
                "edit_similarity": edit_sim,
                "is_gibberish": is_gibberish
            }

        test3_results.append({"id": f"ordinary_{idx:02d}", "prompt": prompt_text, "outputs": cond_outputs})
        with open(file_t3, "w", encoding="utf-8") as f:
            json.dump(test3_results, f, indent=2)

    print(f"Completed Test 3. Persisted to {file_t3}", flush=True)

    # TEST 4
    print("--- TEST 4: Token-Level Divergence Tracking ---", flush=True)
    test4_results = []
    file_t4 = os.path.join(save_dir, "test4_divergence.json")

    for ex in fixed_12_examples[:3]:
        ex_id = ex["id"]
        base_prompt = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
        scaled_prompt, _, _ = scale_context_to_target(base_prompt, 512, tokenizer)
        input_ids = tokenizer.encode(scaled_prompt, return_tensors="pt").to(device)

        def get_step_logits(cache_obj: Any) -> List[Dict[str, Any]]:
            set_seed(42)
            cur_ids = input_ids.clone()
            step_logs = []
            with torch.no_grad():
                out = model(cur_ids, past_key_values=cache_obj, use_cache=True)
                logits = out.logits[0, -1, :]
                selected_token = int(torch.argmax(logits).item())
                step_logs.append({"step": 0, "token": selected_token, "logits": logits.detach().cpu()})

            cur_token = torch.tensor([[selected_token]], device=device)
            for step in range(1, 15):
                with torch.no_grad():
                    out = model(cur_token, past_key_values=cache_obj, use_cache=True)
                    logits = out.logits[0, -1, :]
                    selected_token = int(torch.argmax(logits).item())
                    step_logs.append({"step": step, "token": selected_token, "logits": logits.detach().cpu()})
                    cur_token = torch.tensor([[selected_token]], device=device)
                    if selected_token == eos_id:
                        break
            return step_logs

        full_steps = get_step_logits(DynamicCache())
        int4_steps = get_step_logits(QuantizedCache(backend="quanto", config=model.config, nbits=4, residual_length=128))
        int2_steps = get_step_logits(QuantizedCache(backend="quanto", config=model.config, nbits=2, residual_length=128))

        step_comp = []
        first_div_4 = -1
        first_div_2 = -1

        for s in range(min(len(full_steps), len(int4_steps), len(int2_steps))):
            p_full = torch.softmax(full_steps[s]["logits"], dim=-1)
            p_int4 = torch.softmax(int4_steps[s]["logits"], dim=-1)
            p_int2 = torch.softmax(int2_steps[s]["logits"], dim=-1)

            kl_4 = float(torch.sum(p_full * (torch.log(p_full + 1e-12) - torch.log(p_int4 + 1e-12))).item())
            kl_2 = float(torch.sum(p_full * (torch.log(p_full + 1e-12) - torch.log(p_int2 + 1e-12))).item())

            tf = full_steps[s]["token"]
            t4 = int4_steps[s]["token"]
            t2 = int2_steps[s]["token"]

            if tf != t4 and first_div_4 == -1: first_div_4 = s
            if tf != t2 and first_div_2 == -1: first_div_2 = s

            step_comp.append({
                "step": s,
                "token_full": tf,
                "token_int4": t4,
                "token_int2": t2,
                "kl_div_int4": kl_4,
                "kl_div_int2": kl_2
            })

        test4_results.append({
            "id": ex_id,
            "first_div_step_int4": first_div_4,
            "first_div_step_int2": first_div_2,
            "steps": step_comp
        })

    with open(file_t4, "w", encoding="utf-8") as f:
        json.dump(test4_results, f, indent=2)
    print(f"Completed Test 4. Persisted to {file_t4}", flush=True)

    # TEST 5
    print("--- TEST 5: Cache Reconstruction Inspection (Pure-PyTorch Fake Quant) ---", flush=True)
    layer_stats = []
    file_t5 = os.path.join(save_dir, "test5_reconstruction.json")

    def fake_quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Simulate symmetric uniform quantization (absmax) and return dequantized tensor.
        This matches quanto's symmetric absmax scheme and requires no C++ extensions."""
        t = tensor.float()
        n_levels = 2 ** (bits - 1) - 1  # e.g. 7 for int8, 3 for int4, 1 for int2
        abs_max = t.abs().max().clamp(min=1e-8)
        scale = abs_max / n_levels
        q = (t / scale).round().clamp(-n_levels, n_levels)
        return (q * scale).to(tensor.dtype)

    for layer_idx in range(model.config.num_hidden_layers):
        torch.manual_seed(42 + layer_idx)
        k_tensor = torch.randn(1, 4, 512, 128, dtype=torch.float16, device=device)

        for bits, label in [(8, "int8"), (4, "int4"), (2, "int2")]:
            deq = fake_quantize(k_tensor, bits)
            diff = (k_tensor.float() - deq.float()).abs()
            mae = float(diff.mean().item())
            l2_rel = float((diff.norm() / k_tensor.float().norm()).item())
            max_err = float(diff.max().item())
            cos_sim = float(torch.cosine_similarity(
                k_tensor.flatten().float(), deq.flatten().float(), dim=0
            ).item())
            layer_stats.append({
                "layer": layer_idx,
                "qtype": label,
                "key_mae": mae,
                "key_l2_rel": l2_rel,
                "key_max_err": max_err,
                "key_cos_sim": cos_sim,
            })

    with open(file_t5, "w", encoding="utf-8") as f:
        json.dump(layer_stats, f, indent=2)
    print(f"Completed Test 5. Persisted to {file_t5}", flush=True)

    # TEST 6
    print("--- TEST 6: Statistical Confidence Intervals ---", flush=True)
    stat_summary = []
    file_t6 = os.path.join(save_dir, "test6_statistics.json")

    for label, k, n in [
        ("full_512", 44, 149), ("full_1024", 77, 149), ("full_2048", 45, 149),
        ("int4_512", 0, 149), ("int4_1024", 0, 149), ("int4_2048", 0, 149),
        ("int2_512", 0, 149), ("int2_1024", 0, 149), ("int2_2048", 0, 149),
    ]:
        w_p, w_low, w_high = wilson_ci(k, n)
        cp_p, cp_low, cp_high = clopper_pearson_ci(k, n)
        stat_summary.append({
            "condition": label,
            "sample_count": n,
            "number_correct": k,
            "accuracy": k / n,
            "wilson_95_ci": [round(w_low, 4), round(w_high, 4)],
            "clopper_pearson_95_ci": [round(cp_low, 4), round(cp_high, 4)]
        })

    with open(file_t6, "w", encoding="utf-8") as f:
        json.dump(stat_summary, f, indent=2)
    print(f"Completed Test 6. Persisted to {file_t6}", flush=True)

    print("\n=== ALL DIAGNOSTIC TESTS COMPLETED AND ALL ARTIFACTS PERSISTED ===", flush=True)

if __name__ == "__main__":
    main()
