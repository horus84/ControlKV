import os, sys, json, time, gc
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, QuantizedCache
from controlkv.utils.reproducibility import set_seed
from controlkv.models.tool_template import format_qwen_tool_prompt
from controlkv.benchmarks.context_scaling import scale_context_to_target
import re

with open("results/dataset_subset.json", "r", encoding="utf-8") as f:
    _all_ex = json.load(f)["examples"]

simple_ex = [e for e in _all_ex if e["category"] == "simple"][:4]
ORDINARY_4 = [
    "What is the capital city of France?",
    "List three common programming languages used for web development.",
    "Explain what a database transaction is in simple terms.",
    "What is the primary function of the human heart?",
]
FIXED_PROMPTS = {"tool": simple_ex, "ordinary": ORDINARY_4}

CTX = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_tool_calls_v2(text: str):
    if not text or not text.strip(): return False, []
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if matches:
        calls = []
        for m in matches:
            try:
                obj = json.loads(m)
                if "name" in obj: calls.append({"name": str(obj["name"]), "arguments": obj.get("arguments", {})})
            except Exception: pass
        if calls: return True, calls
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "name" in data: return True, [{"name": str(data["name"]), "arguments": data.get("arguments", {})}]
        if isinstance(data, list):
            calls = [{"name": str(i["name"]), "arguments": i.get("arguments", {})} for i in data if isinstance(i, dict) and "name" in i]
            if calls: return True, calls
    except Exception: pass
    m = re.search(r"(\{.*?\}|\[.*?\])", stripped, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "name" in data: return True, [{"name": str(data["name"]), "arguments": data.get("arguments", {})}]
            if isinstance(data, list):
                calls = [{"name": str(i["name"]), "arguments": i.get("arguments", {})} for i in data if isinstance(i, dict) and "name" in i]
                if calls: return True, calls
        except Exception: pass
    return False, []

def evaluate_correctness(parsed, gt):
    if not parsed or not gt: return False
    if isinstance(gt, dict): gt = [gt]
    if len(parsed) != len(gt): return False
    return all(str(p.get("name","")).lower() == str(g.get("name","")).lower() for p, g in zip(parsed, gt))

def repetition_rate(text: str) -> float:
    words = text.split()
    if len(words) < 2: return 0.0
    bg = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return 1.0 - len(set(bg)) / len(bg)

def greedy_generate(model, input_ids, cache, max_new_tokens=128, eos_id=0):
    tokens = []
    cur = input_ids.clone()
    with torch.no_grad():
        out = model(cur, past_key_values=cache, use_cache=True)
        tok = int(out.logits[0, -1, :].argmax().item())
        tokens.append(tok)
        if tok == eos_id: return tokens
        cur = torch.tensor([[tok]], device=input_ids.device)
        for _ in range(max_new_tokens - 1):
            out = model(cur, past_key_values=cache, use_cache=True)
            tok = int(out.logits[0, -1, :].argmax().item())
            tokens.append(tok)
            if tok == eos_id: break
            cur = torch.tensor([[tok]], device=input_ids.device)
    return tokens

def main():
    save_dir = "runs/phase3/agent_a_backend"
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, "old_version_results.jsonl")
    with open(out_file, "w") as f: pass

    models = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
    bits = [4, 2]

    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        eos_id = int(tokenizer.eos_token_id or 0)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
        model.eval()

        tool_ids = []
        for ex in FIXED_PROMPTS["tool"]:
            bp = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
            sp, _, _ = scale_context_to_target(bp, CTX, tokenizer)
            tool_ids.append(tokenizer.encode(sp, return_tensors="pt").to(DEVICE))
        ord_ids = []
        for p in FIXED_PROMPTS["ordinary"]:
            fmt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            ord_ids.append(tokenizer.encode(fmt, return_tensors="pt").to(DEVICE))

        def _run_cond(cond_name, cache_factory):
            print(f"  Condition: {cond_name}")
            dyn_tool_toks, dyn_ord_toks = [], []
            if cond_name != "dynamic":
                for idx, inp in enumerate(tool_ids):
                    set_seed(42)
                    dyn_tool_toks.append(greedy_generate(model, inp, DynamicCache(), max_new_tokens=128, eos_id=eos_id))
                for idx, inp in enumerate(ord_ids):
                    set_seed(42)
                    dyn_ord_toks.append(greedy_generate(model, inp, DynamicCache(), max_new_tokens=128, eos_id=eos_id))
            
            torch.cuda.reset_peak_memory_stats() if DEVICE=="cuda" else None
            
            for idx, inp in enumerate(tool_ids):
                set_seed(42)
                cache = cache_factory()
                start_t = time.time()
                toks = greedy_generate(model, inp, cache, max_new_tokens=128, eos_id=eos_id)
                lat = time.time() - start_t
                text = tokenizer.decode(toks, skip_special_tokens=True)
                valid, calls = parse_tool_calls_v2(text)
                
                fdp, byte_id = -1, False
                if cond_name == "dynamic":
                    byte_id = True
                else:
                    dyn = dyn_tool_toks[idx]
                    byte_id = (toks == dyn)
                    if not byte_id:
                        for i, (a, b) in enumerate(zip(dyn, toks)):
                            if a != b:
                                fdp = i; break
                        if fdp == -1: fdp = min(len(dyn), len(toks))

                res = {
                    "model": model_name, "condition": cond_name, "type": "tool", "prompt_idx": idx,
                    "valid": valid, "correct": evaluate_correctness(calls, FIXED_PROMPTS["tool"][idx].get("ground_truth", [])),
                    "byte_identical": byte_id, "first_divergence_token": fdp,
                    "repetition_rate": repetition_rate(text), "latency": lat,
                    "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2) if DEVICE=="cuda" else 0,
                    "generated": text
                }
                with open(out_file, "a") as f: f.write(json.dumps(res) + "\n")

            for idx, inp in enumerate(ord_ids):
                set_seed(42)
                cache = cache_factory()
                start_t = time.time()
                toks = greedy_generate(model, inp, cache, max_new_tokens=128, eos_id=eos_id)
                lat = time.time() - start_t
                text = tokenizer.decode(toks, skip_special_tokens=True)
                
                fdp, byte_id = -1, False
                if cond_name == "dynamic":
                    byte_id = True
                else:
                    dyn = dyn_ord_toks[idx]
                    byte_id = (toks == dyn)
                    if not byte_id:
                        for i, (a, b) in enumerate(zip(dyn, toks)):
                            if a != b:
                                fdp = i; break
                        if fdp == -1: fdp = min(len(dyn), len(toks))

                res = {
                    "model": model_name, "condition": cond_name, "type": "ordinary", "prompt_idx": idx,
                    "byte_identical": byte_id, "first_divergence_token": fdp,
                    "repetition_rate": repetition_rate(text), "latency": lat,
                    "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2) if DEVICE=="cuda" else 0,
                    "generated": text
                }
                with open(out_file, "a") as f: f.write(json.dumps(res) + "\n")

        _run_cond("dynamic", lambda: DynamicCache())
        
        from transformers.cache_utils import QuantizedCacheConfig, QuantoQuantizedCache
        for b in bits:
            def make_quanto(b=b):
                cfg = QuantizedCacheConfig(backend="quanto", nbits=b)
                return QuantoQuantizedCache(cache_config=cfg)
            try:
                cfg = QuantizedCacheConfig(backend="quanto", nbits=b)
                QuantoQuantizedCache(cache_config=cfg)
                _run_cond(f"quanto_{b}bit", make_quanto)
            except Exception as e:
                print(f"Skipping quanto_{b}bit: {e}")

        del model
        gc.collect()
        if DEVICE=="cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
