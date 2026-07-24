import os, sys, json, time, csv, random, gc
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from controlkv.utils.reproducibility import set_seed
from controlkv.models.tool_template import format_qwen_tool_prompt
from controlkv.benchmarks.context_scaling import scale_context_to_target
import re

import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "runs/phase3/agent_c_benchmark"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------- PARSER & UTILS -----------------
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

def load_data():
    with open("results/dataset_subset.json", "r", encoding="utf-8") as f:
        _all = json.load(f)["examples"]
    tool_ex = [e for e in _all if e["category"] in ["simple", "multiple", "parallel"]]
    
    # Generate some synthetic ordinary prompts if missing
    ordinary_ex = []
    ord_questions = [
        "What is the capital city of France?",
        "List three common programming languages used for web development.",
        "Explain what a database transaction is in simple terms.",
        "What is the primary function of the human heart?",
        "How do airplanes stay in the air?",
        "Write a python script to reverse a string.",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Why is the sky blue?",
        "What are the benefits of eating vegetables?",
        "Explain quantum entanglement simply.",
        "How does a computer CPU work?",
        "What is the largest planet in our solar system?",
        "What is the boiling point of water?",
        "Who wrote the play Hamlet?",
        "How many continents are there on Earth?"
    ]
    for i, q in enumerate(ord_questions):
        ordinary_ex.append({"id": f"ord_{i}", "category": "ordinary", "question": q})
        
    random.seed(42)
    sample_tool = random.sample(tool_ex, min(25, len(tool_ex)))
    sample_ord = ordinary_ex
    return sample_tool, sample_ord

def main():
    MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]
    CONTEXTS = [512, 1024, 2048]
    CONDITIONS = ["dynamic", "hqq_8bit", "hqq_4bit"]
    
    tool_data, ord_data = load_data()
    print(f"Loaded {len(tool_data)} tool prompts and {len(ord_data)} ordinary prompts.")
    
    out_file = os.path.join(SAVE_DIR, "benchmark_results.jsonl")
    with open(out_file, "w") as f: pass

    try:
        from transformers.cache_utils import HQQQuantizedCache
    except ImportError:
        HQQQuantizedCache = None

    for model_name in MODELS:
        print(f"\n======================================")
        print(f"Evaluating {model_name}")
        print(f"======================================")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            eos_id = int(tokenizer.eos_token_id or 0)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
            model.eval()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        for ctx in CONTEXTS:
            print(f"\n  -- Context Length: {ctx} --")
            
            tool_ids, ord_ids = [], []
            for ex in tool_data:
                if "Qwen" in model_name:
                    bp = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
                else:
                    bp = f"<|im_start|>system\nYou are a helpful assistant with access to tools. Tools: {json.dumps(ex['tools'])}<|im_end|>\n<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
                try:
                    sp, _, _ = scale_context_to_target(bp, ctx, tokenizer)
                    tool_ids.append((ex, tokenizer.encode(sp, return_tensors="pt").to(DEVICE)))
                except: pass
                
            for ex in ord_data:
                bp = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
                try:
                    sp, _, _ = scale_context_to_target(bp, ctx, tokenizer)
                    ord_ids.append((ex, tokenizer.encode(sp, return_tensors="pt").to(DEVICE)))
                except: pass

            for cond in CONDITIONS:
                def get_cache():
                    if cond == "dynamic": return DynamicCache()
                    if "hqq" in cond and HQQQuantizedCache is not None:
                        bits = int(cond.split("_")[1].replace("bit",""))
                        return HQQQuantizedCache(config=model.config, nbits=bits, axis_key=0, axis_value=0, q_group_size=64, residual_length=128)
                    return DynamicCache()
                
                print(f"     Condition: {cond}")
                
                # Run Tool
                for ex, inp in tool_ids:
                    set_seed(42)
                    cache = get_cache()
                    start_t = time.time()
                    toks = greedy_generate(model, inp, cache, max_new_tokens=128, eos_id=eos_id)
                    lat = time.time() - start_t
                    text = tokenizer.decode(toks, skip_special_tokens=True)
                    val, calls = parse_tool_calls_v2(text)
                    cor = evaluate_correctness(calls, ex.get("ground_truth", []))
                    rep = repetition_rate(text)
                    
                    res = {
                        "model": model_name, "ctx": ctx, "cond": cond, "type": "tool",
                        "id": ex["id"], "valid": val, "correct": cor, "rep": rep, "latency": lat,
                        "generated": text
                    }
                    with open(out_file, "a") as f: f.write(json.dumps(res)+"\n")

                # Run Ordinary
                for ex, inp in ord_ids:
                    set_seed(42)
                    cache = get_cache()
                    start_t = time.time()
                    toks = greedy_generate(model, inp, cache, max_new_tokens=128, eos_id=eos_id)
                    lat = time.time() - start_t
                    text = tokenizer.decode(toks, skip_special_tokens=True)
                    rep = repetition_rate(text)
                    
                    res = {
                        "model": model_name, "ctx": ctx, "cond": cond, "type": "ordinary",
                        "id": ex["id"], "valid": True, "correct": True, "rep": rep, "latency": lat,
                        "generated": text
                    }
                    with open(out_file, "a") as f: f.write(json.dumps(res)+"\n")

        del model
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        
    print("Agent C Benchmark Done!")

if __name__ == "__main__":
    main()
