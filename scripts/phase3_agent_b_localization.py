import os, sys, json, time, gc
import torch
import numpy as np
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, QuantizedCache, QuantoQuantizedLayer, DynamicLayer
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
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SAVE_DIR = "runs/phase3/agent_b_localization"
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

# ----------------- CACHE CLASSES -----------------
class PartialQuantizedLayer(QuantoQuantizedLayer):
    """Allows quantizing only K, only V, or profiling errors."""
    def __init__(self, layer_idx, quantize_k=True, quantize_v=True, profile_errors=False, error_list=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_idx = layer_idx
        self.quantize_k = quantize_k
        self.quantize_v = quantize_v
        self.profile_errors = profile_errors
        self.error_list = error_list

    def update(self, key_states, value_states, *args, **kwargs):
        self._q_mode = "k"
        self.cumulative_length += key_states.shape[-2]
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self._q_mode = "k"
            self._quantized_keys = super()._quantize(key_states.contiguous(), axis=self.axis_key) if self.quantize_k else key_states.clone().contiguous()
            self._q_mode = "v"
            self._quantized_values = super()._quantize(value_states.contiguous(), axis=self.axis_value) if self.quantize_v else value_states.clone().contiguous()
            return key_states, value_states

        dequant_keys = super()._dequantize(self._quantized_keys) if self.quantize_k else self._quantized_keys
        dequant_values = super()._dequantize(self._quantized_values) if self.quantize_v else self._quantized_values
        keys_to_return = torch.cat([dequant_keys, self.keys, key_states], dim=-2)
        values_to_return = torch.cat([dequant_values, self.values, value_states], dim=-2)

        if self.keys.dim() == 4 and self.keys.shape[-2] + 1 >= self.residual_length:
            self._q_mode = "k"
            self._quantized_keys = super()._quantize(keys_to_return.contiguous(), axis=self.axis_key) if self.quantize_k else keys_to_return.clone().contiguous()
            self._q_mode = "v"
            self._quantized_values = super()._quantize(values_to_return.contiguous(), axis=self.axis_value) if self.quantize_v else values_to_return.clone().contiguous()
            self.keys = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
            self.values = torch.tensor([], dtype=key_states.dtype, device=key_states.device)
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        return keys_to_return, values_to_return

    def _quantize(self, tensor, axis):
        q_tensor = super()._quantize(tensor, axis)
        if self.profile_errors and self.error_list is not None:
            dequant = super()._dequantize(q_tensor)
            mae = (dequant - tensor).abs().mean().item()
            l2 = torch.nn.functional.mse_loss(dequant, tensor).item()
            cos = torch.nn.functional.cosine_similarity(dequant.flatten(), tensor.flatten(), dim=0).item()
            nr = dequant.norm().item() / max(tensor.norm().item(), 1e-9)
            self.error_list.append({
                "layer": self.layer_idx,
                "type": self._q_mode,
                "mae": mae, "mse": l2, "cosine": cos, "norm_ratio": nr
            })
        return q_tensor

class BlockAblationCache(QuantizedCache):
    def __init__(self, config, layer_quant_flags, quantize_k=True, quantize_v=True, profile_errors=False, error_list=None, nbits=4):
        cfg = config.get_text_config(decoder=True)
        layers = []
        for i in range(cfg.num_hidden_layers):
            if layer_quant_flags[i]:
                layers.append(PartialQuantizedLayer(
                    layer_idx=i, quantize_k=quantize_k, quantize_v=quantize_v,
                    profile_errors=profile_errors, error_list=error_list,
                    nbits=nbits, axis_key=0, axis_value=0, q_group_size=64, residual_length=128
                ))
            else:
                layers.append(DynamicLayer())
        # QuantizedCache __init__ just assigns self.layers if we don't call super
        super().__init__(backend="quanto", config=config, nbits=nbits)
        self.layers = layers # override with our custom layers

def main():
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eos_id = int(tokenizer.eos_token_id or 0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    num_layers = model.config.num_hidden_layers

    tool_ids, ord_ids = [], []
    for ex in FIXED_PROMPTS["tool"]:
        bp = format_qwen_tool_prompt(ex["tools"], ex["question"], tokenizer)
        sp, _, _ = scale_context_to_target(bp, CTX, tokenizer)
        tool_ids.append(tokenizer.encode(sp, return_tensors="pt").to(DEVICE))
    for p in FIXED_PROMPTS["ordinary"]:
        fmt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        ord_ids.append(tokenizer.encode(fmt, return_tensors="pt").to(DEVICE))

    results_file = os.path.join(SAVE_DIR, "b_ablation_results.jsonl")
    error_list = []
    
    with open(results_file, "w") as f: pass

    def run_cond(cond_name, flags, qk=True, qv=True, prof=False, bits=4):
        print(f"Condition: {cond_name}")
        for idx, inp in enumerate(tool_ids):
            set_seed(42)
            cache = BlockAblationCache(model.config, flags, quantize_k=qk, quantize_v=qv, profile_errors=prof, error_list=error_list, nbits=bits)
            toks = greedy_generate(model, inp, cache, max_new_tokens=128, eos_id=eos_id)
            text = tokenizer.decode(toks, skip_special_tokens=True)
            val, _ = parse_tool_calls_v2(text)
            res = {"cond": cond_name, "type": "tool", "valid": val, "rep": repetition_rate(text)}
            with open(results_file, "a") as f: f.write(json.dumps(res)+"\n")
            
        for idx, inp in enumerate(ord_ids):
            set_seed(42)
            cache = BlockAblationCache(model.config, flags, quantize_k=qk, quantize_v=qv, profile_errors=prof, error_list=error_list, nbits=bits)
            toks = greedy_generate(model, inp, cache, max_new_tokens=128, eos_id=eos_id)
            text = tokenizer.decode(toks, skip_special_tokens=True)
            res = {"cond": cond_name, "type": "ordinary", "valid": "-", "rep": repetition_rate(text)}
            with open(results_file, "a") as f: f.write(json.dumps(res)+"\n")

    # B1: K/V Ablation
    run_cond("k_only_int4", [True]*num_layers, qk=True, qv=False, bits=4)
    run_cond("v_only_int4", [True]*num_layers, qk=False, qv=True, bits=4)
    run_cond("k_only_int2", [True]*num_layers, qk=True, qv=False, bits=2)
    run_cond("v_only_int2", [True]*num_layers, qk=False, qv=True, bits=2)

    # B2: Layer Blocks (INT4)
    # block 0-7, 8-15, 16-23, 24-27
    b_sizes = [(0,6), (7,13), (14,20), (21,27)]
    for start, end in b_sizes:
        flags = [True if start<=i<=end else False for i in range(num_layers)]
        run_cond(f"block_{start}_{end}_int4", flags, qk=True, qv=True, bits=4)

    # Prefixes / Suffixes
    run_cond("prefix_first_14_int4", [True]*14 + [False]*14, qk=True, qv=True, bits=4)
    run_cond("suffix_last_14_int4", [False]*14 + [True]*14, qk=True, qv=True, bits=4)
    
    # B3: Reconstruction Errors (profiling on one prompt)
    print("Profiling Reconstruction Errors...")
    set_seed(42)
    cache = BlockAblationCache(model.config, [True]*num_layers, quantize_k=True, quantize_v=True, profile_errors=True, error_list=error_list, nbits=4)
    greedy_generate(model, tool_ids[0], cache, max_new_tokens=5, eos_id=eos_id)

    # Dump error list
    with open(os.path.join(SAVE_DIR, "layer_errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "type", "mae", "mse", "cosine", "norm_ratio"])
        w.writeheader()
        w.writerows(error_list)

    print("Agent B localization done.")

if __name__ == "__main__":
    main()
