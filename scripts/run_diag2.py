#!/usr/bin/env python3
"""
Second-round diagnostic investigation (Tests 1-7).
Preserves all prior outputs; writes to runs/diag2/
"""
import os, sys, json, math, copy, gc, re
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# ── project imports ──────────────────────────────────────────────────────────
from controlkv.utils.reproducibility import set_seed
from controlkv.models.tool_template import format_qwen_tool_prompt
from controlkv.benchmarks.context_scaling import scale_context_to_target

SAVE_DIR = os.path.abspath("runs/diag2")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0. VERSION RECORD  (also used in Test 7)
# ─────────────────────────────────────────────────────────────────────────────
import importlib.metadata

def _pkg_ver(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "not-installed"

VERSION_INFO = {
    "transformers":   _pkg_ver("transformers"),
    "optimum":        _pkg_ver("optimum"),
    "optimum_quanto": _pkg_ver("optimum-quanto"),
    "torch":          _pkg_ver("torch"),
    "python":         sys.version.split()[0],
}
with open(os.path.join(SAVE_DIR, "versions.json"), "w") as f:
    json.dump(VERSION_INFO, f, indent=2)
print("Versions:", VERSION_INFO, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — cache infrastructure
# ─────────────────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import (
    Cache, DynamicCache, QuantizedCache,
    QuantizedLayer, QuantoQuantizedLayer, DynamicLayer,
    get_layer_types_and_kwargs,
)
from transformers import PreTrainedConfig
import optimum.quanto as quanto
from optimum.quanto import MaxOptimizer, qint2, qint4, quantize_weight

# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY QUANTIZED LAYER  (Test 1)
# ─────────────────────────────────────────────────────────────────────────────
class IdentityQuantizedLayer(QuantizedLayer):
    """
    Uses the IDENTICAL control-flow as QuantizedLayer / QuantoQuantizedLayer
    (lazy init, cumulative_length, residual eviction) but stores exact tensor
    clones without any numerical conversion.
    If this matches DynamicCache => control-flow is correct.
    If it diverges => investigate cache control-flow before quantization quality.
    """
    def _quantize(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        return tensor.clone()

    def _dequantize(self, stored: torch.Tensor) -> torch.Tensor:
        return stored


class IdentityQuantizedCache(Cache):
    """Drop-in for QuantizedCache using IdentityQuantizedLayer."""
    def __init__(self, config: PreTrainedConfig, residual_length: int = 128):
        cfg = config.get_text_config(decoder=True)
        layers = [
            IdentityQuantizedLayer(
                nbits=8, axis_key=0, axis_value=0,
                q_group_size=64, residual_length=residual_length,
            )
            for _ in range(cfg.num_hidden_layers)
        ]
        super().__init__(layers=layers)


# ─────────────────────────────────────────────────────────────────────────────
# PARSER v2  (Test 6 — used across all tests for consistent evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def parse_tool_calls_v2(text: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Extended parser accepting:
      1. <tool_call>...</tool_call>   (Qwen native)
      2. {"name":..., "arguments":...}  (bare JSON dict)
      3. [{"name":..., ...}]           (JSON array)
      4. First {...} or [...] substring
    """
    if not text or not text.strip():
        return False, []

    # Pattern 0: Qwen native
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if matches:
        calls = []
        for m in matches:
            try:
                obj = json.loads(m)
                if "name" in obj:
                    calls.append({"name": str(obj["name"]), "arguments": obj.get("arguments", {})})
            except Exception:
                pass
        if calls:
            return True, calls

    # Pattern 1-3: direct JSON parse of full string
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "name" in data:
            return True, [{"name": str(data["name"]), "arguments": data.get("arguments", {})}]
        if isinstance(data, list):
            calls = [{"name": str(i["name"]), "arguments": i.get("arguments", {})}
                     for i in data if isinstance(i, dict) and "name" in i]
            if calls:
                return True, calls
    except Exception:
        pass

    # Pattern 4: extract first JSON object/array substring
    m = re.search(r"(\{.*?\}|\[.*?\])", stripped, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "name" in data:
                return True, [{"name": str(data["name"]), "arguments": data.get("arguments", {})}]
            if isinstance(data, list):
                calls = [{"name": str(i["name"]), "arguments": i.get("arguments", {})}
                         for i in data if isinstance(i, dict) and "name" in i]
                if calls:
                    return True, calls
        except Exception:
            pass

    return False, []


def evaluate_correctness(parsed: List[Dict], gt: Any) -> bool:
    if not parsed or not gt:
        return False
    if isinstance(gt, dict):
        gt = [gt]
    if len(parsed) != len(gt):
        return False
    return all(str(p.get("name","")).lower() == str(g.get("name","")).lower()
               for p, g in zip(parsed, gt))


# ─────────────────────────────────────────────────────────────────────────────
# STABLE JSD  (Test 3)
# ─────────────────────────────────────────────────────────────────────────────
def stable_jsd(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """JSD via float32 log_softmax — never produces NaN for finite inputs."""
    lp = torch.nn.functional.log_softmax(logits_p.float(), dim=-1)
    lq = torch.nn.functional.log_softmax(logits_q.float(), dim=-1)
    p, q = lp.exp(), lq.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-40))
    kl_pm = (p * (lp - log_m)).sum()
    kl_qm = (q * (lq - log_m)).sum()
    return float((0.5 * (kl_pm + kl_qm)).clamp(min=0.0).item())


def logit_stats(logits: torch.Tensor, top_k: int = 10) -> Dict[str, Any]:
    nan_c  = int(logits.isnan().sum().item())
    pinf_c = int((logits == float("inf")).sum().item())
    ninf_c = int((logits == float("-inf")).sum().item())
    fin    = logits[logits.isfinite()]
    min_f  = float(fin.min().item()) if fin.numel() > 0 else None
    max_f  = float(fin.max().item()) if fin.numel() > 0 else None
    topk_v, topk_i = logits.topk(min(top_k, logits.numel()))
    return {
        "nan_count": nan_c, "posinf_count": pinf_c, "neginf_count": ninf_c,
        "min_finite": min_f, "max_finite": max_f,
        "top10_ids": topk_i.tolist(),
        "top10_vals": [round(v, 4) for v in topk_v.tolist()],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY DECODE LOOP with per-step logit capture
# ─────────────────────────────────────────────────────────────────────────────
def greedy_with_logits(
    model, input_ids: torch.Tensor, cache: Any,
    max_new_tokens: int = 32, eos_id: int = 0
) -> Tuple[List[int], List[torch.Tensor]]:
    tokens: List[int] = []
    all_logits: List[torch.Tensor] = []
    cur = input_ids.clone()
    with torch.no_grad():
        out = model(cur, past_key_values=cache, use_cache=True)
        l1d = out.logits[0, -1, :].detach().cpu()
        tok = int(l1d.argmax().item())
        tokens.append(tok); all_logits.append(l1d)
        if tok == eos_id:
            return tokens, all_logits
        cur = torch.tensor([[tok]], device=input_ids.device)
        for _ in range(max_new_tokens - 1):
            out = model(cur, past_key_values=cache, use_cache=True)
            l1d = out.logits[0, -1, :].detach().cpu()
            tok = int(l1d.argmax().item())
            tokens.append(tok); all_logits.append(l1d)
            if tok == eos_id:
                break
            cur = torch.tensor([[tok]], device=input_ids.device)
    return tokens, all_logits


def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    d = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        nd = [i]
        for j, cb in enumerate(b, 1):
            nd.append(min(nd[-1]+1, d[j]+1, d[j-1]+(0 if ca == cb else 1)))
        d = nd
    return d[-1]


def repetition_rate(text: str) -> float:
    words = text.split()
    if len(words) < 2: return 0.0
    bg = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return 1.0 - len(set(bg)) / len(bg)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRIMARY MODEL
# ─────────────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"\n=== Loading {MODEL_NAME} on {DEVICE} ===", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS_ID = int(tokenizer.eos_token_id or 0)

model15 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
)
model15.eval()
print("Model loaded.", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIXED EXAMPLE SETS
# ─────────────────────────────────────────────────────────────────────────────
with open("results/dataset_subset.json", "r", encoding="utf-8") as f:
    _all_ex = json.load(f)["examples"]

simple_ex   = [e for e in _all_ex if e["category"] == "simple"][:4]
multiple_ex = [e for e in _all_ex if e["category"] == "multiple"][:4]
parallel_ex = [e for e in _all_ex if e["category"] == "parallel"][:4]
FIXED_12 = simple_ex + multiple_ex + parallel_ex

ORDINARY_4 = [
    "What is the capital city of France?",
    "List three common programming languages used for web development.",
    "Explain what a database transaction is in simple terms.",
    "What is the primary function of the human heart?",
]

CTX = 512  # context length used across all diagnostic tests
print(f"Fixed 12 examples loaded. CTX={CTX}.", flush=True)


def _make_cache(cond: str, config) -> Any:
    if cond == "dynamic":
        return DynamicCache()
    elif cond == "identity":
        return IdentityQuantizedCache(config, residual_length=128)
    elif cond == "int4":
        return QuantizedCache(backend="quanto", config=config, nbits=4,
                              axis_key=0, axis_value=0, q_group_size=64, residual_length=128)
    elif cond == "int2":
        return QuantizedCache(backend="quanto", config=config, nbits=2,
                              axis_key=0, axis_value=0, q_group_size=64, residual_length=128)
    raise ValueError(cond)


def _encode_tool(ex, tok, ctx):
    bp = format_qwen_tool_prompt(ex["tools"], ex["question"], tok)
    sp, _, _ = scale_context_to_target(bp, ctx, tok)
    return tok.encode(sp, return_tensors="pt").to(DEVICE)


def _encode_ordinary(p, tok):
    fmt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
           f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
    return tok.encode(fmt, return_tensors="pt").to(DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — IDENTITY QUANTIZED CACHE CONTROL
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 1: Identity QuantizedCache Control ---", flush=True)
FILE_T1 = os.path.join(SAVE_DIR, "test1_identity_control.json")
t1_results = []

for ex in FIXED_12:
    input_ids = _encode_tool(ex, tokenizer, CTX)
    tokens_per: Dict[str, List[int]] = {}
    for cond in ["dynamic", "identity", "int4", "int2"]:
        set_seed(42)
        cache = _make_cache(cond, model15.config)
        toks, _ = greedy_with_logits(model15, input_ids, cache,
                                      max_new_tokens=32, eos_id=EOS_ID)
        tokens_per[cond] = toks

    dyn = tokens_per["dynamic"]
    for cond in ["dynamic", "identity", "int4", "int2"]:
        toks = tokens_per[cond]
        byte_id = (toks == dyn)
        fdp = -1
        if not byte_id:
            for i, (a, b) in enumerate(zip(dyn, toks)):
                if a != b:
                    fdp = i
                    break
            if fdp == -1:
                fdp = min(len(dyn), len(toks))
        decoded = tokenizer.decode(toks, skip_special_tokens=True)
        valid, calls = parse_tool_calls_v2(decoded)
        t1_results.append({
            "id": ex["id"], "category": ex["category"], "condition": cond,
            "token_ids": toks, "decoded": decoded,
            "byte_identical_to_dynamic": byte_id,
            "first_diff_pos": fdp,
            "valid_structured": valid,
            "parsed_calls": calls,
            "correct": evaluate_correctness(calls, ex.get("ground_truth", [])),
        })

for cond in ["dynamic", "identity", "int4", "int2"]:
    rows = [r for r in t1_results if r["condition"] == cond]
    n_id  = sum(1 for r in rows if r["byte_identical_to_dynamic"])
    n_val = sum(1 for r in rows if r["valid_structured"])
    n_cor = sum(1 for r in rows if r["correct"])
    fdps  = [r["first_diff_pos"] for r in rows if r["first_diff_pos"] >= 0]
    mean_fdp_str = f"{np.mean(fdps):.1f}" if fdps else "-1"
    print(f"  {cond:10s}  byte-id={n_id}/{len(rows)}  valid={n_val}/{len(rows)}  "
          f"correct={n_cor}/{len(rows)}  mean_fdp={mean_fdp_str}", flush=True)

with open(FILE_T1, "w", encoding="utf-8") as f:
    json.dump(t1_results, f, indent=2)
print(f"Test 1 saved → {FILE_T1}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — ACTUAL QUANTO ROUND-TRIP on real KV tensors
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 2: Actual Quanto Round-Trip on Real KV Tensors ---", flush=True)
FILE_T2 = os.path.join(SAVE_DIR, "test2_quanto_roundtrip.json")
t2_results = []

# Capture real KV via a prefill pass with DynamicCache
ex_rt      = FIXED_12[0]
ids_rt     = _encode_tool(ex_rt, tokenizer, CTX)

set_seed(42)
_dyn_cap = DynamicCache()
with torch.no_grad():
    _ = model15(ids_rt, past_key_values=_dyn_cap, use_cache=True)

optimizer = MaxOptimizer()

# The MaxOptimizer returns (scale, shift) via optimizer(tensor, qtype, axis, group_size)
# quantize_weight(tensor, qtype, axis, scale, shift, group_size)

for layer_idx, layer in enumerate(_dyn_cap.layers):
    k_orig = layer.keys.float().contiguous()   # [B, H, S, D] float32
    v_orig = layer.values.float().contiguous()

    rec: Dict[str, Any] = {
        "layer":       layer_idx,
        "key_shape":   list(k_orig.shape),
        "value_shape": list(v_orig.shape),
        "key_dtype":   str(layer.keys.dtype),
        "value_dtype": str(layer.values.dtype),
    }

    for nbits, qtype, qname in [(4, qint4, "int4"), (2, qint2, "int2")]:
        for axis_lbl, axis in [("ax0", 0), ("axN1", -1)]:
            tag = f"{qname}_{axis_lbl}"
            try:
                # Exact Quanto path used by QuantoQuantizedLayer
                k_sc, k_sh = optimizer(k_orig, qtype, axis, 64)
                qk = quantize_weight(k_orig, qtype, axis, k_sc, k_sh, 64)
                dk = qk.dequantize().to(k_orig.dtype)

                v_sc, v_sh = optimizer(v_orig, qtype, axis, 64)
                qv = quantize_weight(v_orig, qtype, axis, v_sc, v_sh, 64)
                dv = qv.dequantize().to(v_orig.dtype)

                def _stats(orig: torch.Tensor, deq: torch.Tensor,
                            sc: torch.Tensor, sh: torch.Tensor) -> Dict:
                    diff = (orig - deq).abs()
                    fin  = orig[orig.isfinite()]
                    per_head_cos = []
                    if orig.ndim == 4:
                        for h in range(orig.shape[1]):
                            o_h = orig[0, h].flatten()
                            d_h = deq[0, h].flatten()
                            per_head_cos.append(round(
                                float(torch.cosine_similarity(o_h, d_h, dim=0).item()), 6))
                    return {
                        "orig_min": float(fin.min().item()) if fin.numel() > 0 else None,
                        "orig_max": float(fin.max().item()) if fin.numel() > 0 else None,
                        "finite_count": int(orig.isfinite().sum().item()),
                        "mae":      float(diff.mean().item()),
                        "l2_rel":   float((diff.norm() / orig.norm().clamp(1e-8)).item()),
                        "max_err":  float(diff.max().item()),
                        "cos_global": float(torch.cosine_similarity(
                            orig.flatten(), deq.flatten(), dim=0).item()),
                        "cos_per_head": per_head_cos,
                        "scale_min": float(sc.min().item()),
                        "scale_max": float(sc.max().item()),
                        "shift_min": float(sh.min().item()),
                        "shift_max": float(sh.max().item()),
                        "qtype": qname, "axis": axis, "q_group_size": 64,
                    }

                rec[f"key_{tag}"]   = _stats(k_orig, dk, k_sc, k_sh)
                rec[f"value_{tag}"] = _stats(v_orig, dv, v_sc, v_sh)

            except Exception as e:
                rec[f"key_{tag}"]   = {"error": str(e)}
                rec[f"value_{tag}"] = {"error": str(e)}

    t2_results.append(rec)

for qname in ["int4", "int2"]:
    maes = [r[f"key_{qname}_ax0"]["mae"] for r in t2_results
             if f"key_{qname}_ax0" in r and "error" not in r[f"key_{qname}_ax0"]]
    coss = [r[f"key_{qname}_ax0"]["cos_global"] for r in t2_results
             if f"key_{qname}_ax0" in r and "error" not in r[f"key_{qname}_ax0"]]
    if maes:
        print(f"  {qname} key(ax=0) across layers: MAE={np.mean(maes):.4f}  "
              f"cos mean={np.mean(coss):.4f}  min={np.min(coss):.4f}", flush=True)

del _dyn_cap
if DEVICE == "cuda":
    torch.cuda.empty_cache()

with open(FILE_T2, "w", encoding="utf-8") as f:
    json.dump(t2_results, f, indent=2)
print(f"Test 2 saved → {FILE_T2}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — LOGIT FINITENESS + STABLE JSD
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 3: Logit Finiteness and Stable JSD ---", flush=True)
FILE_T3 = os.path.join(SAVE_DIR, "test3_logit_finiteness.json")
t3_results = []

for ex in FIXED_12[:3]:
    input_ids = _encode_tool(ex, tokenizer, CTX)
    cond_toks:   Dict[str, List[int]]             = {}
    cond_logits: Dict[str, List[torch.Tensor]]    = {}

    for cond in ["dynamic", "int4", "int2"]:
        set_seed(42)
        cache = _make_cache(cond, model15.config)
        toks, logit_seq = greedy_with_logits(model15, input_ids, cache,
                                              max_new_tokens=20, eos_id=EOS_ID)
        cond_toks[cond]   = toks
        cond_logits[cond] = logit_seq

    n_steps = min(len(cond_logits["dynamic"]),
                  len(cond_logits["int4"]),
                  len(cond_logits["int2"]))

    steps = []
    for s in range(n_steps):
        ld  = cond_logits["dynamic"][s]
        l4  = cond_logits["int4"][s]
        l2  = cond_logits["int2"][s]
        sd, s4, s2 = logit_stats(ld), logit_stats(l4), logit_stats(l2)

        # Stable JSD — only when both sides are fully finite
        def _jsd_safe(la, lb, sa, sb):
            if sa["nan_count"] == 0 and sb["nan_count"] == 0:
                return stable_jsd(la, lb)
            return None

        steps.append({
            "step":            s,
            "tok_dynamic":     cond_toks["dynamic"][s] if s < len(cond_toks["dynamic"]) else None,
            "tok_int4":        cond_toks["int4"][s]    if s < len(cond_toks["int4"])    else None,
            "tok_int2":        cond_toks["int2"][s]    if s < len(cond_toks["int2"])    else None,
            "logits_dynamic":  sd,
            "logits_int4":     s4,
            "logits_int2":     s2,
            "jsd_dyn_vs_int4": _jsd_safe(ld, l4, sd, s4),
            "jsd_dyn_vs_int2": _jsd_safe(ld, l2, sd, s2),
            "int4_finite":     (s4["nan_count"] == 0 and s4["posinf_count"] == 0),
            "int2_finite":     (s2["nan_count"] == 0 and s2["posinf_count"] == 0),
        })

    t3_results.append({"id": ex["id"], "steps": steps})

for r in t3_results:
    n4f = sum(1 for s in r["steps"] if s["int4_finite"])
    n2f = sum(1 for s in r["steps"] if s["int2_finite"])
    N   = len(r["steps"])
    j4  = [s["jsd_dyn_vs_int4"] for s in r["steps"] if s["jsd_dyn_vs_int4"] is not None]
    j2  = [s["jsd_dyn_vs_int2"] for s in r["steps"] if s["jsd_dyn_vs_int2"] is not None]
    j4s = f"{np.mean(j4):.4f}" if j4 else "n/a"
    j2s = f"{np.mean(j2):.4f}" if j2 else "n/a"
    print(f"  {r['id']}: int4 finite={n4f}/{N}  int2 finite={n2f}/{N}  "
          f"mean_JSD int4={j4s}  int2={j2s}", flush=True)

with open(FILE_T3, "w", encoding="utf-8") as f:
    json.dump(t3_results, f, indent=2)
print(f"Test 3 saved → {FILE_T3}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — AXIS × GROUP-SIZE GRID
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 4: Axis x Group-size Grid ---", flush=True)
FILE_T4 = os.path.join(SAVE_DIR, "test4_axis_grid.json")
t4_results = []

TOOL_4   = FIXED_12[:4]
ORD_4    = ORDINARY_4

tool_ids = [_encode_tool(ex, tokenizer, CTX) for ex in TOOL_4]
ord_ids  = [_encode_ordinary(p, tokenizer) for p in ORD_4]

# Precompute DynamicCache baseline outputs
dyn_tool_toks = []
dyn_tool_text = []
for inp in tool_ids:
    set_seed(42); c = DynamicCache()
    t, _ = greedy_with_logits(model15, inp, c, max_new_tokens=32, eos_id=EOS_ID)
    dyn_tool_toks.append(t)
    dyn_tool_text.append(tokenizer.decode(t, skip_special_tokens=True))

dyn_ord_toks = []
dyn_ord_text = []
for inp in ord_ids:
    set_seed(42); c = DynamicCache()
    t, _ = greedy_with_logits(model15, inp, c, max_new_tokens=32, eos_id=EOS_ID)
    dyn_ord_toks.append(t)
    dyn_ord_text.append(tokenizer.decode(t, skip_special_tokens=True))

AXIS_CFGS  = [(0, 0), (-1, -1), (0, -1), (-1, 0)]
GROUP_SIZES = [32, 64, 128]

for axis_key, axis_value in AXIS_CFGS:
    for gs in GROUP_SIZES:
        tag = f"ak{axis_key}_av{axis_value}_gs{gs}"
        row: Dict[str, Any] = {
            "axis_key": axis_key, "axis_value": axis_value, "q_group_size": gs,
            "nbits": 4, "tag": tag,
            "ordinary": [], "tool": [],
            "skipped": False, "skip_reason": None,
        }

        # Probe: can we instantiate the cache at all?
        try:
            _probe = QuantizedCache(
                backend="quanto", config=model15.config, nbits=4,
                axis_key=axis_key, axis_value=axis_value,
                q_group_size=gs, residual_length=128
            )
        except Exception as e:
            row["skipped"] = True
            row["skip_reason"] = str(e)
            t4_results.append(row)
            print(f"  {tag} SKIPPED at init: {e}", flush=True)
            continue

        # Ordinary prompts
        for idx, inp in enumerate(ord_ids):
            set_seed(42)
            try:
                cache = QuantizedCache(
                    backend="quanto", config=model15.config, nbits=4,
                    axis_key=axis_key, axis_value=axis_value,
                    q_group_size=gs, residual_length=128
                )
                toks, _ = greedy_with_logits(model15, inp, cache, max_new_tokens=32, eos_id=EOS_ID)
                text = tokenizer.decode(toks, skip_special_tokens=True)
                byte_id = (toks == dyn_ord_toks[idx])
                lev = levenshtein(text, dyn_ord_text[idx])
                ml  = max(len(text), len(dyn_ord_text[idx]), 1)
                row["ordinary"].append({
                    "idx": idx, "byte_identical": byte_id,
                    "edit_similarity": round(1.0 - lev/ml, 4),
                    "repetition_rate": round(repetition_rate(text), 4),
                    "generated": text[:200],
                })
            except Exception as e:
                row["ordinary"].append({"idx": idx, "error": str(e)})

        # Tool prompts
        for idx, inp in enumerate(tool_ids):
            set_seed(42)
            try:
                cache = QuantizedCache(
                    backend="quanto", config=model15.config, nbits=4,
                    axis_key=axis_key, axis_value=axis_value,
                    q_group_size=gs, residual_length=128
                )
                toks, _ = greedy_with_logits(model15, inp, cache, max_new_tokens=32, eos_id=EOS_ID)
                text = tokenizer.decode(toks, skip_special_tokens=True)
                valid, calls = parse_tool_calls_v2(text)
                byte_id = (toks == dyn_tool_toks[idx])
                lev = levenshtein(text, dyn_tool_text[idx])
                ml  = max(len(text), len(dyn_tool_text[idx]), 1)
                row["tool"].append({
                    "id": TOOL_4[idx]["id"], "byte_identical": byte_id,
                    "edit_similarity": round(1.0 - lev/ml, 4),
                    "valid_structured": valid,
                    "correct": evaluate_correctness(calls, TOOL_4[idx].get("ground_truth", [])),
                    "generated": text[:200],
                })
            except Exception as e:
                row["tool"].append({"id": TOOL_4[idx]["id"], "error": str(e)})

        n_ord_id = sum(1 for r in row["ordinary"] if r.get("byte_identical", False))
        n_tl_id  = sum(1 for r in row["tool"]     if r.get("byte_identical", False))
        n_valid  = sum(1 for r in row["tool"]     if r.get("valid_structured", False))
        print(f"  {tag}: ord_byte_id={n_ord_id}/{len(ord_ids)}  "
              f"tool_byte_id={n_tl_id}/{len(tool_ids)}  tool_valid={n_valid}/{len(tool_ids)}", flush=True)
        t4_results.append(row)

with open(FILE_T4, "w", encoding="utf-8") as f:
    json.dump(t4_results, f, indent=2)
print(f"Test 4 saved → {FILE_T4}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — BACKEND AND MODEL CONTROLS
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 5: Backend and Model Controls ---", flush=True)
FILE_T5 = os.path.join(SAVE_DIR, "test5_backend_model.json")
t5_results: Dict[str, Any] = {}


def _run_8(model, tok, eos, tool_input_ids, ord_input_ids, cache_factory, tool_exs):
    results = []
    for idx, inp in enumerate(tool_input_ids):
        set_seed(42)
        try:
            cache = cache_factory()
            toks, _ = greedy_with_logits(model, inp, cache, max_new_tokens=32, eos_id=eos)
            text = tok.decode(toks, skip_special_tokens=True)
            valid, calls = parse_tool_calls_v2(text)
            results.append({"type": "tool", "id": tool_exs[idx]["id"],
                             "valid": valid, "generated": text[:200]})
        except Exception as e:
            results.append({"type": "tool", "id": tool_exs[idx]["id"], "error": str(e)})
    for idx, inp in enumerate(ord_input_ids):
        set_seed(42)
        try:
            cache = cache_factory()
            toks, _ = greedy_with_logits(model, inp, cache, max_new_tokens=32, eos_id=eos)
            text = tok.decode(toks, skip_special_tokens=True)
            results.append({"type": "ordinary", "idx": idx, "generated": text[:200]})
        except Exception as e:
            results.append({"type": "ordinary", "idx": idx, "error": str(e)})
    return results


# A) HQQ backend on 1.5B
try:
    _hqq_test = QuantizedCache(backend="hqq", config=model15.config, nbits=4, residual_length=128)
    del _hqq_test
    t5_results["A_hqq_1.5B"] = {
        "backend": "hqq", "model": MODEL_NAME, "nbits": 4,
        "results": _run_8(model15, tokenizer, EOS_ID, tool_ids, ord_ids,
                          lambda: QuantizedCache(backend="hqq", config=model15.config,
                                                 nbits=4, residual_length=128),
                          TOOL_4)
    }
    n_valid_hqq = sum(1 for r in t5_results["A_hqq_1.5B"]["results"] if r.get("valid", False))
    print(f"  A) HQQ 1.5B: valid_tool={n_valid_hqq}/{len(tool_ids)}", flush=True)
except Exception as e:
    t5_results["A_hqq_1.5B"] = {"backend": "hqq", "error": str(e)}
    print(f"  A) HQQ not available: {e}", flush=True)


# B-D) Qwen2.5-0.5B-Instruct
MODEL_05B = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"  Loading {MODEL_05B}...", flush=True)
try:
    tok05 = AutoTokenizer.from_pretrained(MODEL_05B, trust_remote_code=True)
    if tok05.pad_token is None:
        tok05.pad_token = tok05.eos_token
    eos05 = int(tok05.eos_token_id or 0)
    model05 = AutoModelForCausalLM.from_pretrained(
        MODEL_05B, dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
    )
    model05.eval()

    tool_ids05 = [_encode_tool(ex, tok05, CTX) for ex in TOOL_4]
    ord_ids05  = [_encode_ordinary(p, tok05) for p in ORD_4]

    t5_results["B_dynamic_0.5B"] = {
        "backend": "dynamic", "model": MODEL_05B,
        "results": _run_8(model05, tok05, eos05, tool_ids05, ord_ids05,
                          DynamicCache, TOOL_4)
    }
    t5_results["C_int4_0.5B"] = {
        "backend": "quanto_int4", "model": MODEL_05B, "nbits": 4,
        "results": _run_8(model05, tok05, eos05, tool_ids05, ord_ids05,
                          lambda: QuantizedCache(backend="quanto", config=model05.config,
                                                 nbits=4, axis_key=0, axis_value=0,
                                                 q_group_size=64, residual_length=128),
                          TOOL_4)
    }
    t5_results["D_int2_0.5B"] = {
        "backend": "quanto_int2", "model": MODEL_05B, "nbits": 2,
        "results": _run_8(model05, tok05, eos05, tool_ids05, ord_ids05,
                          lambda: QuantizedCache(backend="quanto", config=model05.config,
                                                 nbits=2, axis_key=0, axis_value=0,
                                                 q_group_size=64, residual_length=128),
                          TOOL_4)
    }

    for key in ["B_dynamic_0.5B", "C_int4_0.5B", "D_int2_0.5B"]:
        n_ok = sum(1 for r in t5_results[key]["results"] if r.get("valid") or r.get("generated"))
        print(f"  {key}: responded={n_ok}/8", flush=True)

    del model05
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

except Exception as e:
    t5_results["0.5B_error"] = {"error": str(e)}
    print(f"  0.5B failed: {e}", flush=True)

with open(FILE_T5, "w", encoding="utf-8") as f:
    json.dump(t5_results, f, indent=2)
print(f"Test 5 saved → {FILE_T5}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — PARSER REPAIR
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 6: Parser Repair ---", flush=True)
FILE_T6 = os.path.join(SAVE_DIR, "test6_parser_repair.json")

from controlkv.parsing.tool_calls import extract_tool_calls as orig_parser

t6_results = []
for ex in FIXED_12:
    input_ids = _encode_tool(ex, tokenizer, CTX)
    set_seed(42)
    cache = DynamicCache()
    toks, _ = greedy_with_logits(model15, input_ids, cache,
                                  max_new_tokens=64, eos_id=EOS_ID)
    decoded = tokenizer.decode(toks, skip_special_tokens=True)

    ov, oc = orig_parser(decoded)
    rv, rc = parse_tool_calls_v2(decoded)
    t6_results.append({
        "id": ex["id"], "category": ex["category"],
        "raw_output": decoded,
        "ground_truth": ex.get("ground_truth", []),
        "orig_valid": ov, "orig_calls": oc,
        "orig_correct": evaluate_correctness(oc, ex.get("ground_truth", [])),
        "repaired_valid": rv, "repaired_calls": rc,
        "repaired_correct": evaluate_correctness(rc, ex.get("ground_truth", [])),
    })

n = len(t6_results)
print(f"  Original parser : valid={sum(r['orig_valid'] for r in t6_results)}/{n}  "
      f"correct={sum(r['orig_correct'] for r in t6_results)}/{n}", flush=True)
print(f"  Repaired parser : valid={sum(r['repaired_valid'] for r in t6_results)}/{n}  "
      f"correct={sum(r['repaired_correct'] for r in t6_results)}/{n}", flush=True)

with open(FILE_T6, "w", encoding="utf-8") as f:
    json.dump(t6_results, f, indent=2)
print(f"Test 6 saved → {FILE_T6}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — VERSION RECORD
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- TEST 7: Version Record ---", flush=True)
FILE_T7 = os.path.join(SAVE_DIR, "test7_versions.json")
import subprocess
try:
    pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
except Exception as e:
    pip_out = f"Error: {e}"

t7 = {
    "versions": VERSION_INFO,
    "pip_freeze": pip_out,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "note": (
        "To test a prior release, create a separate conda env with: "
        "pip install transformers==4.46.0 optimum-quanto==0.2.7 "
        "and rerun with the same prompts."
    ),
}
with open(FILE_T7, "w") as f:
    json.dump(t7, f, indent=2)
print(f"Test 7 saved → {FILE_T7}", flush=True)

print(f"\n=== ALL DIAG2 TESTS COMPLETED. Artifacts in: {SAVE_DIR} ===", flush=True)
