# Claim Audit
| Claim | Evidence Source | Methodology | Confidence |
|-------|-----------------|-------------|------------|
| 4-bit causes tool collapse | `runs/phase3/agent_a_backend/backend_matrix.csv` | Qwen2.5-1.5B tool valid rate drops from 1.0 to 0.0 with HQQ 4-bit | High |
| Coherence remains high | `runs/phase3/agent_a_backend/backend_matrix.csv` | Ordinary text coherence is 50% vs 0% tool rate | Medium |
| Values are safe to quantize | `runs/phase3/agent_b_localization/b_ablation_results.jsonl` | `v_only_int4` maintains 1.0 valid rate, while `k_only_int4` drops to 0.0 | High |
| Early layers drive collapse | `runs/phase3/agent_b_localization/b_ablation_results.jsonl` | `block_0_6_int4` collapses to 0.0, all other blocks maintain 1.0 | High |
