# Does KV-Cache Quantization Preserve Tool Decisions?

An ACL Rolling Review (ARR August 3, 2026) evaluation study investigating how KV-cache quantization impacts structured tool decision correctness, surface text agreement, and memory/latency trade-offs in instruction-tuned language models.

## Overview
- **Models**: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`
- **Cache Conditions**:
  1. `full` (`DynamicCache` - full precision baseline)
  2. `offloaded` (`OffloadedCache` - lossless system control)
  3. `quanto_int4` (`QuantizedCache` - Quanto 4-bit)
  4. `quanto_int2` (`QuantizedCache` - Quanto 2-bit)
- **Benchmark**: Fixed 150-example subset of single-turn Berkeley Function Calling Leaderboard (BFCL)
  - 50 Simple examples
  - 50 Multiple-function examples
  - 50 Parallel-function examples
- **Context Scaling**: ~512, ~1,024, and ~2,048 tokens via deterministic token-level neutral padding.

## Quick Start

```bash
# Install package in editable mode
pip install -e .

# Run test suite
pytest tests/

# Step 1: Build fixed benchmark subset
python scripts/build_subset.py

# Step 2: Run full-cache smoke test (20 examples determinism check)
python scripts/smoke_generate.py --config configs/smoke.yaml

# Step 3: Run main experimental matrix
python scripts/run_experiment.py --config configs/qwen15b_main.yaml
python scripts/run_experiment.py --config configs/qwen3b_main.yaml

# Step 4: Aggregate results and generate statistical analysis
python scripts/aggregate_results.py

# Step 5: Programmatically generate paper tables and figures
python scripts/make_paper_tables.py
```

## Repository Structure
See `STATUS.md` for project history, environment details, and execution verification.
