# Cloud Migration Guide for ControlKV

## Repository Overview
This repository contains the ControlKV benchmarking suite. It evaluates the impact of KV-Cache quantization on structured tool decisions.

## Installation
Use the provided setup scripts for automated installation on cloud VMs.
- Kaggle Dual T4: `bash migration/setup_kaggle.sh`
- H100 Instance: `bash migration/setup_h100.sh`

## Requirements
- CUDA 12.1+
- Python 3.10+
- Torch >= 2.6.0
- Transformers >= 5.14.1
- Optimum >= 2.2.0
- Optimum-Quanto >= 0.2.7
- HQQ >= 0.1.7.post3

## Expected Directory Structure
```text
ControlKV/
├── archive/              # Older results and logs
├── configs/              # Model and benchmark YAML configs
├── controlkv/            # Core benchmark framework
├── migration/            # Cloud deployment scripts & docs
├── scripts/              # Entrypoints for benchmarking
├── tests/                # Smoke tests and unit tests
└── README_CLOUD.md       # This file
```

## How to Run Smoke Tests
```bash
python migration/run_smoke_test.py
```

## How to Resume Benchmarks
The benchmark automatically skips completed runs.
```bash
python scripts/phase3_agent_c_benchmark.py \
    --models qwen0.5,qwen1.5,smollm1.7 \
    --contexts 512 1024 2048 \
    --conditions dynamic offloaded hqq8 hqq4 quanto4 quanto2 \
    --resume
```
