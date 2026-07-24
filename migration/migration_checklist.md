# Cloud Migration Validation Checklist

## Pre-Flight
- [x] Repository frozen and tagged (`phase3_windows_baseline`)
- [x] Local outputs moved to `archive/`
- [x] `.gitignore` updated to prevent tracking temporary files
- [x] `README_CLOUD.md` generated with launch instructions
- [x] Exact dependencies pinned in `requirements_cloud.txt`
- [x] HQQ missing dependency appended

## Setup Scripts
- [x] `setup_kaggle.sh` prepared and verified
- [x] `setup_h100.sh` prepared and verified
- [x] Scripts contain CUDA/dependency verification steps

## Dataset Validation
- [x] `datasets_manifest.json` generated with file paths, sizes, and hashes
- [x] Dataset structure matches cloud expectations

## Execution Validation
- [x] `run_smoke_test.py` prepared (tests model load, generation, metrics)
- [x] Entrypoint command documented with `--resume` parameter
- [x] `progress_schema.json` defined for checkpointing and resuming

**Status:** READY FOR CLOUD EXECUTION
