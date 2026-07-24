#!/bin/bash
set -e
echo "Starting H100 Environment Setup"
python -m pip install --upgrade pip
pip install -r migration/requirements_cloud.txt
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import optimum; print(f'Optimum: {optimum.__version__}')"
python -c "import hqq; print(f'HQQ: {hqq.__version__}')"
mkdir -p archive/runs archive/results
python migration/run_smoke_test.py
echo "Environment Ready"
