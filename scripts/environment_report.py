import sys
import os
import platform
import json
from typing import Dict, Any

def get_env_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    
    # Check PyTorch & CUDA
    try:
        import torch
        import torch.version
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_version"] = torch.version.cuda
            report["device_count"] = torch.cuda.device_count()
            report["device_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            report["gpu_total_memory_gb"] = round(total_mem, 2)
        else:
            report["cuda_version"] = None
            report["device_count"] = 0
            report["device_name"] = None
            report["gpu_total_memory_gb"] = 0
    except ImportError:
        report["torch_version"] = None
        report["cuda_available"] = False

    # Check key packages
    packages = [
        "transformers",
        "optimum",
        "quanto",
        "optimum.quanto",
        "datasets",
        "accelerate",
        "pytest",
        "scipy",
        "statsmodels",
        "pandas",
        "numpy",
        "matplotlib",
        "pydantic",
        "yaml"
    ]
    
    pkg_status: Dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            pkg_status[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkg_status[pkg] = "MISSING"
    
    report["packages"] = pkg_status
    return report

if __name__ == "__main__":
    rep = get_env_report()
    print(json.dumps(rep, indent=2))
