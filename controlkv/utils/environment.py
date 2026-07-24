import sys
import platform
import json
import torch
from typing import Dict, Any

def capture_environment_metadata() -> Dict[str, Any]:
    """Capture snapshot of python, cuda, torch, transformers, and hardware environment."""
    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["device_count"] = torch.cuda.device_count()
        metadata["device_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory

    packages = ["transformers", "optimum", "optimum.quanto", "datasets", "accelerate", "scipy", "pandas", "numpy"]
    pkg_dict = {}
    for p in packages:
        try:
            mod = __import__(p)
            pkg_dict[p] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkg_dict[p] = "missing"
    metadata["installed_packages"] = pkg_dict
    return metadata
