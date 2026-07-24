import torch
from typing import Dict, Any

class CUDAMemoryTracker:
    """Tracks PyTorch CUDA memory allocated, reserved, and peak usage."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = (device.startswith("cuda") and torch.cuda.is_available())

    def reset(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def get_snapshot(self) -> Dict[str, int]:
        pass

    def get_metrics(self) -> Dict[str, Any]:
        if not self.is_cuda:
            return {
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
                "current_allocated_bytes": 0,
                "current_reserved_bytes": 0,
            }
        
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "current_allocated_bytes": torch.cuda.memory_allocated(),
            "current_reserved_bytes": torch.cuda.memory_reserved(),
        }
