import time
import torch
from typing import Dict, Any, Tuple

class CUDATimer:
    """Timer using CUDA events for precise prefill and decode latency measurements."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = (device.startswith("cuda") and torch.cuda.is_available())

    def time_execution(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Execute a function and return (result, latency_seconds)."""
        if self.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            result = func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()

            latency_ms = start_event.elapsed_time(end_event)
            return result, latency_ms / 1000.0
        else:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            t1 = time.perf_counter()
            return result, (t1 - t0)
