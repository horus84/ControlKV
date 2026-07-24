import pytest
from controlkv.utils.reproducibility import set_seed

def test_seed_setting():
    set_seed(42)
    import random, numpy as np, torch
    v1_py = random.randint(0, 1000)
    v1_np = np.random.randint(0, 1000)
    v1_t = torch.randint(0, 1000, (1,)).item()

    set_seed(42)
    v2_py = random.randint(0, 1000)
    v2_np = np.random.randint(0, 1000)
    v2_t = torch.randint(0, 1000, (1,)).item()

    assert v1_py == v2_py
    assert v1_np == v2_np
    assert v1_t == v2_t
