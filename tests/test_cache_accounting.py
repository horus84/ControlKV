import pytest
from controlkv.cache.accounting import compute_logical_kv_cache_bytes

def test_cache_accounting_full():
    # 28 layers, 4 kv heads, 128 head dim, 512 seq len -> float16 (16 bits = 2 bytes)
    # elements per layer = 2 * 1 * 4 * 512 * 128 = 524,288
    # total elements = 28 * 524,288 = 14,680,064
    # total bytes = 14,680,064 * 2 = 29,360,128 bytes (~29.36 MB)
    bytes_full = compute_logical_kv_cache_bytes(28, 4, 128, 512, "full")
    assert bytes_full == 29360128

def test_cache_accounting_int4():
    # int4 (4 bits = 0.5 bytes)
    bytes_int4 = compute_logical_kv_cache_bytes(28, 4, 128, 512, "quanto_int4")
    assert bytes_int4 == 29360128 // 4

def test_cache_accounting_int2():
    # int2 (2 bits = 0.25 bytes)
    bytes_int2 = compute_logical_kv_cache_bytes(28, 4, 128, 512, "quanto_int2")
    assert bytes_int2 == 29360128 // 8
