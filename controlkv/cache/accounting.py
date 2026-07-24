from typing import Dict, Any

def compute_logical_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    cache_condition: str,
    batch_size: int = 1
) -> int:
    """Calculate the theoretical logical memory size of the KV cache in bytes.

    Args:
        num_layers: Total Transformer layers in the model.
        num_kv_heads: Key-Value head count (Grouped Query Attention).
        head_dim: Dimension per head.
        seq_len: Prompt + generated sequence length in tokens.
        cache_condition: ['full', 'offloaded', 'quanto_int4', 'quanto_int2']
        batch_size: Inference batch size (default: 1).

    Returns:
        Logical cache footprint in bytes.
    """
    bits_per_element = {
        "full": 16,
        "offloaded": 16,
        "quanto_int4": 4,
        "quanto_int2": 2
    }
    bits = bits_per_element.get(cache_condition, 16)
    
    total_elements = 2 * batch_size * num_layers * num_kv_heads * seq_len * head_dim
    total_bits = total_elements * bits
    return total_bits // 8
