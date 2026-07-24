import torch
from typing import Tuple, Dict, Any, Optional
from transformers.cache_utils import (
    DynamicCache,
    QuantizedCache
)

def get_kv_cache(
    cache_condition: str,
    model_config: Optional[Any] = None
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Factory function returning the cache object or cache kwargs for model.generate().

    Args:
        cache_condition: One of ['full', 'offloaded', 'quanto_int4', 'quanto_int2']
        model_config: Model configuration object required for QuantizedCache.

    Returns:
        Tuple of (cache_instance_or_none, generate_kwargs_dict)
    """
    if cache_condition == "full":
        cache_obj = DynamicCache()
        return cache_obj, {"past_key_values": cache_obj}
        
    elif cache_condition == "offloaded":
        cache_obj = DynamicCache(offloading=True)
        return cache_obj, {"past_key_values": cache_obj}

    elif cache_condition == "quanto_int4":
        if model_config is None:
            raise ValueError("model_config is required for quanto_int4 QuantizedCache")
        cache_obj = QuantizedCache(
            backend="quanto",
            config=model_config,
            nbits=4
        )
        return cache_obj, {"past_key_values": cache_obj}

    elif cache_condition == "quanto_int2":
        if model_config is None:
            raise ValueError("model_config is required for quanto_int2 QuantizedCache")
        cache_obj = QuantizedCache(
            backend="quanto",
            config=model_config,
            nbits=2
        )
        return cache_obj, {"past_key_values": cache_obj}

    else:
        raise ValueError(f"Unknown cache condition: {cache_condition}")
