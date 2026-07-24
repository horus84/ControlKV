import json
from typing import Any, Dict

def normalize_json_arguments(args: Any) -> Dict[str, Any]:
    """Recursively normalize JSON dictionary keys and values for clean comparison."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {"raw_arg": args}

    if not isinstance(args, dict):
        return {}

    normalized = {}
    for k, v in sorted(args.items()):
        norm_key = str(k).strip()
        if isinstance(v, float):
            # Round floats to 4 decimal places for stable floating point comparison
            normalized[norm_key] = round(v, 4)
        elif isinstance(v, dict):
            normalized[norm_key] = normalize_json_arguments(v)
        elif isinstance(v, list):
            normalized[norm_key] = [
                normalize_json_arguments(x) if isinstance(x, dict) else (round(x, 4) if isinstance(x, float) else x)
                for x in v
            ]
        else:
            normalized[norm_key] = v
    return normalized
