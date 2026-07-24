import hashlib
import json
from typing import Any, Dict

def hash_object(obj: Any) -> str:
    """Compute deterministic SHA-256 hash for a JSON-serializable object or string."""
    if isinstance(obj, str):
        data_bytes = obj.encode("utf-8")
    else:
        data_bytes = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()

def hash_file(file_path: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
