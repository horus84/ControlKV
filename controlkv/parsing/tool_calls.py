import json
import re
from typing import List, Dict, Any, Optional, Tuple
from controlkv.parsing.normalization import normalize_json_arguments

def extract_tool_calls(text: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Parse structured tool calls from model generated output string.

    Returns:
        Tuple of (is_valid_structured_output: bool, parsed_action_list: List[Dict[str, Any]])
    """
    if not text or not text.strip():
        return False, []

    cleaned = text.strip()

    # Pattern 1: Qwen style <tool_call> ... </tool_call>
    tool_call_matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", cleaned, re.DOTALL)
    if tool_call_matches:
        parsed_calls = []
        for match in tool_call_matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and "name" in obj:
                    parsed_calls.append({
                        "name": str(obj.get("name", "")),
                        "arguments": normalize_json_arguments(obj.get("arguments", {}))
                    })
            except Exception:
                continue
        if parsed_calls:
            return True, parsed_calls

    # Pattern 2: Markdown ```json ... ``` blocks
    code_block_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    # Pattern 3: Direct JSON array or single JSON object
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            parsed_calls = []
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    parsed_calls.append({
                        "name": str(item.get("name", "")),
                        "arguments": normalize_json_arguments(item.get("arguments", {}))
                    })
            if parsed_calls:
                return True, parsed_calls
    except Exception:
        pass

    # Pattern 4: Embedded JSON array match
    array_match = re.search(r"(\[\s*\{.*?\}\s*\])", cleaned, re.DOTALL)
    if array_match:
        try:
            data = json.loads(array_match.group(1))
            if isinstance(data, list):
                parsed_calls = []
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        parsed_calls.append({
                            "name": str(item.get("name", "")),
                            "arguments": normalize_json_arguments(item.get("arguments", {}))
                        })
                if parsed_calls:
                    return True, parsed_calls
        except Exception:
            pass

    return False, []
