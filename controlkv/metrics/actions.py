from typing import List, Dict, Any, Tuple

def compare_single_action(
    predicted_call: Dict[str, Any],
    reference_call: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare a single predicted function call against a reference call."""
    pred_name = str(predicted_call.get("name", "")).strip()
    ref_name = str(reference_call.get("name", "")).strip()
    
    name_match = (pred_name == ref_name)
    
    pred_args = predicted_call.get("arguments", {})
    ref_args = reference_call.get("arguments", {})
    
    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(ref_args, dict):
        ref_args = {}

    args_exact_match = (pred_args == ref_args)
    
    ref_keys = set(ref_args.keys())
    if ref_keys:
        matching_fields = sum(1 for k in ref_keys if k in pred_args and pred_args[k] == ref_args[k])
        field_accuracy = matching_fields / len(ref_keys)
    else:
        field_accuracy = 1.0 if not pred_args else 0.0

    complete_action_match = name_match and args_exact_match

    return {
        "tool_name_match": name_match,
        "args_exact_match": args_exact_match,
        "field_accuracy": field_accuracy,
        "complete_action_match": complete_action_match
    }

def evaluate_ground_truth_correctness(
    parsed_calls: List[Dict[str, Any]],
    ground_truth_calls: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate ground-truth BFCL correctness for parsed action list."""
    if not parsed_calls or not ground_truth_calls:
        return {
            "correct": False,
            "tool_name_match": False,
            "arg_exact_match": False,
            "field_accuracy": 0.0,
            "complete_action_match": False
        }

    if len(parsed_calls) != len(ground_truth_calls):
        return {
            "correct": False,
            "tool_name_match": False,
            "arg_exact_match": False,
            "field_accuracy": 0.0,
            "complete_action_match": False
        }

    matches = [compare_single_action(p, r) for p, r in zip(parsed_calls, ground_truth_calls)]
    all_name_match = all(m["tool_name_match"] for m in matches)
    all_args_match = all(m["args_exact_match"] for m in matches)
    avg_field_acc = sum(m["field_accuracy"] for m in matches) / len(matches)
    all_action_match = all(m["complete_action_match"] for m in matches)

    return {
        "correct": all_action_match,
        "tool_name_match": all_name_match,
        "arg_exact_match": all_args_match,
        "field_accuracy": avg_field_acc,
        "complete_action_match": all_action_match
    }

def evaluate_agreement_against_full_cache(
    cand_parsed: List[Dict[str, Any]],
    full_parsed: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute agreement metrics against full precision baseline cache output."""
    if not cand_parsed and not full_parsed:
        return {
            "tool_name_agreement": True,
            "arg_exact_agreement": True,
            "complete_action_agreement": True
        }
        
    if len(cand_parsed) != len(full_parsed):
        return {
            "tool_name_agreement": False,
            "arg_exact_agreement": False,
            "complete_action_agreement": False
        }

    matches = [compare_single_action(c, f) for c, f in zip(cand_parsed, full_parsed)]
    return {
        "tool_name_agreement": all(m["tool_name_match"] for m in matches),
        "arg_exact_agreement": all(m["args_exact_match"] for m in matches),
        "complete_action_agreement": all(m["complete_action_match"] for m in matches)
    }
