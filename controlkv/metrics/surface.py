from typing import List, Dict, Any

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_surface_metrics(
    cand_text: str,
    full_text: str,
    cand_token_ids: List[int],
    full_token_ids: List[int]
) -> Dict[str, Any]:
    """Compute surface text agreement metrics against full precision cache output."""
    exact_string_match = (cand_text == full_text)
    
    max_len = max(len(cand_text), len(full_text))
    if max_len == 0:
        normalized_edit_sim = 1.0
    else:
        edit_dist = levenshtein_distance(cand_text, full_text)
        normalized_edit_sim = 1.0 - (edit_dist / max_len)

    exact_token_match = (cand_token_ids == full_token_ids)
    
    if not cand_token_ids and not full_token_ids:
        token_agreement_rate = 1.0
    elif len(cand_token_ids) == len(full_token_ids):
        matching_tokens = sum(1 for a, b in zip(cand_token_ids, full_token_ids) if a == b)
        token_agreement_rate = matching_tokens / len(cand_token_ids)
    else:
        min_len = min(len(cand_token_ids), len(full_token_ids))
        max_t_len = max(len(cand_token_ids), len(full_token_ids))
        matching_tokens = sum(1 for a, b in zip(cand_token_ids[:min_len], full_token_ids[:min_len]) if a == b)
        token_agreement_rate = matching_tokens / max_t_len

    return {
        "exact_output_string_agreement": exact_string_match,
        "normalized_edit_similarity": normalized_edit_sim,
        "exact_token_agreement": exact_token_match,
        "token_agreement_rate": token_agreement_rate
    }
