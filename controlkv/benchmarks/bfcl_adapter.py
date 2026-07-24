from typing import Dict, Any, List

def parse_bfcl_entry(raw_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw BFCL dataset entry into standard benchmark record."""
    entry_id = str(raw_entry.get("id", raw_entry.get("question_id", "entry_0")))
    category = raw_entry.get("category", "simple")
    question = raw_entry.get("question", raw_entry.get("prompt", ""))
    if isinstance(question, list):
        question = "\n".join([str(q.get("content", q)) if isinstance(q, dict) else str(q) for q in question])
    
    function_schemas = raw_entry.get("function", raw_entry.get("functions", []))
    ground_truth = raw_entry.get("ground_truth", raw_entry.get("answers", []))

    return {
        "id": entry_id,
        "category": category,
        "question": question,
        "tools": function_schemas,
        "ground_truth": ground_truth
    }
