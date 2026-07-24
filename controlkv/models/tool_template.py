import json
from typing import List, Dict, Any

def format_qwen_tool_prompt(
    tools: List[Dict[str, Any]],
    user_query: str,
    tokenizer: Any,
    distractor_text: str = ""
) -> str:
    """Format tools and user query into a standard Qwen2.5 chat template prompt.
    Prepend optional distractor text to the user prompt if needed for context scaling.
    """
    system_content = (
        "You are a helpful assistant with access to the following functions. "
        "Use them if required:\n\n"
        f"{json.dumps(tools, indent=2)}\n\n"
        "Return function calls as a JSON list of objects with 'name' and 'arguments' keys."
    )
    
    full_user_content = f"{distractor_text}\n\n{user_query}".strip() if distractor_text else user_query

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": full_user_content}
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{full_user_content}<|im_end|>\n<|im_start|>assistant\n"
        
    return prompt
