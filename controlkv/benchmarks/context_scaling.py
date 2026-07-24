from typing import Any, Tuple

NEUTRAL_DISTRACTOR_TEXT = (
    "The following is background technical documentation for general reference purposes. "
    "System architecture guidelines specify modular design patterns, clean separation of concerns, "
    "and robust error handling mechanisms across microservices. Service endpoints communicate via "
    "gRPC and REST APIs using structured payload schemas with strict validation protocols. "
    "Database connections rely on connection pooling, automatic failover configurations, and "
    "read-replica load balancing to optimize query latency and system throughput. "
    "Logging frameworks capture timestamped events, trace identifiers, and diagnostic metrics "
    "to facilitate continuous monitoring and automated alert triggers. "
)

def build_distractor_tokens(tokenizer: Any, target_token_count: int) -> list[int]:
    """Generate exact token IDs for neutral distractor padding."""
    base_distractor_ids = tokenizer.encode(NEUTRAL_DISTRACTOR_TEXT, add_special_tokens=False)
    repeated_ids = []
    while len(repeated_ids) < target_token_count:
        repeated_ids.extend(base_distractor_ids)
    return repeated_ids[:target_token_count]

def scale_context_to_target(
    prompt: str,
    target_tokens: int,
    tokenizer: Any
) -> Tuple[str, str, int]:
    """Scale prompt to target token length using deterministic token-level neutral padding.

    Args:
        prompt: Base formatted tool prompt string.
        target_tokens: Target token count (512, 1024, 2048).
        tokenizer: Model tokenizer.

    Returns:
        Tuple of (scaled_prompt_text, distractor_text_added, actual_token_count)
    """
    base_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    base_len = len(base_token_ids)

    if base_len >= target_tokens:
        return prompt, "", base_len

    needed_tokens = target_tokens - base_len
    distractor_ids = build_distractor_tokens(tokenizer, needed_tokens)
    distractor_text = tokenizer.decode(distractor_ids, skip_special_tokens=True)

    # Prepend neutral distractor text cleanly
    scaled_prompt = f"Background Context:\n{distractor_text}\n\n{prompt}"
    actual_token_ids = tokenizer.encode(scaled_prompt, add_special_tokens=False)
    actual_len = len(actual_token_ids)

    return scaled_prompt, distractor_text, actual_len
