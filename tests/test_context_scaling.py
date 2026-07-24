import pytest
from controlkv.benchmarks.context_scaling import scale_context_to_target

class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 100 for c in text]
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join([chr(65 + (t % 26)) for t in token_ids])

def test_context_scaling_token_length():
    tok = MockTokenizer()
    prompt = "What is the weather in Tokyo?"
    scaled_prompt, distractor, actual_len = scale_context_to_target(prompt, 100, tok)
    assert actual_len >= 100
    assert "What is the weather in Tokyo?" in scaled_prompt
