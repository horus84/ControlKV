import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any

def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "cuda",
    torch_dtype: str = "float16"
) -> Tuple[Any, Any]:
    """Load model and tokenizer with reproducible settings."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    return model, tokenizer
