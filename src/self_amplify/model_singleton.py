# at top of some shared utils file
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@lru_cache(maxsize=1)
def get_hf_tokenizer():
    return AutoTokenizer.from_pretrained("ministral/Ministral-3b-instruct")

@lru_cache(maxsize=1)
def get_hf_model():
    model = AutoModelForCausalLM.from_pretrained(
        "ministral/Ministral-3b-instruct",
        torch_dtype=torch.float16
    )
    return model.to(_DEVICE)