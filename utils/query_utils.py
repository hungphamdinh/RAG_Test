import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Reuse the same device setting as query_data.py (you may need to adjust the import path)
try:
    from query_data import device
except ImportError:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class QueryRewriter:
    """
    Rewrite or expand the raw user query using Flan-T5.
    """
    def __init__(self, model_name: str):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    def rewrite(self, query: str) -> str:
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs, max_length=64, num_beams=5, early_stopping=True
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


class CrossEncoderRanker:
    """
    Rerank a list of passages by relevance to the query using a cross-encoder.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def rerank(self, query: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        enc = self.tokenizer(
            [query] * len(passages),
            passages,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            if logits.size(1) == 2:
                scores = logits[:, 1].cpu().numpy()
            else:
                scores = logits.squeeze(-1).cpu().numpy()
        ranked = sorted(zip(passages, scores), key=lambda x: -x[1])
        return [p for p, _ in ranked]