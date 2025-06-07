import pandas as pd
import argparse
import json
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from transformers import TextIteratorStreamer
import threading
from queue import Empty

import logging
import time
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

import re
import os
import uuid
from datetime import datetime

from get_embedding_function import get_embedding_function
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# For post hoc explanations:
import numpy as np
from captum.attr import DeepLift, LayerDeepLift, KernelShap, LLMAttribution, TextTemplateInput
import torch.nn as nn

class CustomWrapper(nn.Module):
    def __init__(self, model):
        super(CustomWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Return logits for the last token
        return self.model(x).logits[:, -1, :]
    
# Device setup (use MPS on Mac if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------------------------------------
# Tokenizer & Model loading
# ------------------------------------------------------------
# Query Rewriter uses Flan-T5
tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-small")
model_t5     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
model_t5.to(device)

# Cross-Encoder reranker (MonoT5 or similar)
# We'll pass its tokenizer into CrossEncoderRanker
# (AutoTokenizer & AutoModelForSequenceClassification as needed later)

# Hugging Face Mistral 7B (instruction-tuned variant) for few-shot selection & rationale
HF_MISTRAL_REPO = "ministral/Ministral-3b-instruct"
hf_tokenizer     = AutoTokenizer.from_pretrained(HF_MISTRAL_REPO)
hf_model = AutoModelForCausalLM.from_pretrained(
    HF_MISTRAL_REPO,
    low_cpu_mem_usage=True,
    device_map="cpu"
)
# hf_model         = AutoModelForCausalLM.from_pretrained(HF_MISTRAL_REPO, low_cpu_mem_usage=True).to(device)

# ------------------------------------------------------------
# Preprocessing helper (standalone, replicates self_amplifier.preprocess logic)
# ------------------------------------------------------------
def preprocess(prompt: str, with_bracket: bool = True):
    """
    Take a raw question string and return (prompt_str, idx_tensor),
    where idx_tensor is what Mistral expects to generate the next token.
    If with_bracket=True, we wrap the prompt with [INST]... </s> so that
    Mistral outputs a single-letter answer (e.g. "A").
    """
    if with_bracket:
        # Use Mistral’s chat template: [INST]\n<question></s>
        formatted = f"[INST]\n{prompt}</s>"
    else:
        formatted = prompt
    enc = hf_tokenizer(formatted, return_tensors="pt", padding=False)
    idx_tensor = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    return formatted, idx_tensor, attn_mask

# ------------------------------------------------------------
# QueryRewriter and CrossEncoderRanker (unchanged except naming)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Helper: Generate top-k keyword rationale via DeepLift or KernelShap
# ------------------------------------------------------------
def generate_rationale(model, tokenizer, idx, target, explainer, topk_words):
    """
    Use Captum's DeepLift or KernelShap to compute token attributions for the
    final token's logit, aggregate to word level, and return the topk words.
    - model: a Hugging Face CausalLM with .forward() returning logits
    - tokenizer: matching tokenizer for that model
    - idx: a (1 x seq_len) tensor, input_ids already on device
    - target: the single-token string label (e.g. "A")
    - explainer: "DeepLift" or "KernelShap"
    - topk_words: number of keywords to return
    """
    # 1) Identify question span: everything between "[INST]\n" and "</s>"
    input_ids = idx[0].tolist()
    start_tok = tokenizer.encode("[INST]\n", add_special_tokens=False)
    end_tok   = tokenizer.encode("</s>", add_special_tokens=False)

    # find index_min
    index_min = None
    for i in range(len(input_ids) - len(start_tok) + 1):
        if input_ids[i : i + len(start_tok)] == start_tok:
            index_min = i + len(start_tok)
            break

    # find index_max (just before "</s>")
    index_max = None
    for j in range(index_min, len(input_ids) - len(end_tok) + 1):
        if input_ids[j : j + len(end_tok)] == end_tok:
            index_max = j - 1
            break

    # 2) Create a baseline by replacing [index_min:index_max] tokens with pad_id
    baseline = idx.clone()
    pad_id = tokenizer.pad_token_id
    baseline[0, index_min : index_max + 1] = pad_id

    # 3) Define forward function that returns last-token logits
    def forward_fn(input_ids_tensor):
        out = model(input_ids_tensor.to(model.device))
        # return shape: (batch, vocab_size) for the final position
        return out.logits[:, -1, :]


    if explainer == "DeepLift":
        # Wrap the model so that it returns next-token logits
        wrapper = CustomWrapper(model)
        # Use LayerDeepLift with the wrapper and embedding layer
        lfi = LayerDeepLift(wrapper, model.get_input_embeddings())
        attr = lfi.attribute(
            inputs=idx.to(model.device),
            baselines=baseline.to(model.device),
            target=tokenizer.encode(target, add_special_tokens=False)[0],
        )

        # 4) Sum over embedding dim → get shape (1, seq_len)
        attributions = attr.sum(dim=2).detach().cpu().numpy()[0, index_min : index_max + 1]
        attributions = attributions / (np.sum(attributions) + 1e-12)

        # 5) Map subtokens to decoded words
        subtokens = [tokenizer.decode([tid]).strip() for tid in input_ids[index_min : index_max + 1]]
        decoded_words = tokenizer.decode(idx[0, index_min : index_max + 1]).split()

        word_attribs = []
        k = 0
        buffer = ""
        accum = 0.0
        for subidx, subtoken in enumerate(subtokens):
            buffer += subtoken.replace(" ", "")
            accum += attributions[subidx]
            if buffer == decoded_words[k]:
                # Zero out if stopword
                if decoded_words[k].lower() in {"the", "a", "to", "is", "of", "on", "in", "and"}:
                    accum = 0.0
                word_attribs.append((decoded_words[k], accum))
                k += 1
                buffer = ""
                accum = 0.0

        # 6) Pick top-k words by descending attribution
        word_attribs.sort(key=lambda x: -x[1])
        topk_words_list = [w for w, _ in word_attribs[:topk_words]]
        return topk_words_list

    elif explainer == "KernelShap":
        ks = KernelShap(forward_fn)

        # Build a template: replace each word in the question with "{}"
        question_text = tokenizer.decode(idx[0, index_min : index_max + 1])
        words = question_text.split()
        placeholder = " ".join(["{}"] * len(words))
        full_prompt = tokenizer.decode(idx[0])
        template = full_prompt.replace(question_text, placeholder)

        inp = TextTemplateInput(template=template, values=words)
        attr_res = LLMAttribution(ks, tokenizer).attribute(inp, **{"n_samples": 64})

        word_attrs = np.array(attr_res.seq_attr)
        tokens_clean = words
        attrs_clean  = word_attrs

        # Zero out stopwords
        stopwords = {"the", "a", "to", "is", "of", "on", "in", "and"}
        for i, w in enumerate(tokens_clean):
            if w.lower() in stopwords:
                attrs_clean[i] = 0.0

        topk_idx = np.argpartition(-attrs_clean, topk_words)[:topk_words]
        topk_words_list = [tokens_clean[i] for i in sorted(topk_idx)]
        return topk_words_list

    else:
        raise ValueError(f"Unsupported explainer: {explainer}")


# ------------------------------------------------------------
# In-memory chat history utilities
# ------------------------------------------------------------
CURRENT_MODULE = None
chat_history   = {}
CHROMA_PATH    = "chroma"
CONFIG_PATH    = ".rag_config.json"
MEMORY_PATH    = "memory"
HISTORY_FILE   = os.path.join(MEMORY_PATH, "history.json")
RETRIEVAL_METHOD = "mmr"
GENERAL_CONTEXT = """
You are a master of React Native and JavaScript, with the expertise of a senior Technical Architect.
You know every detail of this codebase. Your mission is to assist the user by explaining any part they don’t understand.

- Cite any file and line you reference using “[file.py:42]”.
- Format all code snippets inside Markdown triple backticks.
- Do not invent behavior outside the provided context; if unsure, reply “I don’t have enough context to answer that.”
- Hooks (in the Context folder) define functions that dispatch Redux actions and handle API calls (e.g., useBooking, useTaskManagement).
- If a function in a Context hook (e.g., useBooking) is wrapped with `withLoadingAndErrorHandling` or `withErrorHandling`, it denotes an API-calling handler.
- To identify API-handling functions, first inspect `Context/<Module>/Hooks/<useHook>.js` for those wrappers, then check action constants in `Context/<Module>/Actions.js` (e.g., `ADD_TASK`, `UPDATE_TASK`).
- Reducers (in the Context folder) store and update global state based on dispatched actions.
- Actions (in the Context folder) define the action type constants used throughout hooks and reducers.
- Answers should be concise yet complete, providing necessary information without exceeding token limits.
- For any module, the general flow is: UI screens live under `Screens/<Module>`; API logic resides in hooks under `Context/<Module>/Hooks/<useHook>` (and any hooks those import); action types in `Context/<Module>/Actions.js`; and state updates in `Context/<Module>/Reducers.js`.
""".strip()

MODULE_CONTEXTS = {
    "TaskManagement": """
    Module: TaskManagement
    - Located under `Screens/TaskManagement`, which includes:
      • A Home screen with list view and filter functionality.
      • A TaskDetail screen for adding and editing tasks.
    - All API handling related with TaskManagement resides in the `useTaskManagement` hook under `Context/TaskManagement/Hooks`.
    """.strip(),
    "Booking": """
    Module: Booking
    - Hooks: useBooking provides createBooking, cancelBooking.
    - Services: bookingService.create, bookingService.search.
    - Utils: formatDate, calculateDepositPrice in bookingHelpers.
    """.strip(),
    # Add more modules as needed
}


def load_history():
    """Load chat history from disk if available."""
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_history():
    """Persist chat history to disk."""
    os.makedirs(MEMORY_PATH, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)


def add_to_history(speaker: str, text: str):
    """Append a speaker’s message to history with a unique key."""
    key = str(uuid.uuid4())
    chat_history[key] = {
        "speaker": speaker,
        "text": text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "module": CURRENT_MODULE
    }


def delete_history_entry(key: str):
    """Remove a history entry by its unique key."""
    if key in chat_history:
        del chat_history[key]
    else:
        print(f"No history entry found for key: {key}")


# ------------------------------------------------------------
# Few-shot selection: pick indices based on success/error/random
# ------------------------------------------------------------
def generate_context_idx(model, tokenizer_instance, df, nb_shot, selection_strategy):
    """
    Select `nb_shot` examples from `df` such that either:
      - 'error': model’s single-token answer != df.AnswerKey AND answer in answer_keys
      - 'success': model’s single-token answer == df.AnswerKey
      - 'random': random sample
    Returns a list of integer indices from df.index.
    """
    shot_list = []
    examples_found = 0
    answer_keys = np.sort(df["AnswerKey"].unique()).tolist()
    attempts = 0

    logging.info(f"→ Starting few-shot selection (need {nb_shot} examples).")
    while examples_found < nb_shot:
        attempts += 1
        if attempts % 5 == 0:
            logging.info(f"   Checked {attempts} candidates so far, found {examples_found} valid shots.")

        # Sample a random example not already chosen
        i = df[df.index.isin(shot_list) == False].sample(n=1, replace=True).index[0]
        prompt = df.at[i, "question"]
        target = df.at[i, "AnswerKey"]

        # Tokenize example question for Mistral
        _, idx_tensor, attn_mask = preprocess(prompt, with_bracket=True)
        len_input = idx_tensor.shape[1]

        # Generate exactly one token (the letter) using greedy decode (1 beam)
        t0 = time.time()
        outputs = model.generate(
            input_ids=idx_tensor.to(model.device),
            attention_mask=attn_mask.to(model.device),
            max_new_tokens=1,
            pad_token_id=hf_tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1
        )
        t1 = time.time()
        logging.info(f"     → Iteration {attempts}: generate() took {(t1 - t0):.2f}s")

        answer = tokenizer_instance.decode(
            outputs[0][len_input :], skip_special_tokens=True
        ).upper()

        # Check selection criteria
        if selection_strategy == "error":
            if (target != answer) and (answer in answer_keys):
                shot_list.append(i)
                examples_found += 1
                logging.info(f"       • Chose index {i} (prompt ‘{prompt}’ → ‘{answer}’).")
        elif selection_strategy == "success":
            if (target == answer) and (answer in answer_keys):
                shot_list.append(i)
                examples_found += 1
        elif selection_strategy == "random":
            shot_list = df.sample(n=nb_shot, replace=True).index.tolist()
            examples_found = nb_shot
        else:
            break

        # Prevent infinite loops
        if attempts >= 50 and examples_found < nb_shot:
            logging.warning(f"   → Stopped after {attempts} attempts; only {examples_found} examples found.")
            break

    logging.info(f"→ Finished few-shot selection: picked indices {shot_list} after {attempts} attempts.")
    return shot_list


# ------------------------------------------------------------
# The main RAG + Self-AMPlIFY pipeline:
# ------------------------------------------------------------
def query_rag(query_text: str, module: str = None):
    # Reload history
    global chat_history
    chat_history = load_history() or {}

    # If module passed, persist it; else load previous
    if module:
        with open(CONFIG_PATH, "w") as f:
            json.dump({"module": module}, f)
    else:
        try:
            module = json.load(open(CONFIG_PATH)).get("module")
        except:
            module = None

    # 1) Rewrite the user query with Flan-T5
    logging.info("Step 1/6: Rewriting query with Flan-T5")
    start_time = time.time()
    rewriter = QueryRewriter("google/flan-t5-small")
    rewritten_query = rewriter.rewrite(query_text)
    elapsed = time.time() - start_time
    logging.info(f"   → Rewritten query = \"{rewritten_query}\"  (took {elapsed:.2f}s)")

    # 2) Retrieve documents from Chroma
    logging.info("Step 2/6: Retrieving documents from Chroma")
    start_time = time.time()
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    query_embedding = embedding_function.embed_query(rewritten_query)

    if RETRIEVAL_METHOD == "similarity":
        docs = db.similarity_search_by_vector(query_embedding, k=10)
    else:
        docs = db.max_marginal_relevance_search_by_vector(
            query_embedding, k=10, fetch_k=100, lambda_mult=0.7
        )
    elapsed = time.time() - start_time
    logging.info(f"   → Retrieved {len(docs)} documents (took {elapsed:.2f}s)")

    all_contents = [d.page_content for d in docs]
    # 3) Rerank with CrossEncoder
    logging.info("Step 3/6: Re-ranking with CrossEncoder")
    start_time = time.time()
    ranker = CrossEncoderRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_contents = ranker.rerank(rewritten_query, all_contents)[:10]
    docs = [d for d in docs if d.page_content in top_contents]
    results = [(d, None) for d in docs]
    elapsed = time.time() - start_time
    logging.info(f"   → Reranked to top {len(docs)} docs (took {elapsed:.2f}s)")

    if module:
        results = [(d, s) for d, s in results if module in d.metadata.get("source", "")]

    context_text = "\n\n---\n\n".join([d.page_content for d, _ in results])

    # 4) Build conversation history for this module
    filtered = [e for e in chat_history.values() if e.get("module") == module]
    conversation = "\n".join(f"{e['speaker']}: {e['text']}" for e in filtered)
    module_ctx = MODULE_CONTEXTS.get(module, "")

    # 5) Prepare few-shot DataFrame (must match exactly Mistral's output format)
    df_fewshot = pd.DataFrame([
        {"question": "Which planet is known as the Red Planet?", "AnswerKey": "A"},
        {"question": "When did the Berlin Wall fall?",        "AnswerKey": "B"},
        {"question": "Who wrote 'Pride and Prejudice'?",      "AnswerKey": "C"},
    ])

    # 6) Select few-shot examples (e.g. those Mistral got wrong)
    logging.info("Step 4/6: Selecting few-shot examples with Mistral-3B")
    start_time = time.time()
    shot_indices = generate_context_idx(
        model=hf_model,
        tokenizer_instance=hf_tokenizer,
        df=df_fewshot,
        nb_shot=3,
        selection_strategy="random"
    )
    elapsed = time.time() - start_time
    logging.info(f"   → Selected indices {shot_indices} (took {elapsed:.2f}s)")

    # 7) Generate rationales for each selected example
    logging.info("Step 5/6: Generating rationales via Captum/DeepLift")
    fewshot_strings = []
    for idx in tqdm(shot_indices, desc="Generating shot rationales"):
        logging.info(f"   • Rationale for example #{idx}")
        ex_q = df_fewshot.at[idx, "question"]
        ex_a = df_fewshot.at[idx, "AnswerKey"]

        # Tokenize example question
        _, idx_tensor, _ = preprocess(ex_q, with_bracket=True)

        # Top-3 keywords via DeepLift
        start_r = time.time()
        keywords = generate_rationale(
            model=hf_model,
            tokenizer=hf_tokenizer,
            idx=idx_tensor,
            target=ex_a,
            explainer="DeepLift",
            topk_words=3
        )
        elapsed_r = time.time() - start_r
        logging.info(f"     – Keywords = {keywords} (took {elapsed_r:.2f}s)")

        # Format as "Q: ...  A: The 3 keywords 'X', 'Y', and 'Z' are important to predict ..."
        kw_str = ""
        for i, w in enumerate(keywords):
            w_clean = w.replace(".", "").strip()
            if i == len(keywords) - 1:
                kw_str += "and " + f"'{w_clean}'"
            else:
                kw_str += f"'{w_clean}', "

        rationale_line = (
            f"Q: {ex_q}\n"
            f"A: The 3 keywords {kw_str} are important to predict that the answer is ({ex_a})."
        )
        fewshot_strings.append(rationale_line)

    # 8) Construct final prompt for Llama2, embedding Mistral rationales directly
    prompt = (
        # (a) System instructions and module context
        f"System Instructions:\n{GENERAL_CONTEXT}\n\n{module_ctx}\n\n"
        # (b) Few-Shot Examples and their Mistral rationales
        "Few-Shot Examples:\n"
    )
    # Append each few-shot rationale (which already contains "Q: ... A: ..." lines)
    for rationale in fewshot_strings:
        prompt += f"{rationale}\n\n"

    # Continue with retrieved context and conversation
    prompt += (
        f"Relevant Context:\n{context_text}\n\n"
        f"Conversation History:\n{conversation}\n\n"
        # (c) Finally, the new user question
        f"User Question:\n{query_text}\n\n"
        "Please think step by step and explain your reasoning clearly.\n"
        "Step 1:"
    )

    # (Optional) Log the prompt size
    token_count = len(prompt.strip().split())
    logging.info(f"   → Prompt length: {token_count} tokens")

    # Invoke Llama-2 for the final answer
    logging.info("   → Invoking Llama-2 to generate final answer")
    llama = Ollama(model="llama2:7b")
    response_text = llama.invoke(prompt)
    logging.info("   → Llama-2 final answer generated")

    add_to_history("Assistant", response_text)
    sources = [d.metadata.get("id") for d, _ in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--module", type=str, help="Module to scope context (e.g. TaskManagement).")
    parser.add_argument("--delete-history", type=str, help="Delete a history entry by its key.")
    parser.add_argument(
        "--retrieval-method",
        choices=["mmr", "similarity"],
        default="mmr",
        help="Choose retrieval method: 'mmr' or 'similarity'."
    )
    args = parser.parse_args()

    global CURRENT_MODULE, RETRIEVAL_METHOD, chat_history
    CURRENT_MODULE = args.module
    RETRIEVAL_METHOD = args.retrieval_method
    chat_history = load_history() or {}

    if args.delete_history:
        delete_history_entry(args.delete_history)
        return

    query_text = args.query_text
    add_to_history("User", query_text)
    _ = query_rag(query_text, args.module)

    save_input = input("Store this conversation to memory? (y/n): ").strip().lower()
    if save_input.startswith("y"):
        save_history()
        print("History saved.")


if __name__ == "__main__":
    main()
