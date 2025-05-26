import argparse
import json
from langchain.vectorstores.chroma import Chroma
#
from langchain_community.llms.ollama import Ollama

import re
import os
import uuid
from datetime import datetime

from get_embedding_function import get_embedding_function
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification

# Device setup for Mac M1 (use MPS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class QueryRewriter:
    """
    Rewrite or expand the raw user query using a seq2seq model (e.g. T5).
    """
    def __init__(self, model_name: str):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        # Move model to chosen device
        self.model.to(device)

    def rewrite(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding="longest")
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated = self.model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

class CrossEncoderRanker:
    """
    Rerank a list of passages by relevance to the query using a cross-encoder.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Move model to chosen device
        self.model.to(device)

    def rerank(self, query: str, passages: list[str]) -> list[str]:
        # Return empty list early if there are no passages to rank
        if not passages:
            return []
        enc = self.tokenizer([query] * len(passages), passages, truncation=True, padding=True, return_tensors="pt")
        # Move inputs to device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            # Handle both binary-classifier (2 logits) and regression (1 logit) models
            if logits.size(1) == 2:
                scores = logits[:, 1].cpu().numpy()
            else:
                scores = logits.squeeze(-1).cpu().numpy()
        ranked = sorted(zip(passages, scores), key=lambda x: -x[1])
        return [p for p, _ in ranked]

# Current module context for history entries
CURRENT_MODULE = None

# Retrieval method: 'mmr' or 'similarity'
RETRIEVAL_METHOD = "mmr"

# In-memory chat history for contextual memory

# File to persist chat history


CHROMA_PATH = "chroma"
CONFIG_PATH = ".rag_config.json"
# Directory for storing memory text files
MEMORY_PATH = "memory"

HISTORY_FILE = os.path.join(MEMORY_PATH, "history.json")

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

# Single history file for all modules; entries include module for filtering
chat_history = load_history() or {}

# Flag to enable history persistence
SAVE_HISTORY = False

# General and module-specific prompt contexts for RAG
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

embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Initialize query rewriter and cross-encoder ranker
rewriter = QueryRewriter("google/flan-t5-small")
ranker = CrossEncoderRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Use MonoT5 (T5-base fine-tuned on MS-MARCO) for cross-encoder re-ranking
# ranker = CrossEncoderRanker("castorini/monot5-base-msmarco")

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
        if SAVE_HISTORY:
            save_history()
    else:
        print(f"No history entry found for key: {key}")

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--module", type=str, help="Module to scope context (e.g. TaskManagement).")
    parser.add_argument("--save-history", action="store_true", help="Enable chat history loading and saving")
    parser.add_argument("--delete-history", type=str, help="Delete a history entry by its key")
    parser.add_argument(
        "--retrieval-method",
        choices=["mmr", "similarity"],
        default="mmr",
        help="Choose retrieval method: 'mmr' (default) or 'similarity'."
    )
    args = parser.parse_args()
    global SAVE_HISTORY
    SAVE_HISTORY = args.save_history
    module = args.module
    global CURRENT_MODULE
    CURRENT_MODULE = module
    global RETRIEVAL_METHOD
    RETRIEVAL_METHOD = args.retrieval_method
    global chat_history
    chat_history = load_history() or {}
    if args.delete_history:
        delete_history_entry(args.delete_history)
        return
    query_text = args.query_text
    add_to_history("User", query_text)
    if SAVE_HISTORY:
        save_history()
    query_rag(query_text, module)


def query_rag(query_text: str, module: str = None):
    # Reload history from disk if persistence is enabled
    if SAVE_HISTORY:
        global chat_history
        chat_history = load_history()
    # Determine module context, persisting choice
    if module:
        with open(CONFIG_PATH, "w") as f:
            json.dump({"module": module}, f)
    else:
        try:
            with open(CONFIG_PATH) as f:
                module = json.load(f).get("module")
        except FileNotFoundError:
            module = None

    # Use global db, prompt_template
    # Use MMR for diverse, top results with reduced payload
    fetch_k = 100  # fetch more then filter down
    mmr_k = 10     # only use top 5 after MMR
    # Rewrite the raw user query for better retrieval
    rewritten_query = rewriter.rewrite(query_text)
    # Perform Max Marginal Relevance search via vector to get documents
    # Embed the rewritten query
    query_embedding = embedding_function.embed_query(rewritten_query)
    if RETRIEVAL_METHOD == "similarity":
        docs = db.similarity_search_by_vector(
            query_embedding,
            k=mmr_k
        )
    else:
        docs = db.max_marginal_relevance_search_by_vector(
            query_embedding,
            k=mmr_k,
            fetch_k=fetch_k,
            lambda_mult=0.7
        )
    # Rerank the retrieved documents to select the top 10 most relevant
    all_contents = [doc.page_content for doc in docs]
    top_contents = ranker.rerank(rewritten_query, all_contents)[:10]
    # Filter docs to only those top-ranked
    docs = [doc for doc in docs if doc.page_content in top_contents]
    # Pair each document with a None score placeholder
    results = [(doc, None) for doc in docs]
    if module:
        results = [(doc, score) for doc, score in results if module in doc.metadata.get("source", "")]

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # Build conversation context filtered by module
    filtered = [entry for entry in chat_history.values() if entry.get("module") == module]
    conversation = "\n".join(f"{e['speaker']}: {e['text']}" for e in filtered)
    # Build prompt with general and module-specific context
    module_ctx = MODULE_CONTEXTS.get(module, "")
    prompt = (
        f"{GENERAL_CONTEXT}\n\n"
        f"{module_ctx}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Question: {query_text}\n\n"
        "Answer:"
    )

    model = Ollama(model="llama2:7b")
    response_text = model.invoke(prompt)

    add_to_history("Assistant", response_text)
    if SAVE_HISTORY:
        save_history()

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
