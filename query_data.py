import pandas as pd
import argparse
from llama_cpp import Llama
from utils.query_utils import QueryRewriter, CrossEncoderRanker
from self_amplifier import self_amplifier
from constant.constant import CHROMA_PATH, RETRIEVAL_METHOD
from utils.history_utils import load_history, save_history, add_to_history, delete_history_entry
import logging
import time

from get_embedding_function import get_embedding_function
from src.self_amplify.self_amplify import SelfAmplify
import os, json

# 5) Prepare simplified few-shot DataFrame and mapping by loading from JSON
_fs_path = os.path.join(os.path.dirname(__file__), "few_shot/tm.json")
with open(_fs_path, "r") as _fsf:
    _fs_data = json.load(_fsf)
df_fewshot  = pd.DataFrame(_fs_data["examples"])
fewshot_map = _fs_data["fewshot_map"]

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

def get_llama_client():
    # Reuse the llama_client already instantiated in the SelfAmplify singleton
    return self_amp.llama_client


self_amp = SelfAmplify()
CURRENT_MODULE = None
chat_history   = {}
GENERAL_CONTEXT = ("You are a master of React Native and JavaScript, with the expertise of a senior Technical Architect. You know every detail of this codebase. Your mission is to assist the user by explaining any part they don’t understand.")

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
    - All API handling related with Booking resides in the `useBooking` hook under `Context/Booking/Hooks`.
    """.strip(),
    # Add more modules as needed
}
def invoke_model(prompt: str) -> str:
    logging.info("Step 8/8: Invoking local llama to generate final answer")
    client = get_llama_client()
    # Use llama_cpp client to generate text with tuned parameters
    resp = client(
        prompt,
        max_tokens=2048,
        temperature=0.2,
    )
    return resp["choices"][0]["text"].strip()

# ------------------------------------------------------------
# The main RAG + Self-AMPlIFY pipeline:
# ------------------------------------------------------------
def query_rag(query_text: str, module: str = None, args=None):
    # Step 0: Load chat history and module config
    chat_history = self_amp.load_chat_history()
    module = self_amp.persist_module_config(module)

    # 1) Rewrite the user query with Flan-T5 (modularized)
    rewritten_query = self_amp.rewrite_query(query_text)

    # 2 & 3) Retrieve context and rerank (modularized)
    docs, context_text = self_amp.retrieve_context(rewritten_query, module)
    results = [(d, None) for d in docs]

    # 4) Build conversation history for this module
    conversation = self_amp.build_conversation(module)
    module_ctx = MODULE_CONTEXTS.get(module, "")

    # 6) Select few-shot examples (e.g. those Mistral got wrong)
    logging.info("Step 4/6: Selecting few-shot examples with Mistral-3B")
    start_time = time.time()
    shot_indices = self_amp.select_few_shot_indices(df_fewshot, 3, 'error')
    elapsed = time.time() - start_time
    logging.info(f"   → Selected indices {shot_indices} (took {elapsed:.2f}s)")

    # 7) Generate rationales for each selected example
    logging.info("Step 5/6: Generating rationales with explainer '%s'", args.explainer)
    fewshot_strings = self_amp.generate_few_shot_rationales(df_fewshot, shot_indices, fewshot_map, args)

    # 8) Construct final prompt for Llama2, embedding Mistral rationales directly
    prompt = self_amp.build_prompt(GENERAL_CONTEXT + '\n' + MODULE_CONTEXTS.get(module,''), fewshot_strings, context_text, conversation, query_text, module_ctx)

    # (Optional) Log the prompt size
    token_count = len(prompt.strip().split())
    logging.info(f"   → Prompt length: {token_count} tokens")

    # Invoke Llama-2 for the final answer (modularized)
    response_text = invoke_model(prompt)
    logging.info("   → mistral final answer generated")

    add_to_history("Assistant", response_text, CURRENT_MODULE, chat_history)
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
    parser.add_argument(
        "--explainer",
        choices=[
            "deeplift", "ig", "grad_act",
            "kernel_shap", "lime", "shap", "shap_s",
            "self_topk", "self_exp", "auto_cot", "random"
        ],
        default="deeplift",
        help="Which explanation strategy to use for few-shot rationales"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="If set, run evaluate_fs_with_exp to benchmark on a test set"
    )
    args = parser.parse_args()

    global CURRENT_MODULE, RETRIEVAL_METHOD, chat_history
    CURRENT_MODULE = args.module
    RETRIEVAL_METHOD = args.retrieval_method
    chat_history = load_history() or {}

    if args.delete_history:
        delete_history_entry(args.delete_history, chat_history)
        return

    query_text = args.query_text
    add_to_history("User", query_text, CURRENT_MODULE, chat_history)
    _ = query_rag(query_text, args.module, args=args)

    save_input = input("Store this conversation to memory? (y/n): ").strip().lower()
    if save_input.startswith("y"):
        save_history(chat_history)
        print("History saved.")


if __name__ == "__main__":
    main()
