import pandas as pd
import argparse
from langchain_community.llms.ollama import Ollama
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

_llama_client = None
def get_llama_client():
    global _llama_client
    if _llama_client is None:
        _llama_client = Ollama(model="mistral:7b")
    return _llama_client


self_amp = SelfAmplify()
CURRENT_MODULE = None
chat_history   = {}
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
    - All API handling related with Booking resides in the `useBooking` hook under `Context/Booking/Hooks`.
    """.strip(),
    # Add more modules as needed
}
def invoke_model(prompt: str) -> str:
    logging.info("Step 8/8: Invoking mistral to generate final answer")
    client = get_llama_client()
    return client.invoke(prompt)

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

    # # Optional benchmarking
    # if args.benchmark:
    #     amp = self_amplifier(model=hf_model, tokenizer=hf_tokenizer, device=device)
    #     captum_map = {
    #         "deeplift": "DeepLift",
    #         "ig": "LayerIntegratedGradients",
    #         "grad_act": "LayerGradientXActivation",
    #         "kernel_shap": "KernelShap",
    #         "lime": "Lime",
    #         "shap": "ShapleyValues",
    #         "shap_s": "ShapleyValueSampling",
    #         "random": "random"
    #     }
    #     logging.info("Running benchmark (evaluate_fs_with_exp)...")
    #     df_test = pd.read_csv("fewshot_test.csv")
    #     bench = amp.evaluate_fs_with_exp(
    #         df_train=df_fewshot,
    #         df_test=df_test,
    #         model=hf_model,
    #         max_new_tokens=1,
    #         explainer=captum_map.get(args.explainer, "DeepLift"),
    #         idx_list_fs=shot_indices,
    #         topk_words=3,
    #         split_dict=None
    #     )
    #     print(bench)

    # 8) Construct final prompt for Llama2, embedding Mistral rationales directly
    prompt = self_amp.build_prompt(GENERAL_CONTEXT + '\n' + MODULE_CONTEXTS.get(module,''), fewshot_strings, context_text, conversation, query_text, module_ctx)
    print('prompt', prompt)

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
