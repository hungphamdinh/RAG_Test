import json
import time
import pandas as pd
from langchain.vectorstores.chroma import Chroma
from constant.constant import CHROMA_PATH, RETRIEVAL_METHOD, CONFIG_PATH
from utils.history_utils import load_history
from get_embedding_function import get_embedding_function
from utils.query_utils import QueryRewriter, CrossEncoderRanker
import numpy as np
import logging
from self_amplifier import self_amplifier
import torch


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)



class SelfAmplify:
    """
    Encapsulates all RAG pipeline logic (steps 1–6):
      1) load_chat_history
      2) persist_module_config
      3) build_conversation
      4) rewrite_query
      5) retrieve_context
      6) select_few_shot_indices
      7) generate_few_shot_rationales
      8) build_prompt
    """
    def __init__(self):
        # Initialize singletons
        self.chat_history = {}
        self.rewriter = QueryRewriter("google/flan-t5-small")
        self.ranker   = CrossEncoderRanker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.embed_fn = get_embedding_function()

        # ─── HF MODEL FOR EXPLAINABILITY ────────────────────────────────
        # This full HF model is used by Captum (needs get_input_embeddings(), etc.)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Instantiate llama_client with local GGUF Q4 model
        from llama_cpp import Llama
        self.llama_client = Llama(
            model_path="/Volumes/ExtremeSSD/DesktopSSD/AI_Models/models/mistral-7b-instruct-v0.2.Q5_0.gguf",
            n_ctx=2048,
            n_threads=4,
            use_mlock=True,
            verbose=False
        )
        # Use the quantized GGUF model for both inference and rationale
        self.hf_model = self.llama_client


    def preprocess(self, prompt: str, with_bracket: bool = True) -> str:
        """
        Wrap the prompt if needed and return a formatted string for llama_client.
        """
        if with_bracket:
            formatted = (
                f"[INST]\n{prompt}[/INST]\n"
                "Please choose ONE of the following options (A, B, C, or D) and respond with that letter only.\n"
                "The answer is ("
            )
        else:
            formatted = prompt
        return formatted
    
    def generate_rationale(self, question: str, answer: str, topk_words: int) -> list[str]:
        """
        Use the local llama_client to list the topk_words keywords from the question
        that support the answer.
        """
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Please list the top {topk_words} keywords from the question that justify the answer, "
            "separated by commas."
        )
        resp = self.llama_client(
            prompt,
            max_tokens=128,
            temperature=0.0
        )
        text = resp["choices"][0]["text"].strip()
        # Split on commas and strip whitespace
        return [kw.strip() for kw in text.split(",") if kw.strip()]



    def load_chat_history(self) -> dict:
        """Step 1: Load existing conversation history."""
        self.chat_history = load_history() or {}
        return self.chat_history

    def persist_module_config(self, module: str) -> str:
        """Step 2: Save or load current module context."""
        if module:
            with open(CONFIG_PATH, 'w') as f:
                json.dump({'module': module}, f)
            return module
        try:
            return json.load(open(CONFIG_PATH)).get('module')
        except:
            return None

    def build_conversation(self, module: str) -> str:
        """Step 3: Build chat history string for the module."""
        hist = self.chat_history
        filtered = [e for e in hist.values() if e.get('module') == module]
        return "\n".join(f"{e['speaker']}: {e['text']}" for e in filtered)

    def rewrite_query(self, query: str) -> str:
        """Step 4: Rewrite user query using Flan-T5."""
        start = time.time()
        rewritten = self.rewriter.rewrite(query)
        logging.info(f"Rewritten in {time.time()-start:.2f}s: {rewritten}")
        return rewritten
    
    def generate_context_idx(self, df, nb_shot, selection_strategy):
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
            formatted_prompt = self.preprocess(prompt, with_bracket=True)

            # Generate exactly one token using llama_cpp client
            t0 = time.time()
            resp = self.llama_client(
                formatted_prompt,
                max_tokens=1,
                temperature=0.0
            )
            t1 = time.time()
            logging.info(f"     → Iteration {attempts}: llama_cpp took {(t1 - t0):.2f}s")
            answer = resp["choices"][0]["text"].strip().upper()

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


    def retrieve_context(self, rewritten: str, module: str, k: int = 10) -> tuple[list, str]:
        """Step 5+6: Retrieve and rerank documents."""
        emb = self.embed_fn.embed_query(rewritten)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embed_fn)
        if RETRIEVAL_METHOD == 'similarity':
            docs = db.similarity_search_by_vector(emb, k=k)
        else:
            docs = db.max_marginal_relevance_search_by_vector(emb, k=k, fetch_k=100, lambda_mult=0.7)
        contents = [d.page_content for d in docs]
        top = self.ranker.rerank(rewritten, contents)[:k]
        selected = [d for d in docs if d.page_content in top]
        if module:
            selected = [d for d in selected if module in d.metadata.get('source','')]
        context = "\n\n---\n\n".join(d.page_content for d in selected)
        return selected, context

    def select_few_shot_indices(self, df: pd.DataFrame, nb_shot: int, strategy: str) -> list[int]:
        """Step 7: Select few-shot example indices."""
        idxs = self.generate_context_idx(
            df=df, nb_shot=nb_shot,
            selection_strategy=strategy
        )
        if len(idxs) < nb_shot:
            print('Failed with strategy error -> switch to random')
            idxs = self.generate_context_idx(
                df=df, nb_shot=nb_shot,
                selection_strategy='random'
            )
        return idxs

    def generate_few_shot_rationales(self, df: pd.DataFrame, shot_indices: list[int], fewshot_map: dict, args) -> list[str]:
        """Step 8: Generate rationales for each selected example, parallelized."""
        results = []
        for idx in shot_indices:
            q   = df.at[idx,'question']
            key = df.at[idx,'AnswerKey']
            # Extract keywords via llama-based rationale
            kws = self.generate_rationale(question=q, answer=key, topk_words=3)
            kw_str = ', '.join(f"'{w}'" for w in kws)
            full   = fewshot_map.get(key,key)
            results.append(f"Q: {q}\nA: The 3 keywords {kw_str} are important to predict ({key}); full handler list: {full}.")
        print('rational', results)
        return results

    def build_prompt(self, system_ctx: str, fewshots: list[str], context: str, history: str, question: str, module_ctx) -> str:
        """Step 9: Assemble the final prompt."""
        prompt = f"System Instructions:\n{system_ctx}\n\n{module_ctx}\n\n"
        prompt += "Few-Shot Examples:\n" + '\n\n'.join(fewshots) + '\n\n'
        prompt += f"Relevant Context:\n{context}\n\n"
        prompt += f"Conversation History:\n{history}\n\n"
        prompt += "Scratchpad:\nLet's think step by step:\n"
        prompt += f"User Question:\n{question}\n\n"
        prompt += "Please think step by step and explain your reasoning clearly.\nStep 1:"
        return prompt
