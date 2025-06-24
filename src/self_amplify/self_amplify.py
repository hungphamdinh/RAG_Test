import json
import time
import pandas as pd
import torch
from langchain.vectorstores.chroma import Chroma
from constant.constant import CHROMA_PATH, RETRIEVAL_METHOD, CONFIG_PATH
from utils.history_utils import load_history
from get_embedding_function import get_embedding_function
from utils.query_utils import QueryRewriter, CrossEncoderRanker
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import DeepLift, LayerDeepLift, KernelShap, LLMAttribution, TextTemplateInput
import numpy as np
import logging
from self_amplifier import self_amplifier
import torch.nn as nn


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)


class CustomWrapper(nn.Module):
    def __init__(self, model):
        super(CustomWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Return logits for the last token
        return self.model(x).logits[:, -1, :]

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
        self.hf_tokenizer = AutoTokenizer.from_pretrained("ministral/Ministral-3b-instruct")
        self.hf_model     = AutoModelForCausalLM.from_pretrained(
            "ministral/Ministral-3b-instruct", low_cpu_mem_usage=True, device_map="auto"
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.hf_model.to(self.device)

    def preprocess(self, prompt: str, with_bracket: bool = True):
        """
        Take a raw question string and return (prompt_str, idx_tensor),
        where idx_tensor is what Mistral expects to generate the next token.
        If with_bracket=True, we wrap the prompt with [INST]... </s> so that
        Mistral outputs a single-letter answer (e.g. "A").
        """
        if with_bracket:
            # Wrap the question and inject an assistant prefix so Mistral will output a single-letter answer
            formatted = (
                f"[INST]\n{prompt}[/INST]\n"
                "Please choose ONE of the following options (A, B, C, or D) and respond with that letter only.\n"
                "The answer is ("
            )
        else:
            formatted = prompt
        enc = self.hf_tokenizer(formatted, return_tensors="pt", padding=False, add_special_tokens=False)
        idx_tensor = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)
        return formatted, idx_tensor, attn_mask
    
    def generate_rationale(self, model, tokenizer, idx, target, explainer, topk_words):
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
        # Fallback if start token not found
        if index_min is None:
            index_min = 0

        # find index_max (just before "</s>")
        index_max = None
        for j in range(index_min, len(input_ids) - len(end_tok) + 1):
            if input_ids[j : j + len(end_tok)] == end_tok:
                index_max = j - 1
                break
        # Fallback if end token not found
        if index_max is None:
            index_max = len(input_ids) - 1

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
    
    def generate_context_idx(self, model, tokenizer_instance, df, nb_shot, selection_strategy):
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
            _, idx_tensor, attn_mask = self.preprocess(prompt, with_bracket=True)
            len_input = idx_tensor.shape[1]

            # Generate exactly one token (the letter) using greedy decode (1 beam)
            t0 = time.time()
            outputs = model.generate(
                input_ids=idx_tensor.to(model.device),
                attention_mask=attn_mask.to(model.device),
                max_new_tokens=1,
                pad_token_id=self.hf_tokenizer.eos_token_id,
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
            model=self.hf_model,
            tokenizer_instance=self.hf_tokenizer,
            df=df, nb_shot=nb_shot,
            selection_strategy=strategy
        )
        if len(idxs) < nb_shot:
            print('Failed with strategy error -> switch to random')
            idxs = self.generate_context_idx(
                model=self.hf_model,
                tokenizer_instance=self.hf_tokenizer,
                df=df, nb_shot=nb_shot,
                selection_strategy='random'
            )
        return idxs

    def generate_few_shot_rationales(self, df: pd.DataFrame, shot_indices: list[int], fewshot_map: dict, args) -> list[str]:
        """Step 8: Generate rationales for each selected example."""
        results = []
        captum_map = {
            'deeplift':'DeepLift','ig':'LayerIntegratedGradients','grad_act':'LayerGradientXActivation',
            'kernel_shap':'KernelShap','lime':'Lime','shap':'ShapleyValues','shap_s':'ShapleyValueSampling',
            'random':'random'
        }
        for idx in shot_indices:
            q   = df.at[idx,'question']
            key = df.at[idx,'AnswerKey']
            if args.explainer in ('self_topk','self_exp','auto_cot'):
                amp = self_amplifier(model=self.hf_model, tokenizer=self.hf_tokenizer, device=self.device)
                if args.explainer=='self_topk': _, idx_tensor = amp.preprocess_self_topk(q,key,topk=3)
                elif args.explainer=='self_exp': _, idx_tensor = amp.preprocess_self_exp(q,key,n_steps=3)
                else: _, idx_tensor = amp.preprocess_auto_cot(q)
                out = self.hf_model.generate(idx_tensor.to(self.device), max_new_tokens=300, do_sample=False, num_beams=1)
                raw = self.hf_tokenizer.decode(out[0][idx_tensor.shape[1]:], skip_special_tokens=True).strip()
                full = fewshot_map.get(raw,raw)
                results.append(f"Q: {q}\nA: {full}")
                continue
            _, idx_tensor, _ = self.preprocess(q, with_bracket=False)
            expl = captum_map.get(args.explainer,'DeepLift')
            kws = self.generate_rationale(model=self.hf_model,tokenizer=self.hf_tokenizer,idx=idx_tensor,target=key,explainer=expl,topk_words=3)
            kw_str = ', '.join(f"'{w}'" for w in kws[:-1]) + f" and '{kws[-1]}'"
            full   = fewshot_map.get(key,key)
            results.append(f"Q: {q}\nA: The 3 keywords {kw_str} are important to predict ({key}); full handler list: {full}.")
        return results

    def build_prompt(self, system_ctx: str, fewshots: list[str], context: str, history: str, question: str, module_ctx) -> str:
        """Step 9: Assemble the final prompt."""
        prompt = f"System Instructions:\n{system_ctx}\n\n{module_ctx}\n\n"
        prompt += "Few-Shot Examples:\n" + '\n\n'.join(fewshots) + '\n\n'
        prompt += f"Relevant Context:\n{context}\n\n"
        prompt += f"Conversation History:\n{history}\n\n"
        prompt += f"User Question:\n{question}\n\n"
        prompt += "Please think step by step and explain your reasoning clearly.\nStep 1:"
        return prompt
