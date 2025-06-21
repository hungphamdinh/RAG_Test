import os

CHROMA_PATH    = "chroma"
CONFIG_PATH    = ".rag_config.json"
MEMORY_PATH    = "memory"
HISTORY_FILE   = os.path.join(MEMORY_PATH, "history.json")
RETRIEVAL_METHOD = "mmr"
HF_MISTRAL_REPO = "ministral/Ministral-3b-instruct"

