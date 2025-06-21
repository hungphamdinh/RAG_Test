# history_utils.py
import json, os, uuid
from datetime import datetime
from constant.constant import CHROMA_PATH, RETRIEVAL_METHOD, CONFIG_PATH, MEMORY_PATH, HISTORY_FILE


def load_history():
    """Load chat history from disk if available."""
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}
    
def save_history(chat_history):
    """Persist chat history to disk."""
    os.makedirs(MEMORY_PATH, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

def add_to_history(speaker: str, text: str, CURRENT_MODULE, chat_history):
    """Append a speaker’s message to history with a unique key."""
    key = str(uuid.uuid4())
    chat_history[key] = {
        "speaker": speaker,
        "text": text,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "module": CURRENT_MODULE
    }


def delete_history_entry(key: str, chat_history):
    """Remove a history entry by its unique key."""
    if key in chat_history:
        del chat_history[key]
    else:
        print(f"No history entry found for key: {key}")
