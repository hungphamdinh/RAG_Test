# src/vector_store/manage_vector_store.py

import os
import shutil
from langchain.vectorstores.chroma import Chroma
from langchain.schema.document import Document
from src.embeddings.get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def initialize_vector_store(embedding_service: str = "ollama"):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def add_documents_to_vector_store(db: Chroma, documents: list[Document]):
    db.add_documents(documents)
    db.persist()

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)  # Default to 0 if page not present
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks