# src/populate/populate_database.py

import argparse
import os
from langchain.schema.document import Document
from src.vector_store.manage_vector_store import (
    initialize_vector_store,
    clear_database,
    add_documents_to_vector_store,
    calculate_chunk_ids
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Updated import

def load_documents(source_dir: str = "unit_tests", file_extensions: list = [".test.js"]) -> list[Document]:
    """
    Loads documents from the specified directory with given file extensions.

    Args:
        source_dir (str): Directory containing the documents.
        file_extensions (list): List of file extensions to include.

    Returns:
        list[Document]: A list of Document objects with content and metadata.
    """
    documents = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Optionally process content with LangGraph's tools
                    # processed_content = parse_code_sections(content)  # Uncomment if applicable
                    processed_content = content  # Use raw content if no processing
                    metadata = {
                        "source": file,
                        "file_path": file_path,
                        # Add more metadata fields as needed
                    }
                    documents.append(Document(page_content=processed_content, metadata=metadata))
    return documents

def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 80) -> list[Document]:
    """
    Splits documents into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list[Document]: List of Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def main():
    parser = argparse.ArgumentParser(description="Populate the Chroma vector store.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--embedding_service", type=str, default="ollama", help="Embedding service to use.")
    parser.add_argument("--source_dir", type=str, default="unit_tests", help="Directory containing documents to load.")
    parser.add_argument("--file_extensions", type=str, nargs='+', default=[".test.js"], help="List of file extensions to include.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Initialize vector store
    db = initialize_vector_store()

    # Load and process documents
    documents = load_documents(source_dir=args.source_dir, file_extensions=args.file_extensions)
    print(f"📄 Loaded {len(documents)} documents.")
    
    chunks = split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks.")
    
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add to vector store
    add_documents_to_vector_store(db, chunks_with_ids)
    print("✅ Database populated successfully.")

if __name__ == "__main__":
    main()