# src/query/query_handler.py

import argparse
from src.vector_store.manage_vector_store import initialize_vector_store
from src.state_graph import build_workflow
from src.graph_state import GraphState
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("query_handler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def query_rag_with_workflow(query_text: str, concatenated_content: str):
    """
    Handles the query using the StateGraph workflow.

    Args:
        query_text (str): The user's query.
        concatenated_content (str): The context/content for the model.
    """
    logger.info("Initializing StateGraph workflow.")

    # Build the workflow
    workflow = build_workflow()

    print('concatenated_content', concatenated_content)
    # Initialize state with context
    state: GraphState = {
        "error": "no",
        "messages": [("user", query_text)],
        "generation": "",
        "iterations": 0,
        "context": concatenated_content  # Added context
    }

    # Invoke the workflow
    logger.info("Starting the StateGraph workflow.")
    final_state = workflow.invoke(state)

    # Extract and display the final code solution
    code_solution = final_state.get("generation", "")
    if code_solution:
        formatted_response = (
            f"Description: {code_solution.prefix}\n"
            f"Imports:\n{code_solution.imports}\n\n"
            f"Code:\n{code_solution.code}"
        )
        print(formatted_response)
        return code_solution
    else:
        # logger.error("No code solution generated.")
        # print("No solution could be generated based on the provided context.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Query the RAG system with LangGraph workflow.")
    parser.add_argument("query_text", nargs='+', type=str, help="The query text.")
    args = parser.parse_args()

    # Join the list of words into a single string
    query = ' '.join(args.query_text)
    logger.info(f"Received query: {query}")

    # Initialize the vector store
    db = initialize_vector_store()

    # Perform similarity search
    logger.info(f"Performing similarity search for query: {query}")
    results = db.similarity_search_with_score(query, k=10)

    # Construct context from retrieved documents
    concatenated_content = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    logger.info("Constructed context from retrieved documents.")

    # Perform the query with the workflow
    query_rag_with_workflow(query_text=query, concatenated_content=concatenated_content)

if __name__ == "__main__":
    main()