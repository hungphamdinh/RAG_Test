# src/tests/test_llm_chain.py

from src.code_chains import initialize_code_gen_chains

def test_llm_chain():
    code_gen_chain = initialize_code_gen_chains()
    print(f"Code Gen Chain Type: {type(code_gen_chain)}")  # Confirm the chain type

    try:
        response = code_gen_chain.run(
            {"context": "Sample context about LCEL and LangChain.", "messages": "How do I build a RAG chain in LCEL?"}
        )
        print(f"LLM Chain Response:\n{response}")
        print(f"Type of response: {type(response)}")
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_llm_chain()