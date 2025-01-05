# src/code_chains.py

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # Corrected import
from langchain.chains import LLMChain  # Import LLMChain
from pydantic import BaseModel, Field

from src.graph_state import GraphState  # Ensure this import is correct


class CodeSolution(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


def create_ollama_chain() -> LLMChain:
    """
    Creates a code generation chain using Ollama's llama3 model.

    Returns:
        LLMChain: The configured code generation chain.
    """
    # Define the prompt template
    code_gen_prompt_ollama = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a coding assistant with expertise in LCEL, LangChain expression language. 
Here is a full set of LCEL documentation:  
-------
{context}
-------
Answer the user question based on the above provided documentation. Ensure any code you provide can be executed 
with all required imports and variables defined. Structure your answer strictly in JSON with the following fields: 
prefix, imports, code. Provide the JSON within triple backticks and specify the language as JSON. Ensure the JSON is well-formed with proper syntax and delimiters. Here is the user question:
""",
            ),
            ("user", "{messages}"),
        ]
    )

    # Initialize the Ollama LLM without max_tokens
    llm = OllamaLLM(
        model="llama3",  # Ensure this matches your Ollama setup
        temperature=0.0,  # Adjust as needed
        # max_tokens=1000,   # Removed to fix ValidationError
    )

    # Create the LLMChain
    code_gen_chain_ollama = LLMChain(
        llm=llm,
        prompt=code_gen_prompt_ollama,
        output_key="response"
    )

    return code_gen_chain_ollama


def initialize_code_gen_chains() -> LLMChain:
    """
    Initializes the code generation chain.

    Returns:
        LLMChain: The configured code generation chain.
    """
    ollama_chain = create_ollama_chain()

    return ollama_chain