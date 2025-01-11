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


# ONly use this when you work with full work flow, and the output follow the structure {prefix, imports, code}
# code_gen_prompt_ollama = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     """You are a professional React Native coding assistant. 
#         Before answering, I have retrieved related information from a knowledge base. Use this information to generate the best possible answer.

#         Knowledge Context:
#         -------
#         {context}
#         -------
#         Answer the user question based on the above provided documentation. Ensure any code you provide can be executed 
#         with all required imports and variables defined. 
#         Here is the user question:
#         """
#                 ),
#                 ("user", "{messages}"),
#             ]
#         )


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
                    """You are a professional React Native coding assistant.  
        Your task is to generate **fully executable React Native code** that follows best practices.

        ### Knowledge Context:
        -------
        {context}
        -------

        ### Instructions:
        1. **Use the context provided above** to generate the most accurate and relevant code.  
        2. **Include all necessary imports** (e.g., `react`, `react-native`, and any third-party libraries).  
        3. **Write a complete component** with proper export (`export default`).  
        4. **Ensure all functions, hooks, and variables** are fully defined.  
        5. **Avoid placeholders** and provide real, executable implementations.  
        6. **Output only the executable code**—do not include explanations or comments.

        ### User Question:
        """
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