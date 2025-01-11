
# src/state_graph.py

import logging
import json
import re
from typing import Dict
import ast

from langchain.chains import LLMChain
from pydantic import ValidationError
from typing import Dict, Any
from src.graph_state import GraphState
from src.code_chains import CodeSolution, initialize_code_gen_chains
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # Corrected import
from langchain.chains import LLMChain  # Import LLMChain
from langgraph.graph import END, StateGraph, START  # Ensure correct imports

import logging
import json
import re
from typing import Optional, Dict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("state_graph.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
max_iterations = 3
flag = "do not reflect"


def build_workflow() -> StateGraph:
    """
    Builds the StateGraph workflow.

    Args:
        concatenated_content (str): The context/content for the model.

    Returns:
        StateGraph: The compiled workflow.
    """
    workflow = StateGraph(GraphState)

    # Initialize the code generation chain
    code_gen_chain = initialize_code_gen_chains()

    # Define the nodes
    workflow.add_node("generate", lambda state: generate(state, code_gen_chain))
    workflow.add_node("check_code", code_check)      # Check code
    workflow.add_node("reflect", reflect)            # Reflect on errors

    # Define the edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "reflect": "reflect",
            "generate": "generate",
        },
    )
    workflow.add_edge("reflect", "generate")

    # Compile the workflow
    compiled_workflow = workflow.compile()
    return compiled_workflow
    
def wrap_imports_in_quotes_directly(json_string):
    """
    Fixes the 'imports' field by directly replacing its value
    with a preformatted transformation.
    
    Args:
        json_string (str): The input JSON string.
        
    Returns:
        str: The modified JSON string with the fixed 'imports' field.
    """
    return json_string.replace(
        '"imports": ["langchain"],',
        '"imports": "["langchain"]",'
    )

def extract_json(code_solution_response: str) -> Optional[Dict]:
    """
    Extracts JSON from the LLM response. The JSON can be enclosed within triple backticks and
    a 'json' language specifier or provided directly.

    Args:
        code_solution_response (str): The raw response from the LLM.

    Returns:
        Optional[Dict]: The parsed JSON dictionary if extraction and parsing are successful, else None.
    """
    # Define regex patterns
    patterns = [
        r'```json\s*(\{.*?\})\s*```',  # JSON within ```json ... ```
        r'```(\{.*?\})```',            # JSON within ``` ... ```
        r'(\{.*?\})'                    # Bare JSON
    ]

    code_solution_json = None

    for pattern in patterns:
        json_match = re.search(pattern, code_solution_response, re.DOTALL | re.IGNORECASE)
        if json_match:
            code_solution_json = json_match.group(1)
            logger.debug(f"Matched JSON using pattern '{pattern}':\n{code_solution_json}")
            break  # Stop at the first successful match

    if not code_solution_json:
        # If no pattern matches, attempt to extract JSON from the entire response
        logger.warning("No JSON delimiters found. Attempting to parse the entire response as JSON.")
        code_solution_json = code_solution_response.strip()

    # # Attempt to fix common formatting issues
    # if '"""' in code_solution_json:
    #     logger.debug("Found triple double-quotes in JSON. Replacing them with single double-quotes.")
    #     code_solution_json = code_solution_json.replace('"""', '"')
    # Remove any leading/trailing newlines and unnecessary whitespace
    code_solution_json = code_solution_json.strip()
#     code_solution_json_transform =  code_solution_json.replace(
#     '["',
#     '"""["'
# ).replace('"]', '"]"""')
    
    code_solution_json_parse = f"""{json.dumps(code_solution_json, indent=4)}"""

    try:
        # Parse the JSON string into a dictionary
        code_solution_dict = json.loads(code_solution_json_parse)
        # print(f"Successfully parsed JSON Dictionary:\n{code_solution_dict}")
        return code_solution_dict
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        logger.debug(f"LLM Response for debugging:\n{code_solution_response}")
        return None

def validate_code_solution(code_solution_dict: Dict) -> bool:
    """
    Validates that the code solution dictionary contains the required fields.

    Args:
        code_solution_dict (Dict): The parsed JSON dictionary.

    Returns:
        bool: True if valid, False otherwise.
    """
    required_fields = {'prefix', 'imports', 'code'}
    if not required_fields.issubset(code_solution_dict.keys()):
        missing = required_fields - code_solution_dict.keys()
        logger.error(f"Missing required fields in code solution: {missing}")
        return False
    return True

def generate_code_solution(code_solution_response: str) -> Optional[Dict]:
    """
    Processes the LLM response to extract and validate the code solution.

    Args:
        code_solution_response (str): The raw response from the LLM.

    Returns:
        Optional[Dict]: The validated code solution dictionary or None if invalid.
    """
    code_solution_dict = extract_json(code_solution_response)

    # if not code_solution_dict:
    #     print("Failed to extract JSON from the LLM response.")
    #     return None

    # if not validate_code_solution(code_solution_dict):
    #     print("Code solution validation failed.")
    #     return None
    
    return code_solution_dict

def generate(state: GraphState, code_gen_chain: LLMChain):
    logger.info("---GENERATING CODE SOLUTION---")

    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
    concatenated_context = state.get("context", "No context provided.")  # Fetch context from state

    # Serialize messages into a single string
    serialized_messages = "\n".join([f"{role}: {content}" for role, content in messages])

    code_solution_response = code_gen_chain.run({
            "context": concatenated_context,
            "messages": serialized_messages  # Pass serialized messages
            # "messages": [("user", serialized_messages)]
    })
    print(code_solution_response)

    
# Use this when your prompt has structure {prefix, imports, code}, this one need your output always stable for working
def generate_with_workflow(state: GraphState, code_gen_chain: LLMChain):
    logger.info("---GENERATING CODE SOLUTION---")

    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
    concatenated_context = state.get("context", "No context provided.")  # Fetch context from state

    # Serialize messages into a single string
    serialized_messages = "\n".join([f"{role}: {content}" for role, content in messages])

    try:
        # Run the LLM chain
        code_solution_response = code_gen_chain.run({
            "context": concatenated_context,
            "messages": serialized_messages  # Pass serialized messages
            # "messages": [("user", serialized_messages)]
        })
 
        prefix_pattern = r'"prefix":\s*"([^"]*)"'

    # 2️⃣ Regex pattern to extract 'imports' as a list
        imports_pattern = r'"imports":\s*(\[[^\]]*\])'

        # 3️⃣ Regex pattern to extract 'code' (multi-line string)
        code_pattern = r'"code":\s*`([\s\S]*?)`'

        # Extract values using regex
        prefix_match = re.search(prefix_pattern, code_solution_response)
        imports_match = re.search(imports_pattern, code_solution_response)
        code_match = re.search(code_pattern, code_solution_response)

        # Extracted values or default to None
        prefix = prefix_match.group(1) if prefix_match else None
        imports = json.loads(imports_match.group(1)) if imports_match else None
        code = code_match.group(1) if code_match else None

        # print("Prefix:", prefix)
        # print("Imports:", imports)
        # print("Code:\n", code)
        # Create a CodeSolution instance
        code_solution = CodeSolution(prefix= prefix, imports= f"""{imports}""", code = code)

        # Store the CodeSolution in state
        state["generation"] = code_solution
        state["iterations"] = iterations + 1
        state["error"] = "no"
        return state

    except ValueError as ve:
        logger.error(f"Error during code solution generation: {ve}")
        messages += [("user", f"Error: {ve}. Please ensure the LLM provides a valid JSON response.")]
        state["error"] = "yes"
        state["iterations"] = iterations + 1
        return state
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        messages += [("user", f"An unexpected error occurred: {e}")]
        state["error"] = "yes"
        state["iterations"] = iterations + 1
        return state

def code_check(state: GraphState) -> Dict:
    """
    Check code by executing imports and code block.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict: Updated state with error status.
    """
    logger.info("---CHECKING CODE---")

    # messages = state["messages"]
    # code_solution = state["generation"]
    # iterations = state["iterations"]

    # if not isinstance(code_solution, CodeSolution):
    #     logger.error("Code solution is not an instance of CodeSolution.")
    #     messages += [("user", "Invalid code solution format. Cannot perform code checks.")]
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "iterations": iterations,
    #         "error": "yes",
    #     }

    # imports = code_solution.imports
    # code = code_solution.code

    # # Define a restricted execution environment
    # safe_globals = {}
    # safe_locals = {}

    # # Define allowed built-ins
    # allowed_builtins = {
    #     'print': print,
    #     'len': len,
    #     'range': range,
    #     # Add other safe built-ins as needed
    # }
    # safe_globals['__builtins__'] = allowed_builtins

    # # Check imports
    # try:
    #     exec(imports, safe_globals, safe_locals)
    # except Exception as e:
    #     logger.info("---CODE IMPORT CHECK: FAILED---")
    #     error_message = [("user", f"Your solution failed the import test: {e}")]
    #     messages += error_message
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "iterations": iterations,
    #         "error": "yes",
    #     }

    # # Check execution
    # try:
    #     exec(code, safe_globals, safe_locals)
    # except Exception as e:
    #     logger.info("---CODE BLOCK CHECK: FAILED---")
    #     error_message = [("user", f"Your solution failed the code execution test: {e}")]
    #     messages += error_message
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "iterations": iterations,
    #         "error": "yes",
    #     }

    # # No errors
    # logger.info("---NO CODE TEST FAILURES---")
    # return {
    #     "generation": code_solution,
    #     "messages": messages,
    #     "iterations": iterations,
    #     "error": "no",
    # }


def reflect(state: GraphState) -> Dict:
    """
    Reflect on errors to improve the code solution.

    Args:
        state (GraphState): The current graph state.

    Returns:
        Dict: Updated state with reflections.
    """
    logger.info("---REFLECTING ON ERRORS---")

    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    # Define the reflection prompt
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Based on the previous errors, provide insights and suggestions to improve the code solution. Structure your answer strictly in JSON with the following fields: 
suggestions, explanations. Provide the JSON within triple backticks and specify the language as JSON. Do not include any additional text, explanations, or comments."""
            ),
            ("user", "{messages}"),
        ]
    )

    # Initialize the reflection LLM
    reflection_llm = OllamaLLM(
        model="llama3",
        temperature=0.0,
    )
    reflection_chain = LLMChain(
        llm=reflection_llm,
        prompt=reflection_prompt,
        output_key="response"
    )

    try:
        # Invoke the reflection chain using run
        reflections_response = reflection_chain.run({"messages": messages})

        # Log the raw response for debugging
        logger.debug(f"Raw Reflection Response: {reflections_response}")

        # Ensure the response is a string
        if not isinstance(reflections_response, str):
            raise TypeError(f"Expected string response from LLM, got {type(reflections_response)}")

        # Extract JSON from the response using regex
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', reflections_response, re.DOTALL)
        if json_match:
            reflections_json = json_match.group(1)
            logger.debug(f"Extracted JSON:\n{reflections_json}")
        else:
            # Attempt to parse entire response if not within backticks
            reflections_json = reflections_response.strip()
            logger.debug(f"No backticks found. Using entire response for JSON parsing:\n{reflections_json}")

        # Parse the JSON string into a dictionary
        reflections_dict = json.loads(reflections_json)

        # Extract suggestions and explanations
        suggestions = reflections_dict.get("suggestions", "No suggestions provided.")
        explanations = reflections_dict.get("explanations", "No explanations provided.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed during reflection: {e}")
        logger.debug(f"Reflection LLM Response:\n{reflections_response}")
        messages += [("user", f"Failed to parse reflection JSON: {e}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations + 1,
            "error": "yes",
        }
    except ValidationError as e:
        logger.error(f"Pydantic validation failed during reflection: {e}")
        logger.debug(f"Reflections Dict:\n{reflections_dict}")
        messages += [("user", f"Invalid reflection format: {e}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations + 1,
            "error": "yes",
        }
    except TypeError as e:
        logger.error(f"Type error during reflection: {e}")
        messages += [("user", f"Unexpected reflection response type: {e}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations + 1,
            "error": "yes",
        }
    except Exception as e:
        logger.error(f"Unexpected error during reflection: {e}")
        messages += [("user", f"An unexpected error occurred during reflection: {e}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations + 1,
            "error": "yes",
        }

    # Add reflections to messages
    messages += [
        (
            "assistant",
            f"Suggestions: {suggestions}\nExplanations: {explanations}",
        )
    ]

    iterations += 1

    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def decide_to_finish(state: GraphState) -> str:
    """
    Determines whether to finish the workflow or retry.

    Args:
        state (GraphState): The current graph state.

    Returns:
        str: Next node to call ("end", "reflect", or "generate").
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations >= max_iterations:
        logger.info("---DECISION: FINISH---")
        return "end"
    else:
        logger.info("---DECISION: RE-TRY SOLUTION---")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"

            