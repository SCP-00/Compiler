# AST_to_JSON.py
import json
from typing import Any, Dict # Removed List, Optional as ast_to_json now takes Any
# No es necesario importar todos los nodos aquí si usamos node.to_dict()
# from Nodes_AST import Node # Solo necesitaríamos Node si hacemos chequeo de tipo explícito

# ANSI color codes (can be kept if you like colored output)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def pretty_print_json(json_data: Any, indent: int = 2) -> None:
    """Print JSON data with better formatting and optional coloring."""
    try:
        # Convert the Python dictionary structure to a JSON formatted string
        json_str = json.dumps(json_data, indent=indent, ensure_ascii=False, sort_keys=False) # sort_keys=False to keep order from to_dict
        
        # Apply coloring (optional, can be disabled for non-ANSI terminals)
        # colored_json_str = colorize_json(json_str) 
        colored_json_str = json_str # Disable coloring for now if it's problematic

        print(f"\n{Colors.HEADER}{Colors.BOLD}Abstract Syntax Tree (JSON):{Colors.ENDC}")
        print(f"{Colors.UNDERLINE}{'=' * 60}{Colors.ENDC}")
        print(colored_json_str)
        print(f"{Colors.UNDERLINE}{'=' * 60}{Colors.ENDC}\n")
    except TypeError as e:
        print(f"\nError serializing data to JSON for pretty printing: {str(e)}")
        print("Data structure passed to pretty_print_json might contain non-serializable objects.")
        # print("Problematic data:", json_data) # Uncomment for debugging, can be very verbose

def ast_to_json(node_or_value: Any) -> Any:
    """
    Converts an AST node (or a list of nodes, or a primitive value)
    to a JSON serializable structure by relying on the to_dict() method of AST nodes.
    """
    # Check if the object has a 'to_dict' method (duck typing for Node-like objects)
    if hasattr(node_or_value, 'to_dict') and callable(getattr(node_or_value, 'to_dict')):
        return node_or_value.to_dict()
    elif isinstance(node_or_value, list):
        # Recursively process items in a list
        return [ast_to_json(item) for item in node_or_value]
    
    # For primitive types (int, str, bool, float, None) or other directly serializable objects
    return node_or_value

def save_ast_to_json(ast_serializable_data: Any, filename: str = "ast_output.json") -> None:
    """Guarda la estructura de datos AST serializable en un archivo JSON."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(ast_serializable_data, f, indent=2, ensure_ascii=False, sort_keys=False)
        print(f"AST successfully saved to {filename}")
    except TypeError as e:
        print(f"Error saving AST to JSON file '{filename}': {str(e)}")
        print("The AST data structure might contain non-serializable objects.")
    except IOError as e:
        print(f"I/O error saving AST to JSON file '{filename}': {str(e)}")