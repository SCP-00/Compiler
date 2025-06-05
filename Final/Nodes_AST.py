# Nodes_AST.py
from typing import Any, List, Optional, Dict

# Helper function for to_dict, can be here or imported if in another utils file
def _serialize_node_attribute(attr_value: Any) -> Any:
    if isinstance(attr_value, Node): # Check if it's an instance of the base Node class
        return attr_value.to_dict()
    elif isinstance(attr_value, list):
        return [_serialize_node_attribute(item) for item in attr_value]
    # Add other specific types if needed, e.g., enums, Token objects if stored
    # For now, assume other types are directly serializable (int, str, bool, float, None)
    return attr_value

class Node:
    """Base class for all AST nodes."""
    def __init__(self, lineno: Optional[int] = None):
        self.lineno: Optional[int] = lineno
        # This 'type' attribute will be populated by the Semantic Analyzer
        # with the resolved GoX type of the expression/node.
        self.gox_type: Optional[str] = None 
        # Optional: For storing additional semantic info like symbol table entries
        self.semantic_info: Dict[str, Any] = {} 

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the node to a dictionary.
        Uses introspection to get attributes, excluding private/protected ones
        and ones already handled (like lineno, _type).
        """
        node_dict: Dict[str, Any] = {
            "_type": self.__class__.__name__, # Use "_type" to avoid conflict with a potential "type" attribute
            "lineno": self.lineno,
            "gox_type": self.gox_type # Also serialize the resolved GoX type
        }
        # Iterate over attributes of the specific node instance
        for key, value in self.__dict__.items():
            # Avoid re-serializing attributes already handled or private/dunder attributes.
            if key not in ["lineno", "gox_type", "semantic_info", "_type"] and not key.startswith('_'):
                node_dict[key] = _serialize_node_attribute(value)
        
        # Optionally include semantic_info if it's populated and not empty
        if self.semantic_info:
            node_dict["semantic_info"] = _serialize_node_attribute(self.semantic_info)
            
        return node_dict

    def accept(self, visitor: Any, *args: Any, **kwargs: Any) -> Any: # More generic visitor signature
        """
        Accepts a visitor.
        The visitor pattern typically involves calling visit_NodeName on the visitor.
        """
        method_name = f'visit_{self.__class__.__name__}'
        visitor_method = getattr(visitor, method_name, visitor.generic_visit)
        return visitor_method(self, *args, **kwargs)

# ---------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------

class Integer(Node):
    """Represents an integer literal."""
    def __init__(self, value: int, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.value: int = value
        self.gox_type = 'int' # Literals have their type known immediately

class Float(Node):
    """Represents a float literal."""
    def __init__(self, value: float, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.value: float = value
        self.gox_type = 'float'

class String(Node):
    """Represents a string literal."""
    def __init__(self, value: str, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.value: str = value
        self.gox_type = 'string'

class Char(Node):
    """Represents a character literal."""
    def __init__(self, value: str, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.value: str = value # Should be a single character
        self.gox_type = 'char'

class Boolean(Node):
    """Represents a boolean literal."""
    def __init__(self, value: bool, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.value: bool = value
        self.gox_type = 'bool'

class Location(Node):
    """Represents a variable location (identifier)."""
    def __init__(self, name: str, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.name: str = name
        # gox_type will be filled by semantic analyzer by looking up 'name'

class BinOp(Node):
    """Represents a binary operation (e.g., a + b)."""
    def __init__(self, op: str, left: Node, right: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.op: str = op
        self.left: Node = left
        self.right: Node = right
        # gox_type (result type) will be filled by semantic analyzer

class UnaryOp(Node):
    """Represents a unary operation (e.g., -5, !true, ^size)."""
    def __init__(self, op: str, operand: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.op: str = op
        self.operand: Node = operand
        # gox_type (result type) will be filled by semantic analyzer
        # Note: ^ for memory allocation is now a separate node: MemoryAllocation

class MemoryAllocation(Node): # New Node for ^ operator
    """Represents memory allocation using '^' (e.g., ^1000, ^(n+1))."""
    def __init__(self, size_expr: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.size_expr: Node = size_expr
        self.gox_type = 'int' # The result of memory allocation is a base address (pointer), treated as int

class MemoryAddress(Node): # For `expr (dereferencing an address expression)
    """Represents accessing memory at an address given by an expression (e.g., `addr, `(base+i))."""
    def __init__(self, address_expr: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.address_expr: Node = address_expr
        # gox_type of the value at this address depends on context (read/write type)
        # Semantic analyzer might set this if it can infer, or it's context-dependent.
        # For now, let's assume it can be int/float/char when read.
        # When used as L-value, type checking occurs during assignment.

class FunctionCall(Node):
    """Represents a function call."""
    def __init__(self, name: str, args: List[Node], lineno: Optional[int] = None):
        super().__init__(lineno)
        self.name: str = name
        self.args: List[Node] = args
        # gox_type (return type) will be filled by semantic analyzer

class TypeCast(Node):
    """Represents a type cast (e.g., int(3.14))."""
    def __init__(self, target_gox_type: str, expr: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.target_gox_type: str = target_gox_type # e.g., "int", "float"
        self.expr: Node = expr
        self.gox_type = target_gox_type # The type of the cast expression is the target type

# CompareOp and LogicalOp already derive 'type' from their nature (bool)
class CompareOp(Node):
    """Represents a comparison operation (e.g., x > y)."""
    def __init__(self, op: str, left: Node, right: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.op: str = op
        self.left: Node = left
        self.right: Node = right
        self.gox_type = 'bool'

class LogicalOp(Node):
    """Represents a logical operation (e.g., x && y)."""
    def __init__(self, op: str, left: Node, right: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.op: str = op
        self.left: Node = left
        self.right: Node = right
        self.gox_type = 'bool'

# ArrayLiteral and IndexAccess are not used in your provided GoX examples yet,
# but keeping them for potential future use.
class ArrayLiteral(Node):
    """Represents an array literal."""
    def __init__(self, elements: List[Node], lineno: Optional[int] = None):
        super().__init__(lineno)
        self.elements: List[Node] = elements

class IndexAccess(Node):
    """Represents an index access (e.g., arr[i])."""
    def __init__(self, array_expr: Node, index_expr: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.array_expr: Node = array_expr
        self.index_expr: Node = index_expr

# ---------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------

class VariableDecl(Node):
    """Represents a variable declaration."""
    def __init__(self, name: str, type_spec: Optional[str], value: Optional[Node], lineno: Optional[int] = None):
        super().__init__(lineno)
        self.name: str = name
        self.type_spec: Optional[str] = type_spec # Explicitly declared type
        self.value: Optional[Node] = value
        # gox_type will be set to type_spec or inferred type by semantic analyzer

class ConstantDecl(Node):
    """Represents a constant declaration."""
    def __init__(self, name: str, value: Node, type_spec: Optional[str] = None, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.name: str = name
        self.value: Node = value
        self.type_spec: Optional[str] = type_spec # To store inferred type
        # gox_type will be set to inferred type by semantic analyzer

class Parameter(Node): # Parameter is part of a FunctionDecl, not a standalone declaration
    """Represents a function parameter."""
    def __init__(self, name: str, param_gox_type: str, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.name: str = name
        self.param_gox_type: str = param_gox_type
        self.gox_type = param_gox_type # The type of the parameter itself

class FunctionDecl(Node):
    """Represents a function declaration."""
    def __init__(self, name: str, params: List[Parameter], return_gox_type: str, body: List[Node], lineno: Optional[int] = None):
        super().__init__(lineno)
        self.name: str = name
        self.params: List[Parameter] = params
        self.return_gox_type: str = return_gox_type
        self.body: List[Node] = body # This is effectively a Block of statements
        self.gox_type = 'function' # Special type for function symbols

class ImportDecl(Node):
    """Represents an import declaration (e.g., import myModule;). Not used by current examples."""
    def __init__(self, module_name: str, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.module_name: str = module_name
        self.gox_type = 'module' # Special type for module symbols

class FunctionImportDecl(Node):
    """Represents a function import declaration with signature."""
    def __init__(self, func_name: str, params: List[Parameter], return_gox_type: str, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.func_name: str = func_name
        self.params: List[Parameter] = params
        self.return_gox_type: str = return_gox_type
        self.gox_type = 'function' # Imported functions are also functions

# ---------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------

class Assignment(Node):
    """Represents an assignment (e.g., x = 10 or `addr = val)."""
    # location can be Location (variable) or MemoryAddress
    def __init__(self, target: Node, expr: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.target: Node = target 
        self.expr: Node = expr

class Print(Node):
    """Represents a print statement (e.g., print x)."""
    def __init__(self, expr: Node, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.expr: Node = expr

class If(Node):
    """Represents an if/else statement."""
    def __init__(self, test: Node, consequence: List[Node], alternative: Optional[List[Node]] = None, lineno: Optional[int] = None):
        super().__init__(lineno)
        self.test: Node = test # Condition
        self.consequence: List[Node] = consequence # Statements for true branch (a block)
        self.alternative: Optional[List[Node]] = alternative # Statements for false branch (a block)

class While(Node):
    """Represents a while loop."""
    def __init__(self, test: Node, body: List[Node], lineno: Optional[int] = None):
        super().__init__(lineno)
        self.test: Node = test # Condition
        self.body: List[Node] = body # Statements for loop body (a block)

class Return(Node):
    """Represents a return statement (e.g., return x)."""
    def __init__(self, expr: Optional[Node], lineno: Optional[int] = None): # expr can be None for void returns
        super().__init__(lineno)
        self.expr: Optional[Node] = expr

class Break(Node):
    """Represents a break statement."""
    def __init__(self, lineno: Optional[int] = None):
        super().__init__(lineno)

class Continue(Node):
    """Represents a continue statement."""
    def __init__(self, lineno: Optional[int] = None):
        super().__init__(lineno)

# Block node is implicitly handled by List[Node] in FunctionDecl, If, While.
# If you want an explicit Block node:
# class Block(Node):
#     """Represents a block of statements."""
#     def __init__(self, statements: List[Node], lineno: Optional[int] = None):
#         super().__init__(lineno)
#         self.statements: List[Node] = statements

# ---------------------------------------------------------------------
# Program Root
# ---------------------------------------------------------------------

class Program(Node):
    """Root node representing a complete program."""
    def __init__(self, body: List[Node], lineno: Optional[int] = None): # body is list of top-level declarations/statements
        super().__init__(lineno)
        self.body: List[Node] = body

# Note: SymbolTable and Visitor classes that were previously in Nodes_AST.py
# should ideally be in their own files (e.g., SymbolTable.py, Visitor.py)
# or part of the SemanticAnalyzer/Interpreter if tightly coupled.
# For now, I'm keeping them out of Nodes_AST.py to keep it focused on node definitions.