# Types.py

from typing import Optional

# Set of valid GoX type names (used by semantic analyzer for validation)
gox_typenames = {'bool', 'char', 'float', 'int', 'string', 'void'} # 'void' for function return types

# Binary operations: (left_operand_type, operator_symbol, right_operand_type) -> result_type
# These define the type checking rules for binary operations.
# The semantic analyzer will use this.
bin_ops_type_rules = {
    # Integer arithmetic
    ('int', '+', 'int'): 'int',
    ('int', '-', 'int'): 'int',
    ('int', '*', 'int'): 'int',
    ('int', '/', 'int'): 'int', # Integer division
    ('int', '%', 'int'): 'int',

    # Integer comparisons -> bool
    ('int', '<', 'int'): 'bool',
    ('int', '<=', 'int'): 'bool',
    ('int', '>', 'int'): 'bool',
    ('int', '>=', 'int'): 'bool',
    ('int', '==', 'int'): 'bool',
    ('int', '!=', 'int'): 'bool',

    # Float arithmetic
    ('float', '+', 'float'): 'float',
    ('float', '-', 'float'): 'float',
    ('float', '*', 'float'): 'float',
    ('float', '/', 'float'): 'float', # Float division

    # Float comparisons -> bool
    ('float', '<', 'float'): 'bool',
    ('float', '<=', 'float'): 'bool',
    ('float', '>', 'float'): 'bool',
    ('float', '>=', 'float'): 'bool',
    ('float', '==', 'float'): 'bool',
    ('float', '!=', 'float'): 'bool',

    # Boolean logical operations -> bool
    ('bool', '&&', 'bool'): 'bool',
    ('bool', '||', 'bool'): 'bool',
    ('bool', '==', 'bool'): 'bool', # Equality for booleans
    ('bool', '!=', 'bool'): 'bool', # Inequality for booleans

    # Character comparisons -> bool
    ('char', '<', 'char'): 'bool',
    ('char', '<=', 'char'): 'bool',
    ('char', '>', 'char'): 'bool',
    ('char', '>=', 'char'): 'bool',
    ('char', '==', 'char'): 'bool',
    ('char', '!=', 'char'): 'bool',

    # Mixed type operations (explicitly define allowed ones if any, e.g., for promotion)
    # As per your rules: "operaciones de diferente tipado es considerado un error"
    # So, no mixed arithmetic like ('int', '+', 'float') is defined here.
    # The semantic analyzer will need to handle promotions or report errors.
}

def check_binop_type(op_symbol: str, left_gox_type: Optional[str], right_gox_type: Optional[str]) -> Optional[str]:
    """
    Checks the validity and result type of a binary operation.
    Returns the result type string if valid, None otherwise.
    """
    if left_gox_type is None or right_gox_type is None:
        return None # Operands don't have a determined type
    return bin_ops_type_rules.get((left_gox_type, op_symbol, right_gox_type))

# Unary operations: (operator_symbol, operand_type) -> result_type
unary_ops_type_rules = {
    # Integer
    ('+', 'int'): 'int',
    ('-', 'int'): 'int',
    # ('^', 'int'): 'int', # '^' is now MEM_ALLOC, a special AST node, not a generic UnaryOp for type checking here.
                         # Its type is 'int' (address), but handled by MemoryAllocationNode.
    
    # Float
    ('+', 'float'): 'float',
    ('-', 'float'): 'float',

    # Boolean
    ('!', 'bool'): 'bool', # Logical NOT
}

def check_unaryop_type(op_symbol: str, operand_gox_type: Optional[str]) -> Optional[str]:
    """
    Checks the validity and result type of a unary operation.
    Returns the result type string if valid, None otherwise.
    """
    if operand_gox_type is None:
        return None # Operand doesn't have a determined type
    return unary_ops_type_rules.get((op_symbol, operand_gox_type))