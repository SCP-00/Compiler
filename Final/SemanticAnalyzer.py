# SemanticAnalyzer.py

import sys
from typing import Optional
from Nodes_AST import * # Import all AST node types
from Error import ErrorHandler
from SymbolTable import SymbolTable, SymbolEntry # Use the revised SymbolTable
from Types import gox_typenames, check_binop_type, check_unaryop_type # Use the revised Types

class SemanticAnalyzer:
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler: ErrorHandler = error_handler
        # The global symbol table. All other scopes will be children of this.
        self.global_scope: SymbolTable = SymbolTable("global", scope_level=0)
        self.current_scope: SymbolTable = self.global_scope
        
        # For checking return statements and break/continue
        self.current_function_return_type: Optional[str] = None
        self.loop_depth: int = 0 

    def enter_scope(self, scope_name_prefix: str = "block") -> None:
        """Enters a new scope, making it a child of the current scope."""
        # Generate a unique name for block scopes if needed, e.g., using a counter or line number
        scope_name = f"{scope_name_prefix}_{self.current_scope.scope_level + 1}"
        new_scope = SymbolTable(scope_name, parent_scope=self.current_scope, scope_level=self.current_scope.scope_level + 1)
        self.current_scope = new_scope

    def exit_scope(self) -> None:
        """Exits the current scope, moving to its parent."""
        if self.current_scope.parent_scope:
            self.current_scope = self.current_scope.parent_scope
        else:
            # This should not happen if scopes are managed correctly
            self.error_handler.add_semantic_error("Attempted to exit global scope.", None)

    def analyze(self, node: Optional[Node]) -> Optional[str]: # Returns the GoX type of the node/expression
        """Dispatches to the appropriate analyze_NodeName method."""
        if node is None:
            return None
        
        # Generic accept method in Node calls visit_NodeName here
        return node.accept(self) # Pass self as the visitor

    # --- Visitor Methods (analyze_NodeName) ---

    def visit_Program(self, node: Program) -> None: # Program doesn't have a "type" itself
        for item in node.body:
            self.analyze(item)
        return None # Or some indicator of success/failure

    def visit_Integer(self, node: Integer) -> str:
        node.gox_type = 'int'
        return 'int'

    def visit_Float(self, node: Float) -> str:
        node.gox_type = 'float'
        return 'float'

    def visit_String(self, node: String) -> str:
        node.gox_type = 'string'
        return 'string'

    def visit_Char(self, node: Char) -> str:
        node.gox_type = 'char'
        return 'char'

    def visit_Boolean(self, node: Boolean) -> str:
        node.gox_type = 'bool'
        return 'bool'

    def visit_Location(self, node: Location) -> Optional[str]:
        symbol_entry = self.current_scope.lookup_symbol(node.name)
        if not symbol_entry:
            self.error_handler.add_semantic_error(f"Variable '{node.name}' not defined.", node.lineno)
            return None
        
        # Store symbol info in the node for later stages (e.g., code gen)
        node.semantic_info['symbol_entry'] = symbol_entry 
        node.gox_type = symbol_entry.gox_type
        return symbol_entry.gox_type

    def visit_VariableDecl(self, node: VariableDecl) -> None:
        initial_value_gox_type: Optional[str] = None
        if node.value:
            initial_value_gox_type = self.analyze(node.value) # Analyze initializer type
            if not initial_value_gox_type:
                # Error in initializer, already reported by analyze(node.value)
                return # Cannot proceed with declaration if initializer is bad

        declared_gox_type = node.type_spec

        if declared_gox_type and initial_value_gox_type:
            # Type explicitly declared AND value provided: check for mismatch
            if declared_gox_type != initial_value_gox_type:
                # Allow int to float promotion for initialization
                if declared_gox_type == 'float' and initial_value_gox_type == 'int':
                    # This is an implicit promotion, semantic analyzer can note this
                    # For actual interpretation/codegen, a cast might be inserted
                    pass # Or create/attach a TypeCast node here
                else:
                    self.error_handler.add_semantic_error(
                        f"Type mismatch: Cannot initialize variable '{node.name}' of type '{declared_gox_type}' with value of type '{initial_value_gox_type}'.",
                        node.lineno
                    )
                    return
        elif not declared_gox_type and initial_value_gox_type:
            # Type inferred from initial value
            declared_gox_type = initial_value_gox_type
            node.type_spec = initial_value_gox_type # Update AST node for consistency
        elif declared_gox_type and not initial_value_gox_type:
            # Type declared, no initial value (valid)
            pass
        elif not declared_gox_type and not initial_value_gox_type:
            # Rule: var id; is invalid. Must have type or value.
            self.error_handler.add_semantic_error(
                f"Variable '{node.name}' must have an explicit type or be initialized with a value.",
                node.lineno
            )
            return

        node.gox_type = declared_gox_type # Final type of the variable
        
        try:
            symbol = SymbolEntry(name=node.name, gox_type=declared_gox_type, declaration_node=node)
            self.current_scope.add_symbol(symbol)
        except SymbolTable.SymbolAlreadyDefinedError:
            self.error_handler.add_semantic_error(
                f"Variable '{node.name}' already defined in this scope.", node.lineno
            )
        return None # Declarations don't have a "value type" themselves

    def visit_ConstantDecl(self, node: ConstantDecl) -> None:
        if not node.value: # Should be caught by parser, but double check
            self.error_handler.add_semantic_error(f"Constant '{node.name}' must be initialized.", node.lineno)
            return

        value_gox_type = self.analyze(node.value)
        if not value_gox_type:
            # Error in constant's value expression
            return

        node.gox_type = value_gox_type
        node.type_spec = value_gox_type # Store inferred type in type_spec for consistency
        try:
            # For constants, we could store the actual evaluated value if it's compile-time known
            # For now, just the type and declaration node.
            symbol = SymbolEntry(name=node.name, gox_type=value_gox_type, declaration_node=node, is_constant=True)
            self.current_scope.add_symbol(symbol)
        except SymbolTable.SymbolAlreadyDefinedError:
            self.error_handler.add_semantic_error(
                f"Constant '{node.name}' already defined in this scope.", node.lineno
            )
        return None

    def visit_Assignment(self, node: Assignment) -> None:
        target_gox_type = self.analyze(node.target) # Target can be Location or MemoryAddress
        expr_gox_type = self.analyze(node.expr)

        if not target_gox_type or not expr_gox_type:
            # Errors in target or expression already reported
            return

        # Check if target is assignable (not a constant)
        if isinstance(node.target, Location):
            symbol_entry = self.current_scope.lookup_symbol(node.target.name)
            if symbol_entry and symbol_entry.is_constant:
                self.error_handler.add_semantic_error(
                    f"Cannot assign to constant '{node.target.name}'.", node.lineno
                )
                return
        
        # Type checking for assignment
        if target_gox_type == expr_gox_type:
            pass # Types match
        elif target_gox_type == 'float' and expr_gox_type == 'int':
            # Allow int to float promotion in assignment
            # Optionally, an implicit cast node could be inserted into the AST here
            # node.expr = TypeCast('float', node.expr, node.expr.lineno)
            # self.analyze(node.expr) # Re-analyze the new cast node if inserted
            pass
        # Assignment to memory (`addr = char_val`)
        # If `target_gox_type` for MemoryAddress is contextually 'char' (or 'int' for pointer)
        # and `expr_gox_type` is 'char', this might be okay.
        # Current MemoryAddress.gox_type is 'int' (the address itself).
        # Need a way to determine the type *expected at* the memory location.
        # For `addr = '\xff'`, `target_gox_type` (of `addr) is 'int'. `expr_gox_type` is 'char'.
        # This specific case is allowed by GoX for memory writes.
        elif isinstance(node.target, MemoryAddress) and target_gox_type == 'int' and expr_gox_type == 'char':
            pass # Allowed: `addr = 'char_val'`
        elif isinstance(node.target, MemoryAddress) and target_gox_type == 'int' and expr_gox_type == 'int':
            pass # Allowed: `addr = int_val`
        elif isinstance(node.target, MemoryAddress) and target_gox_type == 'int' and expr_gox_type == 'float':
             # Storing float into generic memory addr. This is often allowed if memory is untyped.
             # In GoX, this is mentioned in memory.gox `addr = 12.34;`
            pass
        else:
            self.error_handler.add_semantic_error(
                f"Type mismatch: Cannot assign value of type '{expr_gox_type}' to target of type '{target_gox_type}'.",
                node.lineno
            )
        return None

    def visit_Print(self, node: Print) -> None:
        # Print can usually handle various types. Type check the expression itself.
        expr_type = self.analyze(node.expr)
        if expr_type is None:
            self.error_handler.add_semantic_error("Invalid expression in print statement.", node.lineno)
        # No specific type restriction on what can be printed, unless GoX defines it.
        return None

    def visit_BinOp(self, node: BinOp) -> Optional[str]:
        left_gox_type = self.analyze(node.left)
        right_gox_type = self.analyze(node.right)

        if not left_gox_type or not right_gox_type:
            return None # Error in operands

        # Your rule: "operaciones de diferente tipado es considerado un error"
        # This means strict type matching for operands, no automatic int-to-float.
        if left_gox_type != right_gox_type:
            # Exception: Comparisons might allow int vs float if one is promoted,
            # but your rule is strict.
            self.error_handler.add_semantic_error(
                f"Type mismatch in binary operation '{node.op}': Cannot operate on '{left_gox_type}' and '{right_gox_type}'.",
                node.lineno
            )
            return None
        
        # Now that types are the same, use Types.py for validation
        result_gox_type = check_binop_type(node.op, left_gox_type, right_gox_type)
        if result_gox_type is None:
            self.error_handler.add_semantic_error(
                f"Invalid binary operation: '{left_gox_type} {node.op} {right_gox_type}'.",
                node.lineno
            )
            return None
        
        node.gox_type = result_gox_type
        return result_gox_type

    def visit_UnaryOp(self, node: UnaryOp) -> Optional[str]:
        operand_gox_type = self.analyze(node.operand)
        if not operand_gox_type:
            return None # Error in operand

        # MEM_ALLOC ('^') is handled by MemoryAllocationNode, not here as UnaryOp.
        if node.op == '^': # This should not happen if parser creates MemoryAllocationNode for ^
             self.error_handler.add_semantic_error(
                f"Internal Error: '^' operator should be a MemoryAllocationNode, not UnaryOp.", node.lineno
            )
             return None


        result_gox_type = check_unaryop_type(node.op, operand_gox_type)
        if result_gox_type is None:
            self.error_handler.add_semantic_error(
                f"Invalid unary operation: '{node.op}{operand_gox_type}'.",
                node.lineno
            )
            return None
        
        node.gox_type = result_gox_type
        return result_gox_type

    def visit_MemoryAllocation(self, node: MemoryAllocation) -> str: # Always returns 'int' (address)
        size_expr_gox_type = self.analyze(node.size_expr)
        if size_expr_gox_type != 'int':
            self.error_handler.add_semantic_error(
                f"Memory allocation size expression must be an integer, got '{size_expr_gox_type}'.",
                node.lineno
            )
            # Fallback or error, but the node itself represents an address
        node.gox_type = 'int' 
        return 'int'

    def visit_MemoryAddress(self, node: MemoryAddress) -> str: # `addr returns the value at addr. Type is context-dependent
        address_expr_gox_type = self.analyze(node.address_expr)
        if address_expr_gox_type != 'int':
            self.error_handler.add_semantic_error(
                f"Expression used as memory address must be an integer, got '{address_expr_gox_type}'.",
                node.lineno
            )
        # The type of `addr is the type of the data stored at that address.
        # This is context-dependent. For now, we'll mark the address itself as 'int'.
        # The type checking for the *value* read from memory happens where it's used.
        # If `addr is on LHS of assignment, Assignment node handles it.
        # If `addr is on RHS, its type needs to be known or inferred.
        # For simplicity now, let's assume `addr reads an int by default or its type is unknown until context.
        # A more advanced system might require type hints for memory or track types stored.
        node.gox_type = 'int' # Defaulting to int as per `print `addr + 0;`
                              # Could also be 'any' or require specific cast like char(`addr)
        return node.gox_type # Or a generic pointer type if you had one.

    def visit_TypeCast(self, node: TypeCast) -> Optional[str]:
        expr_gox_type = self.analyze(node.expr)
        if not expr_gox_type: return None

        target_gox_type = node.target_gox_type
        if target_gox_type not in gox_typenames:
            self.error_handler.add_semantic_error(f"Invalid target type for cast: '{target_gox_type}'.", node.lineno)
            return None

        # Basic valid casts (extend as needed)
        valid_casts = {
            ('int', 'float'), ('float', 'int'),
            ('int', 'char'), ('char', 'int'), # char to int (ASCII) and vice-versa
            # Add other valid casts, e.g., int to bool, float to bool?
        }
        if expr_gox_type == target_gox_type: # Casting to same type is allowed
            node.gox_type = target_gox_type
            return target_gox_type
        
        if (expr_gox_type, target_gox_type) in valid_casts:
            node.gox_type = target_gox_type
            return target_gox_type
        else:
            self.error_handler.add_semantic_error(
                f"Invalid type cast from '{expr_gox_type}' to '{target_gox_type}'.",
                node.lineno
            )
            return None
            
    def visit_If(self, node: If) -> None:
        condition_gox_type = self.analyze(node.test)
        # Solo reportar error de tipo de condición si la condición en sí no tuvo errores y no es bool
        if condition_gox_type is not None and condition_gox_type != 'bool':
            self.error_handler.add_semantic_error(
                f"If statement condition must be boolean, got '{condition_gox_type}'.",
                node.test.lineno if node.test and hasattr(node.test, 'lineno') else node.lineno
            )
        
        self.enter_scope("if_consequence")
        for stmt in node.consequence:
            self.analyze(stmt)
        self.exit_scope()

        if node.alternative:
            self.enter_scope("if_alternative")
            for stmt in node.alternative:
                self.analyze(stmt)
            self.exit_scope()
        return None

    def visit_While(self, node: While) -> None:
        condition_gox_type = self.analyze(node.test)
        # Solo reportar error de tipo de condición si la condición en sí no tuvo errores y no es bool
        if condition_gox_type is not None and condition_gox_type != 'bool':
            self.error_handler.add_semantic_error(
                f"While loop condition must be boolean, got '{condition_gox_type}'.",
                node.test.lineno if node.test and hasattr(node.test, 'lineno') else node.lineno
            )
        
        self.loop_depth += 1
        self.enter_scope("while_body")
        for stmt in node.body:
            self.analyze(stmt)
        self.exit_scope()
        self.loop_depth -= 1
        return None

    def visit_Break(self, node: Break) -> None:
        if self.loop_depth == 0:
            self.error_handler.add_semantic_error(
                "'break' statement not within a loop.", node.lineno
            )
        return None

    def visit_Continue(self, node: Continue) -> None:
        if self.loop_depth == 0:
            self.error_handler.add_semantic_error(
                "'continue' statement not within a loop.", node.lineno
            )
        return None

    def visit_Return(self, node: Return) -> None:
        if self.current_function_return_type is None:
            self.error_handler.add_semantic_error(
                "'return' statement outside of a function.", node.lineno
            )
            return None

        returned_expr_gox_type: Optional[str] = 'void' # Default for `return;`
        if node.expr:
            returned_expr_gox_type = self.analyze(node.expr)
            if not returned_expr_gox_type: return None # Error in return expression
        
        expected_return_type = self.current_function_return_type
        
        if expected_return_type == 'void' and returned_expr_gox_type != 'void':
            self.error_handler.add_semantic_error(
                f"Function declared as void cannot return a value of type '{returned_expr_gox_type}'.", node.lineno
            )
        elif expected_return_type != 'void' and returned_expr_gox_type == 'void':
             self.error_handler.add_semantic_error(
                f"Function declared to return '{expected_return_type}' must return a value.", node.lineno
            )
        elif expected_return_type != returned_expr_gox_type:
            # Allow int to float promotion for return type
            if expected_return_type == 'float' and returned_expr_gox_type == 'int':
                pass # Implicit promotion
            else:
                self.error_handler.add_semantic_error(
                    f"Type mismatch in return statement: Expected '{expected_return_type}', got '{returned_expr_gox_type}'.",
                    node.lineno
                )
        return None # Return statements don't have a "value type" themselves in this context

    def visit_FunctionDecl(self, node: FunctionDecl) -> None:
        # Add function to parent scope (or current if global) *before* entering new scope for its body
        # This allows for recursion.
        try:
            # Store function signature info in SymbolEntry
            param_types = [p.param_gox_type for p in node.params]
            func_symbol_info = {
                'name': node.name, 
                'gox_type': 'function', # Special type for functions
                'return_gox_type': node.return_gox_type,
                'param_gox_types': param_types,
                'declaration_node': node
            }
            func_entry = SymbolEntry(
                name=node.name,
                gox_type='function', # Or construct a more complex function type string
                declaration_node=node,
                # value=func_symbol_info # Storing the dict as value
            )
            # Add to the scope where the function is defined (e.g., global scope)
            self.current_scope.add_symbol(func_entry)
            node.semantic_info['symbol_entry'] = func_entry
        except SymbolTable.SymbolAlreadyDefinedError:
            self.error_handler.add_semantic_error(
                f"Function '{node.name}' already defined.", node.lineno
            )
            # Don't proceed with analyzing this func if redefinition is an error
            return 

        # --- Analyze function body in its own scope ---
        # Save current function's expected return type
        previous_function_return_type = self.current_function_return_type
        self.current_function_return_type = node.return_gox_type
        
        self.enter_scope(f"function_{node.name}")
        
        # Add parameters to the function's new scope
        for param_node in node.params:
            try:
                param_entry = SymbolEntry(
                    name=param_node.name, 
                    gox_type=param_node.param_gox_type, 
                    declaration_node=param_node
                )
                self.current_scope.add_symbol(param_entry)
                param_node.semantic_info['symbol_entry'] = param_entry
                param_node.gox_type = param_node.param_gox_type
            except SymbolTable.SymbolAlreadyDefinedError:
                 self.error_handler.add_semantic_error(
                    f"Parameter '{param_node.name}' redefined in function '{node.name}'.", param_node.lineno
                )

        # Analyze statements in the function body
        has_return_statement = False
        for stmt_node in node.body:
            self.analyze(stmt_node)
            if isinstance(stmt_node, Return):
                has_return_statement = True
        
        # Check for missing return in non-void functions (simple check, control flow not analyzed here)
        if node.return_gox_type != 'void' and not has_return_statement:
            # This is a simplified check. A proper check requires control flow analysis
            # to ensure all paths return a value.
            # For now, just warn if there's no return statement at all at the end of the body list.
            if not node.body or not isinstance(node.body[-1], Return):
                 self.error_handler.add_semantic_error(
                    f"Function '{node.name}' declared to return '{node.return_gox_type}' might not return a value on all paths (missing return?).", 
                    node.lineno # Or line of closing brace
                )

        self.exit_scope()
        # Restore previous function context (for nested function analysis, though GoX doesn't have them)
        self.current_function_return_type = previous_function_return_type
        return None

    def visit_FunctionImportDecl(self, node: FunctionImportDecl) -> None:
        try:
            param_types = [p.param_gox_type for p in node.params]
            func_symbol_info = {
                'name': node.func_name, 
                'gox_type': 'function', # Special type
                'return_gox_type': node.return_gox_type,
                'param_gox_types': param_types,
                'is_imported': True,
                'declaration_node': node
            }
            func_entry = SymbolEntry(
                name=node.func_name,
                gox_type='function',
                declaration_node=node,
                #value=func_symbol_info
            )
            # Imported functions are typically added to the global scope
            self.global_scope.add_symbol(func_entry)
            node.semantic_info['symbol_entry'] = func_entry
        except SymbolTable.SymbolAlreadyDefinedError:
            self.error_handler.add_semantic_error(
                f"Imported function '{node.func_name}' conflicts with an existing definition.", node.lineno
            )
        return None
        
    def visit_FunctionCall(self, node: FunctionCall) -> Optional[str]:
        func_symbol_entry = self.current_scope.lookup_symbol(node.name)

        if not func_symbol_entry:
            self.error_handler.add_semantic_error(f"Function '{node.name}' not defined.", node.lineno)
            return None
        
        if func_symbol_entry.gox_type != 'function':
            self.error_handler.add_semantic_error(f"'{node.name}' is not a function.", node.lineno)
            return None

        # Get function signature from its declaration node stored in SymbolEntry
        func_decl_node = func_symbol_entry.declaration_node
        if not isinstance(func_decl_node, (FunctionDecl, FunctionImportDecl)):
            self.error_handler.add_semantic_error(f"Internal error: Symbol for '{node.name}' is not a function declaration.", node.lineno)
            return None

        expected_params = func_decl_node.params
        actual_args_count = len(node.args)
        expected_params_count = len(expected_params)

        if actual_args_count != expected_params_count:
            self.error_handler.add_semantic_error(
                f"Function '{node.name}' expects {expected_params_count} arguments, but got {actual_args_count}.",
                node.lineno
            )
            return None # Cannot determine return type if arg count is wrong

        # Check argument types
        for i, arg_node in enumerate(node.args):
            arg_gox_type = self.analyze(arg_node)
            if not arg_gox_type: continue # Error in argument expression

            expected_param_gox_type = expected_params[i].param_gox_type
            if arg_gox_type != expected_param_gox_type:
                # Allow int to float promotion for arguments
                if expected_param_gox_type == 'float' and arg_gox_type == 'int':
                    # Optionally, insert TypeCast node into AST for the argument
                    # node.args[i] = TypeCast('float', arg_node, arg_node.lineno)
                    # self.analyze(node.args[i]) # Re-analyze casted arg
                    pass 
                else:
                    self.error_handler.add_semantic_error(
                        f"Type mismatch for argument {i+1} of function '{node.name}': Expected '{expected_param_gox_type}', got '{arg_gox_type}'.",
                        arg_node.lineno
                    )
        
        node.gox_type = func_decl_node.return_gox_type
        node.semantic_info['function_symbol'] = func_symbol_entry
        return func_decl_node.return_gox_type

    def generic_visit(self, node: Node) -> None:
        # Fallback for nodes that don't have a specific visit_NodeName method
        # This should ideally not be called if all node types are handled.
        # print(f"Warning: No specific semantic analysis method for node type {type(node).__name__} at line {node.lineno if node else 'N/A'}")
        # Iterating through children generically can be risky if specific order or context is needed.
        # For simple structures, it might be:
        for attr_name in dir(node):
            if not attr_name.startswith('_'):
                attr_value = getattr(node, attr_name)
                if isinstance(attr_value, Node):
                    self.analyze(attr_value)
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, Node):
                            self.analyze(item)
        return None


# --- Main execution for semantic analysis (example) ---
def main():
    import os 
    from Lexer import tokenize
    from Parser import Parser 
    from AST_to_JSON import ast_to_json, save_ast_to_json

    if len(sys.argv) != 2:
        print("Usage: python SemanticAnalyzer.py <file.gox>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}"); sys.exit(1)
    if not file_path.endswith('.gox'):
        print(f"Error: File must have .gox extension: {file_path}"); sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as file: content = file.read()
    except Exception as e: print(f"Error reading file: {str(e)}"); sys.exit(1)

    print(f"\n--- Analyzing: {file_path} ---")
    error_handler = ErrorHandler()
    
    print("1. Lexical Analysis...")
    tokens = tokenize(content, error_handler)
    if error_handler.has_errors(): print("\nLEXICAL ERRORS:"); error_handler.report_errors(); sys.exit(1)
    print("   Lexical analysis successful.")

    print("2. Parsing...")
    parser = Parser(tokens, error_handler)
    ast_root = parser.parse()
    if error_handler.has_errors(): print("\nSYNTAX ERRORS:"); error_handler.report_errors(); sys.exit(1)
    if not ast_root: print("Parser returned no AST root. Exiting."); sys.exit(1)
    print("   Parsing successful. AST generated.")

    # Save initial AST (before semantic analysis)
    base_name = os.path.splitext(file_path)[0]
    ast_raw_file = f"{base_name}.ast_raw.json"
    save_ast_to_json(ast_to_json(ast_root), ast_raw_file) # ast_to_json needs to be robust
    print(f"Raw AST (pre-semantics) saved to: {ast_raw_file}")


    print("3. Semantic Analysis...")
    semantic_analyzer = SemanticAnalyzer(error_handler)
    semantic_analyzer.analyze(ast_root) # This modifies ast_root in-place


    if error_handler.has_errors():
        print("\nSEMANTIC ERRORS:")
        error_handler.report_errors()
        sys.exit(1) # Exit if semantic errors are found
    
    print("   Semantic analysis successful.")
    print("\n--- Symbol Table Hierarchy (Post-Analysis) ---")
    semantic_analyzer.global_scope.print_table()
    print(f"\n--- Analysis complete for {file_path} ---")

if __name__ == "__main__":
    # Asegúrate que SymbolTable se importe correctamente si está en Sym_tab.py
    from SymbolTable import SymbolTable, SymbolEntry
    from Types import gox_typenames, check_binop_type, check_unaryop_type
    main()
