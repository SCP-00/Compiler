# Parser.py
import sys
from typing import List, Optional, Union, Callable 
from Lexer import Token
from Nodes_AST import * 
from Error import ErrorHandler
from AST_to_JSON import save_ast_to_json, ast_to_json, pretty_print_json

class Parser:
    def __init__(self, tokens: List[Token], error_handler: ErrorHandler):
        self.tokens: List[Token] = tokens
        self.error_handler: ErrorHandler = error_handler
        self.pos: int = 0
        self.current_token: Optional[Token] = self.tokens[0] if tokens else None

    def advance(self) -> None:
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek(self, offset: int = 1) -> Optional[Token]:
        """Looks ahead one token by default."""
        peek_pos = self.pos + offset
        return self.tokens[peek_pos] if peek_pos < len(self.tokens) else None

    def match(self, *expected_types: str) -> Optional[Token]:
        """If current token matches one of expected_types, consume it and return it."""
        if self.current_token and self.current_token.type in expected_types:
            token = self.current_token
            self.advance()
            return token
        return None

    def consume(self, expected_type: str, error_message: str) -> Optional[Token]:
        """Consumes current token if it's expected_type, else reports error."""
        if self.current_token and self.current_token.type == expected_type:
            token = self.current_token
            self.advance()
            return token
        
        err_lineno = self.current_token.lineno if self.current_token else "End of file"
        current_type_str = f"'{self.current_token.type}'" if self.current_token else "end of file"
        full_error_message = f"{error_message} (Expected '{expected_type}', got {current_type_str})"
        self.error_handler.add_syntax_error(full_error_message, err_lineno)
        return None
    
    def consume_type(self, error_message_prefix: str) -> Optional[Token]:
        """Consumes a GoX type keyword token."""
        type_token = self.match('INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE')
        if not type_token:
            err_lineno = self.current_token.lineno if self.current_token else "End of file"
            self.error_handler.add_syntax_error(f"{error_message_prefix} (e.g., int, float, bool, string, char)", err_lineno)
            return None
        # Convert FLOAT_TYPE to "float", INT to "int" etc. for AST consistency
        type_token.value = type_token.value.lower() # Ensure lowercase if types are always lowercase
        if type_token.value == "float_type": type_token.value = "float"
        if type_token.value == "string_type": type_token.value = "string"
        if type_token.value == "char_type": type_token.value = "char"
        return type_token

    # --- Main Parsing Method ---
    def parse(self) -> Optional[Program]:
        top_level_nodes: List[Node] = []
        
        while self.current_token:
            node: Optional[Node] = None
            token_type = self.current_token.type

            if token_type == 'IMPORT':
                node = self.parse_import()
            elif token_type in ['VAR', 'CONST']:
                node = self.parse_declaration()
            elif token_type == 'FUNC':
                node = self.parse_function()
            elif token_type == 'ID' and self.peek() and self.peek().type == 'LPAREN':
                # This is a top-level function call statement
                call_expr = self.parse_primary() # Parses ID(...args...) as FunctionCall
                if call_expr:
                    if self.consume('SEMI', "Expected ';' after top-level function call statement"):
                        node = call_expr # FunctionCall node itself is the statement
                    else: # Error already reported by consume
                        node = None 
            else:
                self.error_handler.add_syntax_error(
                    f"Unexpected token at top level: {self.current_token.type} ('{self.current_token.value}')",
                    self.current_token.lineno
                )
                self.advance() # Skip the problematic token to try to continue

            if node:
                top_level_nodes.append(node)
            elif self.current_token and not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0) :
                # If no node was produced AND no new error was added by the parse_... methods,
                # it means an unhandled case or a bug. Advance to prevent infinite loop.
                # This check is a bit fragile due to error_handler state.
                # A better way is if parse_... methods *always* return a Node or raise/report error and return None.
                if self.current_token: # Ensure there's a token to advance past
                   # self.error_handler.add_syntax_error(f"Internal parser error or unhandled top-level token: {self.current_token.type}", self.current_token.lineno)
                   # self.advance() # Be cautious with auto-advancing on errors
                   pass # Let the outer loop's else handle general unexpected tokens
        
        return Program(top_level_nodes)

    # --- Statement Parsers ---
    def parse_statement(self) -> Optional[Node]:
        node: Optional[Node] = None
        if not self.current_token: return None

        token_type = self.current_token.type
        
        if token_type in ['VAR', 'CONST']:
            node = self.parse_declaration()
        elif token_type == 'PRINT':
            node = self.parse_print()
        elif token_type == 'IF':
            node = self.parse_if()
        elif token_type == 'WHILE':
            node = self.parse_while()
        elif token_type == 'RETURN':
            node = self.parse_return()
        elif token_type == 'BREAK':
            node = self.parse_control_statement('BREAK', Break)
        elif token_type == 'CONTINUE':
            node = self.parse_control_statement('CONTINUE', Continue)
        elif token_type == 'ID' or token_type == 'BACKTICK':
            # Could be an assignment or a function call statement
            # Try parsing as an expression first. If it's a FunctionCall and followed by ';', it's a statement.
            # If it's a Location or MemoryAddress and followed by '=', it's an assignment.
            
            # Simplified: assume assignment if ID is not followed by LPAREN, or if it's BACKTICK
            if token_type == 'ID' and self.peek() and self.peek().type == 'LPAREN':
                # It's a function call statement
                call_expr = self.parse_primary() # This will parse the FunctionCall
                if call_expr:
                    if self.consume('SEMI', "Expected ';' after function call statement"):
                        node = call_expr
            else: # Must be an assignment
                node = self.parse_assignment()
        else:
            self.error_handler.add_syntax_error(
                f"Unexpected token for statement: {token_type} ('{self.current_token.value}')",
                self.current_token.lineno
            )
            self.advance() # Skip the unexpected token

        return node

    def parse_statements_block(self) -> List[Node]:
        statements: List[Node] = []
        if not self.consume('LBRACE', "Expected '{' to start block"):
            return [] # Return empty list if block doesn't start correctly
        
        while self.current_token and self.current_token.type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            else:
                # Error in parsing statement or end of block.
                # If parse_statement returned None due to an error, it should have reported it.
                # We need to advance to avoid infinite loop if parse_statement didn't consume token.
                if self.current_token and self.current_token.type != 'RBRACE':
                    # self.error_handler.add_syntax_error(f"Skipping token '{self.current_token.value}' in block due to parsing error.", self.current_token.lineno)
                    self.advance() 
                else:
                    break # EOF or RBRACE
        
        if not self.consume('RBRACE', "Expected '}' to end block"):
            # Error reported by consume. Statements parsed so far are returned.
            pass
            
        return statements

    def parse_control_statement(self, keyword: str, node_class: type) -> Optional[Node]:
        start_token = self.match(keyword)
        if not start_token: return None 
            
        if not self.consume('SEMI', f"Missing ';' after {keyword.lower()} statement"):
            return None
            
        return node_class(lineno=start_token.lineno)

    # --- Expression Parsers ( siguiendo precedencia y asociatividad ) ---
    # Precedencia:
    # 1. Primary (literals, id, '(', MEM_ALLOC, BACKTICK, type casts, function calls)
    # 2. Unary (+, -, !, MEM_ALLOC)
    # 3. Multiplicative (*, /)
    # 4. Additive (+, -)
    # 5. Relational (<, <=, >, >=)
    # 6. Equality (==, !=)
    # 7. Logical AND (&&)
    # 8. Logical OR (||)
    # (Assignment = is not an expression operator here, it's a statement)

    def parse_expression(self) -> Optional[Node]:
        return self.parse_logical_or() # Start with lowest precedence

    def parse_logical_or(self) -> Optional[Node]:
        node = self.parse_logical_and()
        while node and self.current_token and self.current_token.type == 'LOR':
            op_token = self.current_token
            self.advance()
            right = self.parse_logical_and()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = LogicalOp(op_token.value, node, right, lineno=op_token.lineno)
        return node

    def parse_logical_and(self) -> Optional[Node]:
        node = self.parse_equality()
        while node and self.current_token and self.current_token.type == 'LAND':
            op_token = self.current_token
            self.advance()
            right = self.parse_equality()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = LogicalOp(op_token.value, node, right, lineno=op_token.lineno)
        return node

    def parse_equality(self) -> Optional[Node]:
        node = self.parse_relational()
        while node and self.current_token and self.current_token.type in ['EQ', 'NE']:
            op_token = self.current_token
            self.advance()
            right = self.parse_relational()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = CompareOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
        
    def parse_relational(self) -> Optional[Node]:
        node = self.parse_additive()
        while node and self.current_token and self.current_token.type in ['LT', 'GT', 'LE', 'GE']:
            op_token = self.current_token
            self.advance()
            right = self.parse_additive()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = CompareOp(op_token.value, node, right, lineno=op_token.lineno)
        return node

    def parse_additive(self) -> Optional[Node]:
        node = self.parse_multiplicative()
        while node and self.current_token and self.current_token.type in ['PLUS', 'MINUS']:
            op_token = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = BinOp(op_token.value, node, right, lineno=op_token.lineno)
        return node

    def parse_multiplicative(self) -> Optional[Node]:
        node = self.parse_unary()
        while node and self.current_token and self.current_token.type in ['TIMES', 'DIVIDE', 'MOD']: # Removed INT_DIV, assume / handles both
            op_token = self.current_token
            self.advance()
            right = self.parse_unary()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = BinOp(op_token.value, node, right, lineno=op_token.lineno)
        return node

    def parse_unary(self) -> Optional[Node]:
        if self.current_token and self.current_token.type in ['PLUS', 'MINUS', 'NOT']:
            op_token = self.current_token
            self.advance() 
            operand = self.parse_unary() # Unary ops can be chained (e.g. --x, !!b)
            if not operand:
                self.error_handler.add_syntax_error(f"Expected operand after unary operator '{op_token.value}'", op_token.lineno)
                return None
            return UnaryOp(op_token.value, operand, lineno=op_token.lineno)
        elif self.current_token and self.current_token.type == 'MEM_ALLOC': # Handle ^expr here
            op_token = self.current_token
            self.advance()
            size_expr = self.parse_unary() # ^ can apply to another unary or primary
            if not size_expr:
                self.error_handler.add_syntax_error(f"Expected size expression after memory allocation operator '^'", op_token.lineno)
                return None
            return MemoryAllocation(size_expr, lineno=op_token.lineno)
        
        return self.parse_primary()

    def parse_primary(self) -> Optional[Node]:
        token = self.current_token
        node: Optional[Node] = None

        if not token: return None

        if lit_token := self.match('INTEGER', 'FLOAT', 'STRING', 'CHAR', 'TRUE', 'FALSE'):
            node = self.create_literal_node(lit_token)
        elif self.match('LPAREN'):
            expr = self.parse_expression()
            if not expr: return None # Error already reported
            if not self.consume('RPAREN', "Expected ')' after parenthesized expression"):
                return None
            node = expr
        elif backtick_token := self.match('BACKTICK'):
            # This is for `expr where expr evaluates to an address. Used as R-value or L-value.
            # If `(base + i)`, the (base + i) is an expression.
            address_expression = self.parse_expression() 
            if not address_expression:
                self.error_handler.add_syntax_error("Expected address expression after '`'", backtick_token.lineno)
                return None
            node = MemoryAddress(address_expression, lineno=backtick_token.lineno)
        elif id_token := self.match('ID'):
            if self.current_token and self.current_token.type == 'LPAREN': # Function call as expression
                self.advance() # Consume LPAREN
                args: List[Node] = []
                if self.current_token and self.current_token.type != 'RPAREN':
                    while True:
                        arg = self.parse_expression()
                        if not arg: return None # Error in argument parsing
                        args.append(arg)
                        if not self.match('COMMA'): break
                if not self.consume('RPAREN', "Expected ')' after function arguments"):
                    return None
                node = FunctionCall(id_token.value, args, lineno=id_token.lineno)
            else: # Variable Location
                node = Location(id_token.value, lineno=id_token.lineno)
        elif self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE']:
            # Type Cast: type(expr)
            type_keyword_token = self.consume_type("Expected type keyword for cast") # consume_type also advances
            if not type_keyword_token: return None

            if not self.consume('LPAREN', f"Expected '(' after type '{type_keyword_token.value}' for cast"):
                return None
            expr_to_cast = self.parse_expression()
            if not expr_to_cast:
                self.error_handler.add_syntax_error(f"Expected expression inside cast to '{type_keyword_token.value}'", type_keyword_token.lineno)
                return None
            if not self.consume('RPAREN', f"Expected ')' after cast to '{type_keyword_token.value}'"):
                return None
            node = TypeCast(type_keyword_token.value, expr_to_cast, lineno=type_keyword_token.lineno)
        else:
            self.error_handler.add_syntax_error(f"Unexpected token in expression: {token.type} ('{token.value}')", token.lineno)
            self.advance() # Try to recover by skipping

        return node

    def create_literal_node(self, token: Token) -> Node:
        if token.type == 'INTEGER': return Integer(int(token.value), lineno=token.lineno)
        if token.type == 'FLOAT': return Float(float(token.value), lineno=token.lineno)
        if token.type == 'STRING': return String(token.value, lineno=token.lineno) # Value already processed by lexer
        if token.type == 'CHAR': return Char(token.value, lineno=token.lineno)     # Value already processed by lexer
        if token.type == 'TRUE': return Boolean(True, lineno=token.lineno)
        if token.type == 'FALSE': return Boolean(False, lineno=token.lineno)
        # This part should ideally not be reached if token types are pre-validated
        raise ValueError(f"Parser internal error: create_literal_node called with non-literal token type {token.type}")


    # --- Declaration and Block Parsers ---
    def parse_declaration(self) -> Optional[Node]:
        is_const = True if self.current_token and self.current_token.type == 'CONST' else False
        decl_keyword_token = self.match('VAR', 'CONST')
        if not decl_keyword_token: return None # Should be called when VAR or CONST is current

        name_token = self.consume('ID', "Expected identifier after 'var' or 'const'")
        if not name_token: return None

        type_spec: Optional[str] = None
        # `var id type = val` or `var id type`
        if not is_const and self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE']:
            type_spec_token = self.consume_type("Expected type specifier for variable")
            if not type_spec_token: return None # Error already reported
            type_spec = type_spec_token.value 

        value: Optional[Node] = None
        if self.match('ASSIGN'): # `var id = val` or `var id type = val` or `const id = val`
            value = self.parse_expression()
            if not value: # Error in parsing value expression
                 # If no error was reported by parse_expression, add one here
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error("Expected expression for initialization", self.current_token.lineno if self.current_token else name_token.lineno)
                return None
        
        # Validation: `const` must be initialized.
        if is_const and value is None:
            self.error_handler.add_syntax_error(f"Constant '{name_token.value}' must be initialized.", name_token.lineno)
            return None
        
        # Validation: `var id;` (no type, no value) is invalid as per your rules.
        # `var id type;` (type, no value) is valid.
        # `var id = value;` (no type, value) is valid (type to be inferred).
        if not is_const and type_spec is None and value is None:
            self.error_handler.add_syntax_error(f"Variable '{name_token.value}' must have a type or an initial value.", name_token.lineno)
            return None


        if not self.consume('SEMI', "Expected ';' after declaration"):
            return None

        if is_const:
            # For const, type_spec in AST can store the inferred type from value (done by semantic analyzer)
            # For now, parser can pass None or try basic inference if value is a literal.
            inferred_type_for_const = None
            if value: # Basic inference for AST node if it's a simple literal
                if isinstance(value, Integer): inferred_type_for_const = 'int'
                elif isinstance(value, Float): inferred_type_for_const = 'float'
                # ... and so on for other literals
            return ConstantDecl(name_token.value, value, type_spec=inferred_type_for_const, lineno=name_token.lineno)
        else:
            return VariableDecl(name_token.value, type_spec, value, lineno=name_token.lineno)

    def parse_function(self) -> Optional[Node]:
        func_token = self.match('FUNC')
        if not func_token: return None 

        name_token = self.consume('ID', "Expected function name")
        if not name_token: return None

        if not self.consume('LPAREN', "Expected '(' after function name"):
            return None

        params: List[Parameter] = []
        if self.current_token and self.current_token.type != 'RPAREN':
            while True:
                param_name_token = self.consume('ID', "Expected parameter name")
                if not param_name_token: return None
                
                param_type_token = self.consume_type("Expected parameter type")
                if not param_type_token: return None
                
                params.append(Parameter(param_name_token.value, param_type_token.value, lineno=param_name_token.lineno))
                if not self.match('COMMA'): break # No comma, so end of parameters
        
        if not self.consume('RPAREN', "Expected ')' after parameters"):
            return None

        return_gox_type_token = self.consume_type("Expected return type for function")
        if not return_gox_type_token: return None
        
        body_statements = self.parse_statements_block() 
        # parse_statements_block returns List[Node], [] if block is empty or malformed.
        # It handles LBRACE/RBRACE consumption.

        return FunctionDecl(name_token.value, params, return_gox_type_token.value, body_statements, lineno=func_token.lineno)

    def parse_import(self) -> Optional[Node]:
        import_token = self.match('IMPORT')
        if not import_token: return None
        
        node: Optional[Node] = None
        if self.current_token and self.current_token.type == 'FUNC': # `import func ...`
            self.advance() # Consume FUNC
            
            func_name_token = self.consume('ID', "Expected function name for import")
            if not func_name_token: return None
                
            if not self.consume('LPAREN', "Expected '(' after imported function name"):
                return None
                
            params: List[Parameter] = []
            if self.current_token and self.current_token.type != 'RPAREN':
                while True:
                    param_name = self.consume('ID', "Expected parameter name in import")
                    if not param_name: return None
                    param_type = self.consume_type("Expected parameter type in import") # consume_type returns Token
                    if not param_type: return None
                    params.append(Parameter(param_name.value, param_type.value, lineno=param_name.lineno))
                    if not self.match('COMMA'): break
            
            if not self.consume('RPAREN', "Expected ')' after imported function parameters"):
                return None
                
            return_gox_type = self.consume_type("Expected return type for imported function") # consume_type returns Token
            if not return_gox_type: return None
                
            if not self.consume('SEMI', "Expected ';' after import func statement"):
                return None
                
            node = FunctionImportDecl(func_name_token.value, params, return_gox_type.value, lineno=import_token.lineno)
        
        elif module_token := self.match('ID'): # `import module_name;`
            if not self.consume('SEMI', "Expected ';' after import module statement"):
                return None
            node = ImportDecl(module_token.value, lineno=import_token.lineno)
        else:
            self.error_handler.add_syntax_error("Expected 'func' or module name after 'import'", import_token.lineno)

        return node

    def parse_print(self) -> Optional[Node]:
        print_token = self.match('PRINT')
        if not print_token: return None

        expr = self.parse_expression()
        if not expr:
            self.error_handler.add_syntax_error("Expected expression for print statement", print_token.lineno)
            return None
            
        if not self.consume('SEMI', "Expected ';' after print statement"):
            return None
            
        return Print(expr, lineno=print_token.lineno)

    def parse_if(self) -> Optional[Node]:
        if_token = self.match('IF')
        if not if_token: return None

        # Condition for if/while in GoX examples does not have parentheses
        condition = self.parse_expression()
        if not condition:
            self.error_handler.add_syntax_error("Expected condition expression in if statement", if_token.lineno)
            return None
            
        consequence_statements = self.parse_statements_block() # Returns List[Node]

        alternative_statements: Optional[List[Node]] = None
        if self.match('ELSE'):
            alternative_statements = self.parse_statements_block() # Returns List[Node]
        
        return If(condition, consequence_statements, alternative_statements, lineno=if_token.lineno)

    def parse_while(self) -> Optional[Node]:
        while_token = self.match('WHILE')
        if not while_token: return None

        # Condition for if/while in GoX examples does not have parentheses
        condition = self.parse_expression()
        if not condition:
            self.error_handler.add_syntax_error("Expected condition expression in while statement", while_token.lineno)
            return None
            
        body_statements = self.parse_statements_block() # Returns List[Node]

        return While(condition, body_statements, lineno=while_token.lineno)

    def parse_assignment(self) -> Optional[Node]:
        # lineno for Assignment node should be the line of the target or backtick
        start_lineno = self.current_token.lineno if self.current_token else -1

        # The target of an assignment is a primary expression that results in a Location or MemoryAddress
        # e.g. `id` or `` `(base + i) ``
        target = self.parse_primary() # This will parse ID or `expr
        
        if not target or not isinstance(target, (Location, MemoryAddress)):
            # If parse_primary succeeded but didn't return a valid L-value type
            if target: # Check if a node was returned but was not Location/MemoryAddress
                 self.error_handler.add_syntax_error(f"Invalid target for assignment (not a variable or memory address). Got {type(target).__name__}", start_lineno)
            # If target is None, parse_primary should have already reported an error.
            return None

        if not self.consume('ASSIGN', "Expected '=' after assignment target"):
            return None
        
        expr = self.parse_expression()
        if not expr:
            # Error should be reported by parse_expression or its children
            # Add a generic one if not, to ensure an error is present
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                 self.error_handler.add_syntax_error("Expected expression on the right-hand side of assignment.", 
                                              self.current_token.lineno if self.current_token else start_lineno)
            return None
        
        if not self.consume('SEMI', "Expected ';' after assignment statement"):
            return None
        
        # Use target's lineno if available, otherwise the start_lineno captured
        assign_lineno = target.lineno if target.lineno is not None else start_lineno
        return Assignment(target, expr, lineno=assign_lineno)

    def parse_return(self) -> Optional[Node]:
        return_token = self.match('RETURN')
        if not return_token: return None

        expr: Optional[Node] = None
        # Check if there's an expression to return or if it's a bare `return;`
        if self.current_token and self.current_token.type != 'SEMI':
            expr = self.parse_expression()
            if not expr: # An expression was expected but couldn't be parsed
                self.error_handler.add_syntax_error("Expected expression or ';' after return", return_token.lineno)
                return None
            
        if not self.consume('SEMI', "Expected ';' after return statement"):
            return None
            
        return Return(expr, lineno=return_token.lineno) # expr can be None


# main function adapted from your provided version
def main():
    if len(sys.argv) < 2:
        print("Usage: python Parser.py <source_file> [output_json_file]")
        sys.exit(1)

    source_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else source_file.replace('.gox', '.ast.json') # Default output

    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
        sys.exit(1)
    
    error_handler = ErrorHandler()
    
    print(f"Tokenizing {source_file}...")
    tokens = tokenize(code, error_handler) 

    if error_handler.has_errors():
        print("\nLexical errors found:")
        error_handler.report_errors() # Use the report_errors method
        sys.exit(1)
    print("Tokenization successful.")

    print(f"\nParsing {source_file}...")
    parser = Parser(tokens, error_handler)
    ast_program_node = parser.parse() 

    if error_handler.has_errors():
        print("\nParsing errors found:")
        error_handler.report_errors()
        # Decide if you want to attempt to save/print a partial AST
        # if ast_program_node:
        #     print("\nPartial AST (due to errors):")
        #     ast_data_for_json = ast_to_json(ast_program_node)
        #     pretty_print_json(ast_data_for_json)
        sys.exit(1)

    if ast_program_node:
        print("Parsing successful. AST generated.")
        
        ast_data_for_json = ast_to_json(ast_program_node) 
        save_ast_to_json(ast_data_for_json, output_file)
        print(f"AST saved to {output_file}")
        
        # pretty_print_json(ast_data_for_json) # Uncomment to also print JSON to console
    elif not error_handler.has_errors(): # No AST but no errors
        print("Parsing completed: No AST generated (possibly empty program or parser logic issue).")
    # else: errors already handled

if __name__ == "__main__":
    # Ensure imports are correct if run as script vs part of package
    from Lexer import tokenize # If Lexer.py is in the same directory
    main()