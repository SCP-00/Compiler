# Parser.py
import sys
from typing import List, Optional
from Lexer import Token
from Nodes_AST import * 
from Error import ErrorHandler
# from AST_to_JSON import save_ast_to_json, ast_to_json, pretty_print_json # Not used by Parser class itself

class Parser:
    def __init__(self, tokens: List[Token], error_handler: ErrorHandler):
        self.tokens: List[Token] = tokens
        self.error_handler: ErrorHandler = error_handler
        self.pos: int = 0
        self.current_token: Optional[Token] = self.tokens[0] if tokens else None
        self.debug_mode: bool = False # Set to True for simple debug prints

    def _log_debug(self, message: str):
        if self.debug_mode:
            # Basic indentation based on call stack depth for readability
            # This is a simplified version of your previous debug stack logic
            depth = sum(1 for frame in sys._current_frames().values() if frame.f_code.co_name.startswith('parse_'))
            indent = "  " * depth
            print(f"{indent}[Parser DEBUG] {message}")

    def advance(self) -> None:
        # if self.debug_mode and self.current_token:
        #     self._log_debug(f"Consuming: {self.current_token.type} = '{self.current_token.value}' at line {self.current_token.lineno}")
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
        # if self.debug_mode and self.current_token:
        #     self._log_debug(f"Next token: {self.current_token.type} ('{self.current_token.value}')")


    def peek(self, offset: int = 1) -> Optional[Token]:
        peek_pos = self.pos + offset 
        return self.tokens[peek_pos] if peek_pos < len(self.tokens) else None

    def match(self, *expected_types: str) -> Optional[Token]:
        if self.current_token and self.current_token.type in expected_types:
            token = self.current_token
            # self._log_debug(f"Matched and consumed: {token.type} ('{token.value}')")
            self.advance()
            return token
        return None

    def consume(self, expected_type: str, error_message: str) -> Optional[Token]:
        if self.current_token and self.current_token.type == expected_type:
            token = self.current_token
            # self._log_debug(f"Consumed expected: {token.type} ('{token.value}')")
            self.advance()
            return token
        
        err_lineno = self.current_token.lineno if self.current_token else "End of file"
        current_val_str = f" ('{self.current_token.value}')" if self.current_token and self.current_token.value else ""
        current_type_str = f"'{self.current_token.type}{current_val_str}'" if self.current_token else "end of file"
        full_error_message = f"{error_message} (Expected '{expected_type}', got {current_type_str})"
        self.error_handler.add_syntax_error(full_error_message, err_lineno)
        # self._log_debug(f"Consume FAILED: {full_error_message}")
        return None
    
    def consume_type(self, error_message_prefix: str) -> Optional[Token]:
        type_token = self.match('INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE')
        if not type_token:
            err_lineno = self.current_token.lineno if self.current_token else "End of file"
            self.error_handler.add_syntax_error(f"{error_message_prefix} (e.g., int, float, bool, string, char)", err_lineno)
            return None
        
        # Normalize type value for AST
        if type_token.value == "float_type": type_token.value = "float"
        elif type_token.value == "string_type": type_token.value = "string"
        elif type_token.value == "char_type": type_token.value = "char"
        return type_token

    # --- Main Parsing Method ---
    def parse(self) -> Optional[Program]:
        self._log_debug("--- Starting Program Parsing ---")
        top_level_nodes: List[Node] = []
        
        while self.current_token:
            node: Optional[Node] = None
            # Store error count before attempting to parse a top-level item
            # This helps in detecting if a sub-parser failed silently without advancing or reporting error
            errors_before_item = self.error_handler.get_error_count() 

            token_type = self.current_token.type
            self._log_debug(f"Top-level, current token: {token_type} ('{self.current_token.value}')")

            if token_type == 'IMPORT':
                node = self.parse_import()
            elif token_type in ['VAR', 'CONST']:
                node = self.parse_declaration()
            elif token_type == 'FUNC':
                node = self.parse_function()
            else:
                # For global scripting: try to parse any other valid statement
                # This includes assignments, prints, if, while, func_calls, etc.
                # self._log_debug(f"Attempting to parse as top-level statement: {token_type}")
                node = self.parse_statement() 
                
                if node is None and self.error_handler.get_error_count() == errors_before_item:
                    # parse_statement returned None but did not report a new error.
                    # This means the current token does not start any known statement.
                    # It's an unexpected token at the top level.
                    self.error_handler.add_syntax_error(
                        f"Unexpected token or construct at top level: {self.current_token.type} ('{self.current_token.value}')",
                        self.current_token.lineno
                    )
                    self.advance() # Advance to prevent infinite loop on this token
            
            if node:
                top_level_nodes.append(node)
                # self._log_debug(f"Added top-level node: {type(node).__name__}")
            elif not self.current_token: # EOF reached
                break
            # If node is None and errors *were* added, the loop continues, errors are already logged.

        self._log_debug(f"--- Finished Program Parsing ({len(top_level_nodes)} nodes) ---")
        return Program(top_level_nodes)

    # --- Statement Parsers ---
    def parse_statement(self) -> Optional[Node]:
        # self._log_debug(f"parse_statement, current: {self.current_token.type if self.current_token else 'None'}")
        node: Optional[Node] = None
        if not self.current_token: return None

        token_type = self.current_token.type
        
        if token_type in ['VAR', 'CONST']: # Local declarations if inside a block, or global if parse() calls this
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
        elif token_type == 'ID': 
            # Could be assignment `id = expr;` OR function call statement `id(args);`
            # We need to look at the token *after* the potential function call part.
            # A robust way: parse as primary. If it's FunctionCall and followed by SEMI, it's a statement.
            # If it's Location and followed by ASSIGN, it's an assignment.
            
            # Simpler predictive parsing: if ID is followed by LPAREN, assume func call statement for now.
            # This is tricky because `a = myfunc(b) + c;` starts with `myfunc(b)` as part of expression.
            # The distinction often relies on whether the result of parse_primary() is used in an assignment
            # or stands alone followed by a semicolon.

            # Let's try parsing a primary. This can be Location or FunctionCall
            potential_lhs_or_call = self.parse_primary()

            if isinstance(potential_lhs_or_call, FunctionCall): # e.g. my_func(args)
                if self.consume('SEMI', "Expected ';' after function call statement"):
                    node = potential_lhs_or_call # It was `my_func(args);`
                # else: error already reported by consume.
            elif isinstance(potential_lhs_or_call, (Location, MemoryAddress)): # e.g. x or `addr
                # Now check if it's an assignment
                if self.match('ASSIGN'):
                    expr = self.parse_expression()
                    if not expr: # Error in RHS
                        if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                            self.error_handler.add_syntax_error("Expected expression on the right-hand side of assignment.", 
                                                          self.current_token.lineno if self.current_token else potential_lhs_or_call.lineno)
                        return None # Error in RHS parsing
                    if not self.consume('SEMI', "Expected ';' after assignment statement"):
                        return None # Error reported by consume
                    node = Assignment(potential_lhs_or_call, expr, lineno=potential_lhs_or_call.lineno)
                else: # It was just an ID or `expr not followed by = or ; (if calls are handled above)
                      # This path might be an error if expressions alone cannot be statements (unless it was a call already handled)
                    self.error_handler.add_syntax_error(
                        f"Unexpected token '{self.current_token.value if self.current_token else ''}' after identifier/memory access. Expected '=' for assignment or was not a standalone statement.",
                        potential_lhs_or_call.lineno
                    )
            elif potential_lhs_or_call is None: # Error in parse_primary
                pass # Error already reported
            else: # parse_primary returned something else not assignable and not a call
                self.error_handler.add_syntax_error(
                    f"Invalid start of a statement. Expression of type {type(potential_lhs_or_call).__name__} cannot stand alone here.",
                    potential_lhs_or_call.lineno
                )

        elif token_type == 'BACKTICK': # Assignment to memory: `expr = ...
            # This is now handled by the ID case if parse_primary parses `expr as MemoryAddress
            # And then the assignment logic takes over.
            # If parse_primary doesn't handle `expr directly, then parse_assignment is needed.
            # Given current parse_primary, this direct BACKTICK case might be redundant here.
            # Let's assume BACKTICK is handled by parse_primary for L-Values.
            # If it's `addr = val;`, parse_primary gets `addr, then it's an assignment.
            node = self.parse_assignment() # This will call parse_primary for the target.

        # If node is still None and no error reported by specific parsers above,
        # it means the token_type doesn't start any known statement.
        # This should be caught by the caller (parse() or parse_statements_block)
        # or we can report it here if this function is meant to be exhaustive.
        # For now, let it return None if no statement matched.

        return node

    def parse_statements_block(self) -> List[Node]:
        # self._log_debug("parse_statements_block")
        statements: List[Node] = []
        if not self.consume('LBRACE', "Expected '{' to start block"):
            return [] 
        
        while self.current_token and self.current_token.type != 'RBRACE':
            errors_before_stmt = self.error_handler.get_error_count()
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            elif self.current_token and self.current_token.type != 'RBRACE':
                if self.error_handler.get_error_count() == errors_before_stmt:
                    # parse_statement failed to parse AND didn't log an error.
                    # This means current token is truly unexpected for any statement.
                    self.error_handler.add_syntax_error(
                        f"Unexpected token '{self.current_token.value}' inside block.", 
                        self.current_token.lineno
                    )
                # Always advance if parse_statement failed, to avoid infinite loop on the problematic token.
                self.advance() 
            elif not self.current_token: # EOF before RBRACE
                break # Error will be caught by consume RBRACE
        
        self.consume('RBRACE', "Expected '}' to end block") # Reports error if missing
        return statements

    def parse_control_statement(self, keyword: str, node_class: type) -> Optional[Node]:
        start_token = self.match(keyword)
        if not start_token: return None 
        if not self.consume('SEMI', f"Missing ';' after {keyword.lower()} statement"):
            return None
        return node_class(lineno=start_token.lineno)

    # --- Expression Parsers (Precedence: LogicalOR -> ... -> Primary) ---
    def parse_expression(self) -> Optional[Node]:
        return self.parse_logical_or()

    def parse_logical_or(self) -> Optional[Node]:
        node = self.parse_logical_and()
        while node and self.current_token and self.current_token.type == 'LOR':
            op_token = self.current_token # Save before advancing
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
        while node and self.current_token and self.current_token.type in ['TIMES', 'DIVIDE', 'MOD']:
            op_token = self.current_token
            self.advance()
            right = self.parse_unary()
            if not right:
                self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno)
                return None
            node = BinOp(op_token.value, node, right, lineno=op_token.lineno)
        return node

    def parse_unary(self) -> Optional[Node]:
        if op_token := self.match('PLUS', 'MINUS', 'NOT'):
            operand = self.parse_unary() 
            if not operand:
                # Error already reported if operand parsing failed. If not, means missing operand.
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error(f"Expected operand after unary operator '{op_token.value}'", op_token.lineno)
                return None
            return UnaryOp(op_token.value, operand, lineno=op_token.lineno)
        elif op_token := self.match('MEM_ALLOC'): # ^expr
            size_expr = self.parse_unary() 
            if not size_expr:
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error(f"Expected size expression after memory allocation operator '{op_token.value}'", op_token.lineno)
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
            expr_start_line = self.tokens[self.pos-1].lineno # Line of LPAREN
            expr = self.parse_expression()
            if not expr: # Error should be reported by parse_expression
                # If not, means ( ) which is an error
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                     self.error_handler.add_syntax_error("Empty parentheses in expression.", expr_start_line)
                return None
            if not self.consume('RPAREN', "Expected ')' after parenthesized expression"):
                return None
            node = expr
        elif backtick_token := self.match('BACKTICK'): # `expr
            address_expression = self.parse_expression() 
            if not address_expression:
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error("Expected address expression after '`'", backtick_token.lineno)
                return None
            node = MemoryAddress(address_expression, lineno=backtick_token.lineno)
        elif id_token := self.match('ID'):
            if self.current_token and self.current_token.type == 'LPAREN': # Function call as expression
                self.advance() # Consume LPAREN
                args: List[Node] = []
                if not self.match('RPAREN'): # If not empty arg list
                    while True:
                        arg = self.parse_expression()
                        if not arg: return None 
                        args.append(arg)
                        if not self.match('COMMA'): break # No comma, end of args
                    if not self.consume('RPAREN', "Expected ')' or ',' after function argument"): return None
                node = FunctionCall(id_token.value, args, lineno=id_token.lineno)
            else: # Variable Location
                node = Location(id_token.value, lineno=id_token.lineno)
        elif self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE']:
            type_keyword_token = self.consume_type("Expected type keyword for cast") 
            if not type_keyword_token: return None
            if not self.consume('LPAREN', f"Expected '(' after type '{type_keyword_token.value}' for cast"): return None
            expr_to_cast = self.parse_expression()
            if not expr_to_cast:
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error(f"Expected expression inside cast to '{type_keyword_token.value}'", type_keyword_token.lineno)
                return None
            if not self.consume('RPAREN', f"Expected ')' after cast to '{type_keyword_token.value}'"): return None
            node = TypeCast(type_keyword_token.value, expr_to_cast, lineno=type_keyword_token.lineno)
        else:
            # This token does not start a primary expression.
            # This case should ideally be caught by the caller if it expects a primary expression.
            # If parse_primary is called and this hits, it's an error *at this point*.
            self.error_handler.add_syntax_error(f"Unexpected token, cannot form primary expression: {token.type} ('{token.value}')", token.lineno)
            self.advance() # Skip to try to recover, though recovery from expression errors is hard.
        return node

    def create_literal_node(self, token: Token) -> Node:
        if token.type == 'INTEGER': return Integer(int(token.value), lineno=token.lineno)
        if token.type == 'FLOAT': return Float(float(token.value), lineno=token.lineno)
        if token.type == 'STRING': return String(token.value, lineno=token.lineno) 
        if token.type == 'CHAR': return Char(token.value, lineno=token.lineno)    
        if token.type == 'TRUE': return Boolean(True, lineno=token.lineno)
        if token.type == 'FALSE': return Boolean(False, lineno=token.lineno)
        raise ValueError(f"Internal error: create_literal_node called with non-literal token {token.type}")

    def parse_declaration(self) -> Optional[Node]:
        is_const = True if self.current_token and self.current_token.type == 'CONST' else False
        decl_keyword_token = self.match('VAR', 'CONST')
        if not decl_keyword_token: return None

        name_token = self.consume('ID', "Expected identifier after 'var' or 'const'")
        if not name_token: return None

        type_spec: Optional[str] = None
        if not is_const and self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE']:
            type_spec_token = self.consume_type("Expected type specifier for variable")
            if type_spec_token: type_spec = type_spec_token.value 

        value: Optional[Node] = None
        if self.match('ASSIGN'):
            value = self.parse_expression()
            if not value:
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error("Expected expression for initialization", 
                                                  self.current_token.lineno if self.current_token else name_token.lineno)
                return None
        
        if is_const and value is None:
            self.error_handler.add_syntax_error(f"Constant '{name_token.value}' must be initialized.", name_token.lineno)
            return None
        
        if not is_const and type_spec is None and value is None:
            self.error_handler.add_syntax_error(f"Variable '{name_token.value}' must have an explicit type or an initial value.", name_token.lineno)
            return None

        if not self.consume('SEMI', "Expected ';' after declaration"):
            return None

        if is_const:
            inferred_type_for_const = None # Semantic analyzer will fill this properly
            if value: # Basic inference for AST node if it's a simple literal
                if isinstance(value, Integer): inferred_type_for_const = 'int'
                elif isinstance(value, Float): inferred_type_for_const = 'float'
                elif isinstance(value, String): inferred_type_for_const = 'string'
                elif isinstance(value, Char): inferred_type_for_const = 'char'
                elif isinstance(value, Boolean): inferred_type_for_const = 'bool'
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
        if not self.match('RPAREN'): 
            while True:
                param_name_token = self.consume('ID', "Expected parameter name")
                if not param_name_token: return None
                
                param_type_token = self.consume_type("Expected parameter type")
                if not param_type_token: return None
                
                params.append(Parameter(param_name_token.value, param_type_token.value, lineno=param_name_token.lineno))
                if not self.match('COMMA'): break 
            if not self.consume('RPAREN', "Expected ')' or ',' after parameter"):
                 return None
        
        return_gox_type_token = self.consume_type("Expected return type for function")
        if not return_gox_type_token: return None
        
        body_statements = self.parse_statements_block() 
        return FunctionDecl(name_token.value, params, return_gox_type_token.value, body_statements, lineno=func_token.lineno)

    def parse_import(self) -> Optional[Node]:
        import_token = self.match('IMPORT')
        if not import_token: return None
        
        node: Optional[Node] = None
        if self.match('FUNC'): 
            func_name_token = self.consume('ID', "Expected function name for import")
            if not func_name_token: return None
            if not self.consume('LPAREN', "Expected '(' after imported function name"): return None
            params: List[Parameter] = []
            if not self.match('RPAREN'):
                while True:
                    param_name = self.consume('ID', "Expected parameter name in import")
                    if not param_name: return None
                    param_type = self.consume_type("Expected parameter type in import")
                    if not param_type: return None
                    params.append(Parameter(param_name.value, param_type.value, lineno=param_name.lineno))
                    if not self.match('COMMA'): break
                if not self.consume('RPAREN', "Expected ')' or ',' after imported parameter"): return None
            return_gox_type = self.consume_type("Expected return type for imported function")
            if not return_gox_type: return None
            if not self.consume('SEMI', "Expected ';' after import func statement"): return None
            node = FunctionImportDecl(func_name_token.value, params, return_gox_type.value, lineno=import_token.lineno)
        elif module_token := self.match('ID'): 
            if not self.consume('SEMI', "Expected ';' after import module statement"): return None
            node = ImportDecl(module_token.value, lineno=import_token.lineno)
        else:
            self.error_handler.add_syntax_error("Expected 'func' or module name after 'import'", import_token.lineno)
        return node

    def parse_print(self) -> Optional[Node]:
        print_token = self.match('PRINT')
        if not print_token: return None
        expr = self.parse_expression()
        if not expr:
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                 self.error_handler.add_syntax_error("Expected expression for print statement", print_token.lineno)
            return None
        if not self.consume('SEMI', "Expected ';' after print statement"): return None
        return Print(expr, lineno=print_token.lineno)

    def parse_if(self) -> Optional[Node]:
        if_token = self.match('IF')
        if not if_token: return None
        condition = self.parse_expression() 
        if not condition:
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                self.error_handler.add_syntax_error("Expected condition expression in if statement", if_token.lineno)
            return None
        consequence_statements = self.parse_statements_block()
        alternative_statements: Optional[List[Node]] = None
        if self.match('ELSE'):
            alternative_statements = self.parse_statements_block()
        return If(condition, consequence_statements, alternative_statements, lineno=if_token.lineno)

    def parse_while(self) -> Optional[Node]:
        while_token = self.match('WHILE')
        if not while_token: return None
        condition = self.parse_expression()
        if not condition:
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                self.error_handler.add_syntax_error("Expected condition expression in while statement", while_token.lineno)
            return None
        body_statements = self.parse_statements_block()
        return While(condition, body_statements, lineno=while_token.lineno)

    def parse_assignment(self) -> Optional[Node]:
        # This is called when parse_statement has determined it's likely an assignment
        # (e.g., started with ID not followed by LPAREN, or started with BACKTICK)
        # The target (ID or `expr) should be parsed by parse_primary first
        start_lineno = self.current_token.lineno if self.current_token else -1
        
        # We re-parse primary here to get the target. This might seem redundant
        # if parse_statement already did, but it simplifies parse_statement's logic.
        # Alternatively, parse_statement could pass the potential_lhs node.
        # For now, keep it simple:
        target = self.parse_primary()
        
        if not target or not isinstance(target, (Location, MemoryAddress)):
            if target: 
                 self.error_handler.add_syntax_error(f"Invalid target for assignment. Expected variable or memory location, got {type(target).__name__}", start_lineno)
            # else: parse_primary should have reported if it returned None without error
            elif not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                self.error_handler.add_syntax_error(f"Invalid or missing target for assignment.", start_lineno)
            return None

        if not self.consume('ASSIGN', "Expected '=' after assignment target"): return None
        
        expr = self.parse_expression()
        if not expr:
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                 self.error_handler.add_syntax_error("Expected expression on the right-hand side of assignment.", 
                                              self.current_token.lineno if self.current_token else start_lineno)
            return None
        
        if not self.consume('SEMI', "Expected ';' after assignment statement"): return None
        
        assign_lineno = target.lineno if hasattr(target, 'lineno') and target.lineno is not None else start_lineno
        return Assignment(target, expr, lineno=assign_lineno)

    def parse_return(self) -> Optional[Node]:
        return_token = self.match('RETURN')
        if not return_token: return None
        expr: Optional[Node] = None
        if self.current_token and self.current_token.type != 'SEMI':
            expr = self.parse_expression()
            if not expr: 
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error("Expected expression or ';' after return", return_token.lineno)
                return None
        if not self.consume('SEMI', "Expected ';' after return statement"): return None
        return Return(expr, lineno=return_token.lineno) 

def main():
    from Lexer import tokenize 
    from Error import ErrorHandler 
    from AST_to_JSON import ast_to_json, save_ast_to_json, pretty_print_json

    if len(sys.argv) < 2:
        print("Usage: python Parser.py <source_file> [output_json_file]")
        sys.exit(1)

    source_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else source_file.replace('.gox', '.ast.json')

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
        error_handler.report_errors()
        sys.exit(1)
    print(f"Tokenization successful ({len(tokens)} tokens).")

    print(f"\nParsing {source_file}...")
    parser = Parser(tokens, error_handler)
    ast_program_node = parser.parse() 

    if error_handler.has_errors():
        print("\nParsing errors found:")
        error_handler.report_errors()
        if ast_program_node: # Try to output partial AST if any
            print("\nAttempting to output partial AST (due to parsing errors)...")
            try:
                ast_data_for_json = ast_to_json(ast_program_node)
                error_output_file = output_file.replace('.ast.json', '.ast_errors.json')
                save_ast_to_json(ast_data_for_json, error_output_file)
                # pretty_print_json(ast_data_for_json)
            except Exception as e:
                print(f"Could not serialize partial AST due to: {e}")
        sys.exit(1)

    if ast_program_node:
        print("Parsing successful. AST generated.")
        ast_data_for_json = ast_to_json(ast_program_node) 
        save_ast_to_json(ast_data_for_json, output_file)
        pretty_print_json(ast_data_for_json) 
    elif not error_handler.has_errors():
        print("Parsing completed: No AST generated (program might be empty).")

if __name__ == "__main__":
    main()
