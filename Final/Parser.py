# Parser.py
import sys
from typing import List, Optional, Union, Callable 
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
            depth = 0
            try: # Calculate depth based on 'parse_' methods in call stack
                depth = sum(1 for frame_info in sys._current_frames().values() 
                            if frame_info.f_code.co_name.startswith('parse_'))
            except Exception: # sys._current_frames might not be available or fail
                pass
            indent = "  " * depth
            print(f"{indent}[Parser DEBUG] {message}")

    def advance(self) -> None:
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek(self, offset: int = 1) -> Optional[Token]:
        peek_pos = self.pos + offset 
        return self.tokens[peek_pos] if peek_pos < len(self.tokens) else None

    def match(self, *expected_types: str) -> Optional[Token]:
        if self.current_token and self.current_token.type in expected_types:
            token = self.current_token
            self.advance()
            return token
        return None

    def consume(self, expected_type: str, error_message: str) -> Optional[Token]:
        if self.current_token and self.current_token.type == expected_type:
            token = self.current_token
            self.advance()
            return token
        
        err_lineno = self.current_token.lineno if self.current_token else "End of file"
        current_val_str = f" ('{self.current_token.value}')" if self.current_token and self.current_token.value else ""
        current_type_str = f"'{self.current_token.type}{current_val_str}'" if self.current_token else "end of file"
        full_error_message = f"{error_message} (Expected '{expected_type}', got {current_type_str})"
        self.error_handler.add_syntax_error(full_error_message, err_lineno)
        return None
    
    def consume_type(self, error_message_prefix: str) -> Optional[Token]:
        type_token = self.match('INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE')
        if not type_token:
            err_lineno = self.current_token.lineno if self.current_token else "End of file"
            self.error_handler.add_syntax_error(f"{error_message_prefix} (e.g., int, float, bool, string, char)", err_lineno)
            return None
        
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
            errors_before_item = self.error_handler.get_error_count() 

            token_type = self.current_token.type
            self._log_debug(f"Top-level, current token: {token_type} ('{self.current_token.value if self.current_token else ''}')")

            if token_type == 'IMPORT':
                node = self.parse_import()
            elif token_type in ['VAR', 'CONST']:
                node = self.parse_declaration()
            elif token_type == 'FUNC':
                node = self.parse_function() # This now has better error recovery
            else:
                node = self.parse_statement() 
                if node is None and self.error_handler.get_error_count() == errors_before_item:
                    self.error_handler.add_syntax_error(
                        f"Unexpected token or construct at top level: {self.current_token.type} ('{self.current_token.value}')",
                        self.current_token.lineno
                    )
                    self.advance() 
            
            if node:
                top_level_nodes.append(node)
            elif not self.current_token: 
                break
        
        self._log_debug(f"--- Finished Program Parsing ({len(top_level_nodes)} nodes) ---")
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
            # This logic tries to parse an expression that could be an L-value or a function call.
            # Then, based on what follows, it decides if it's an assignment or a standalone call statement.
            start_pos = self.pos # Save state for potential backtrack if it's not an assignment
            start_token_for_error = self.current_token

            potential_target_or_call = self.parse_primary()

            if isinstance(potential_target_or_call, (Location, MemoryAddress)) and self.match('ASSIGN'):
                # It's an assignment: target = expr;
                expr = self.parse_expression()
                if not expr:
                    if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0): # Check if parse_expression reported
                        self.error_handler.add_syntax_error("Expected expression on the right-hand side of assignment.", 
                                                      self.current_token.lineno if self.current_token else start_token_for_error.lineno)
                    return None 
                if not self.consume('SEMI', "Expected ';' after assignment statement"):
                    return None 
                node = Assignment(potential_target_or_call, expr, lineno=potential_target_or_call.lineno)
            
            elif isinstance(potential_target_or_call, FunctionCall):
                # It was a function call expression: func_call_expr ;
                if self.consume('SEMI', "Expected ';' after function call statement"):
                    node = potential_target_or_call
                # else error reported by consume
            
            elif potential_target_or_call: # It was some other primary expression not forming a valid statement
                self.error_handler.add_syntax_error(
                    f"Expression of type '{type(potential_target_or_call).__name__}' starting with '{start_token_for_error.value}' cannot stand alone as a statement here. Expected assignment or function call.",
                    start_token_for_error.lineno
                )
                # No need to reset pos, parse_primary consumed tokens.
                # If a SEMI follows, consume it to help parser move on.
                if self.current_token and self.current_token.type == 'SEMI':
                    self.advance()

            # If potential_target_or_call is None, parse_primary already reported an error.
        # else: Token does not start a known statement. Will be handled by caller if it's unexpected there.
        
        return node

    def parse_statements_block(self) -> List[Node]:
        statements: List[Node] = []
        if not self.consume('LBRACE', "Expected '{' to start block"):
            return [] 
        
        while self.current_token and self.current_token.type != 'RBRACE':
            errors_before_stmt = self.error_handler.get_error_count()
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            elif self.current_token and self.current_token.type != 'RBRACE': # Error occurred or unhandled token
                if self.error_handler.get_error_count() == errors_before_stmt: # No new error, so current token is the problem
                    self.error_handler.add_syntax_error(
                        f"Unexpected token '{self.current_token.value}' inside block.", 
                        self.current_token.lineno
                    )
                self.advance() # Advance to try to recover from the problematic token
            elif not self.current_token: 
                break 
        
        self.consume('RBRACE', "Expected '}' to end block")
        return statements

    def parse_control_statement(self, keyword: str, node_class: type) -> Optional[Node]:
        start_token = self.match(keyword)
        if not start_token: return None 
        if not self.consume('SEMI', f"Missing ';' after {keyword.lower()} statement"):
            return None
        return node_class(lineno=start_token.lineno)

    # --- Expression Parsers ---
    # (parse_expression down to parse_primary remain largely the same as your last correct version)
    # Ensure they correctly handle operator precedence and associativity.
    def parse_expression(self) -> Optional[Node]:
        return self.parse_logical_or()

    def parse_logical_or(self) -> Optional[Node]: # Lowest precedence
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
        elif lparen_token := self.match('LPAREN'):
            expr = self.parse_expression()
            if not expr: 
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                     self.error_handler.add_syntax_error("Empty or invalid parenthesized expression.", lparen_token.lineno)
                return None
            if not self.consume('RPAREN', "Expected ')' after parenthesized expression"):
                return None
            node = expr
        elif backtick_token := self.match('BACKTICK'): 
            address_expression = self.parse_expression() 
            if not address_expression:
                if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                    self.error_handler.add_syntax_error("Expected address expression after '`'", backtick_token.lineno)
                return None
            node = MemoryAddress(address_expression, lineno=backtick_token.lineno)
        elif id_token := self.match('ID'):
            if self.current_token and self.current_token.type == 'LPAREN': 
                self.advance() 
                args: List[Node] = []
                if not self.match('RPAREN'): 
                    while True:
                        arg = self.parse_expression()
                        if not arg: return None 
                        args.append(arg)
                        if not self.match('COMMA'): break 
                    if not self.consume('RPAREN', "Expected ')' or ',' after function argument"): return None
                node = FunctionCall(id_token.value, args, lineno=id_token.lineno)
            else: 
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
            # Let the caller (e.g., parse_statement or a higher-level expression parser) decide if it's an error.
            # Returning None here indicates failure to parse a primary.
            pass 
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

        if not self.consume('SEMI', "Expected ';' after declaration"): return None

        if is_const:
            inferred_type_for_const = None 
            if value: 
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

        if not self.consume('LPAREN', "Expected '(' after function name"): return None

        params: List[Parameter] = []
        # Parameter parsing loop: `(param1 type1, param2 type2, ...)` or `()`
        if not self.match('RPAREN'): # If not an immediate RPAREN, there are parameters
            while True: # Loop for each parameter
                param_name_token = self.consume('ID', "Expected parameter name")
                if not param_name_token:
                    self._synchronize_to_RBRACE_or_EOF() # Attempt to recover
                    return None 
                
                param_type_token = self.consume_type("Expected parameter type")
                if not param_type_token:
                    self._synchronize_to_RBRACE_or_EOF()
                    return None
                
                params.append(Parameter(param_name_token.value, param_type_token.value, lineno=param_name_token.lineno))
                
                if not self.match('COMMA'): # No comma, so this should be the last parameter
                    break 
            # After loop, expect RPAREN
            if not self.consume('RPAREN', "Expected ')' or ',' after parameter"):
                self._synchronize_to_RBRACE_or_EOF()
                return None
        # If it was an immediate RPAREN, params list remains empty, which is correct.
        
        return_gox_type_token = self.consume_type("Expected return type for function")
        if not return_gox_type_token:
            self._synchronize_to_RBRACE_or_EOF()
            return None
        
        body_statements = self.parse_statements_block() 
        # parse_statements_block handles LBRACE and RBRACE consumption for the block.
        # If it returns [], it means the block was malformed (e.g. missing '{') or empty.

        return FunctionDecl(name_token.value, params, return_gox_type_token.value, body_statements, lineno=func_token.lineno)

    def _synchronize_to_RBRACE_or_EOF(self):
        """Advance tokens until an RBRACE or EOF is found, for error recovery."""
        self._log_debug("Synchronizing: skipping tokens until '}' or EOF...")
        while self.current_token and self.current_token.type != 'RBRACE':
            self.advance()
        if self.current_token and self.current_token.type == 'RBRACE':
             self.advance() # Consume the RBRACE to clean up

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
        # This method is now primarily called when parse_statement identifies an assignment context
        # (e.g., after parsing a Location/MemoryAddress and seeing an '=').
        # However, it can also be called directly if BACKTICK starts the statement.
        
        start_lineno = self.current_token.lineno if self.current_token else -1
        
        # The target (LHS) should have been parsed by parse_primary if called from parse_statement
        # If called directly (e.g. for BACKTICK), parse_primary gets the target.
        target = self.parse_primary()
        
        if not target or not isinstance(target, (Location, MemoryAddress)):
            if target: 
                 self.error_handler.add_syntax_error(f"Invalid target for assignment. Expected variable or memory location, got {type(target).__name__}", start_lineno)
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

# --- Main function for Parser (driver) ---
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
    # parser.debug_mode = True # Enable debug prints for parser
    ast_program_node = parser.parse() 

    if error_handler.has_errors():
        print("\nParsing errors found:")
        error_handler.report_errors()
        if ast_program_node: 
            print("\nAttempting to output partial AST (due to parsing errors)...")
            try:
                ast_data_for_json = ast_to_json(ast_program_node)
                error_output_file = output_file.replace('.ast.json', '.ast_errors.json')
                save_ast_to_json(ast_data_for_json, error_output_file)
                pretty_print_json(ast_data_for_json)
            except Exception as e:
                print(f"Could not serialize/print partial AST due to: {e}")
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
