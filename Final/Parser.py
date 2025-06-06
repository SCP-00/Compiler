# Parser.py
import sys
from typing import List, Optional
from Lexer import Token
from Nodes_AST import * 
from Error import ErrorHandler
# No se usa AST_to_JSON directamente en la clase Parser

class Parser:
    def __init__(self, tokens: List[Token], error_handler: ErrorHandler):
        self.tokens: List[Token] = tokens
        self.error_handler: ErrorHandler = error_handler
        self.pos: int = 0
        self.current_token: Optional[Token] = self.tokens[0] if tokens else None
        # self.debug_mode: bool = False # Debugging can be added with print statements if needed

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
        # Added 'VOID' as a valid type keyword for return types, etc.
        type_token = self.match('INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'VOID')
        if not type_token:
            err_lineno = self.current_token.lineno if self.current_token else "End of file"
            # Check if the current token is an ID that might be an unknown type like 'real'
            current_type_val = f"'{self.current_token.value}'" if self.current_token else "end of file"
            self.error_handler.add_syntax_error(f"{error_message_prefix} (e.g., int, float, bool, string, char, void), got {current_type_val}", err_lineno)
            return None
        
        # Normalize type value for AST
        if type_token.value == "float_type": type_token.value = "float"
        elif type_token.value == "string_type": type_token.value = "string"
        elif type_token.value == "char_type": type_token.value = "char"
        # 'int', 'bool', 'void' are already fine
        return type_token

    def _synchronize(self, recovery_tokens: List[str], stop_before: bool = True):
        """
        General error recovery: advance until a token in recovery_tokens or EOF.
        If stop_before is True, stops *before* consuming the recovery token.
        Otherwise, consumes it.
        """
        while self.current_token:
            if self.current_token.type in recovery_tokens:
                if stop_before:
                    return
                else:
                    self.advance()
                    return
            self.advance()

    def _synchronize_past_block(self):
        """
        Attempts to find an opening LBRACE from the current position (or soon after)
        and skip until its matching RBRACE.
        This is typically called when a construct like a function or block is malformed.
        """
        # Try to find the LBRACE of the block we intend to skip.
        # Advance a few tokens if LBRACE is not immediate, in case of FUNC name ( params ) type {
        skipped_initial_tokens = 0
        max_initial_skip = 10 # Max tokens to look for LBRACE (e.g., func name (params) type)

        while self.current_token and self.current_token.type != 'LBRACE' and skipped_initial_tokens < max_initial_skip:
            # If we hit another major keyword or SEMI before finding LBRACE,
            # it's possible the block was omitted or structure is very wrong.
            if self.current_token.type in ['FUNC', 'VAR', 'CONST', 'IMPORT', 'SEMI', 'RBRACE', 'EOF']:
                return # Give up finding LBRACE for this block, return.
            self.advance()
            skipped_initial_tokens += 1

        if not self.current_token or self.current_token.type != 'LBRACE':
            return # No LBRACE found or EOF.

        # Now we are at an LBRACE (or should be), skip its content
        self.advance() # Consume the LBRACE
        nesting_level = 1
        while self.current_token:
            if self.current_token.type == 'LBRACE':
                nesting_level += 1
            elif self.current_token.type == 'RBRACE':
                nesting_level -= 1
                if nesting_level == 0: # Found the matching closing brace
                    self.advance() # Consume the RBRACE
                    return
            # Safety break: if nesting goes negative or hits EOF
            elif nesting_level < 0 or self.current_token.type == 'EOF':
                return
            self.advance()
        return # Hit EOF while inside nested structure

    # --- Main Parsing Method ---
    def parse(self) -> Program: # Changed to always return a Program node, even if body is empty
        top_level_nodes: List[Node] = []
        
        while self.current_token:
            node: Optional[Node] = None
            errors_before_item = self.error_handler.get_error_count() 

            token_type = self.current_token.type

            if token_type == 'IMPORT':
                node = self.parse_import()
            elif token_type in ['VAR', 'CONST']:
                node = self.parse_declaration()
            elif token_type == 'FUNC':
                node = self.parse_function() 
            else:
                node = self.parse_statement() 
                if node is None and self.error_handler.get_error_count() == errors_before_item:
                    self.error_handler.add_syntax_error(
                        f"Unexpected token or construct at top level: {self.current_token.type} ('{self.current_token.value}')",
                        self.current_token.lineno
                    )
                    self.advance() # Advance to prevent infinite loop on this specific token
            
            if node:
                top_level_nodes.append(node)
            elif not self.current_token: 
                break # EOF
            elif self.error_handler.get_error_count() > errors_before_item:
                # An error was reported and sub-parser returned None. Attempt to synchronize.
                self._synchronize(['FUNC', 'VAR', 'CONST', 'IMPORT', 'SEMI']) # Added SEMI for statement level recovery
                if self.current_token and self.current_token.type == 'SEMI': # Consume SEMI if it's the recovery point
                    self.advance()
        
        return Program(top_level_nodes)

    # --- Statement Parsers ---
    def parse_statement(self) -> Optional[Node]:
        node: Optional[Node] = None
        if not self.current_token or self.current_token.type == 'EOF': return None

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
        elif token_type == 'FUNC': # Detect nested function attempts
            self.error_handler.add_syntax_error(
                "Nested functions are not allowed.",
                self.current_token.lineno
            )
            self.advance() # <<< ADDED THIS LINE: Consume the 'FUNC' token itself
            self._synchronize_past_block() # Try to skip the rest of the malformed func { ... }
            node = None # Indicate error, no valid statement node produced
        elif token_type == 'ID' or token_type == 'BACKTICK':
            start_token_for_error = self.current_token
            potential_lhs_or_call = self.parse_primary()

            if isinstance(potential_lhs_or_call, (Location, MemoryAddress)):
                if self.match('ASSIGN'): 
                    expr = self.parse_expression()
                    if not expr: return None # Error in RHS should be reported by parse_expression
                    if not self.consume('SEMI', "Expected ';' after assignment statement."): return None 
                    node = Assignment(potential_lhs_or_call, expr, lineno=potential_lhs_or_call.lineno)
                else: 
                    self.error_handler.add_syntax_error(
                        f"Identifier or memory access '{start_token_for_error.value}' not followed by '=' for assignment.",
                        start_token_for_error.lineno
                    ) # This might be part of a larger expression if not for statement context
                      # If this function expects only full statements, this is an error.
            elif isinstance(potential_lhs_or_call, FunctionCall):
                if not self.consume('SEMI', "Expected ';' after function call statement."): return None 
                node = potential_lhs_or_call
            elif potential_lhs_or_call is None: pass # Error already reported by parse_primary
            else: 
                self.error_handler.add_syntax_error(
                    f"Expression of type '{type(potential_lhs_or_call).__name__}' cannot stand alone as a statement.",
                    potential_lhs_or_call.lineno
                )
        # If token_type doesn't match any, node remains None. Caller handles it.
        return node

    def parse_statements_block(self) -> List[Node]:
        statements: List[Node] = []
        if not self.consume('LBRACE', "Expected '{' to start block"):
            # If LBRACE is missing, we might be severely off track.
            # Attempt to find a recovery point or the RBRACE.
            # For now, returning empty list and error is reported by consume.
            # A more advanced sync could look for statement starters or the RBRACE.
            # self._synchronize(['RBRACE', 'VAR', 'CONST', 'PRINT', 'IF', 'WHILE', 'ID', 'BACKTICK', 'RETURN', 'BREAK', 'CONTINUE', 'FUNC'], stop_before=True)
            return [] 
        
        while self.current_token and self.current_token.type != 'RBRACE' and self.current_token.type != 'EOF':
            errors_before_stmt = self.error_handler.get_error_count()
            stmt = self.parse_statement()

            if stmt:
                statements.append(stmt)
            else: # stmt is None
                if not self.current_token or self.current_token.type == 'RBRACE' or self.current_token.type == 'EOF':
                    break # Reached end of block or EOF

                # If parse_statement returned None, it either reported an error and should have synchronized,
                # or it didn't know how to handle the current token.
                if self.error_handler.get_error_count() == errors_before_stmt:
                    # parse_statement did not report an error, so this token is unexpected here.
                    self.error_handler.add_syntax_error(
                        f"Unexpected token '{self.current_token.value}' (type: {self.current_token.type}) inside block, cannot start a statement.",
                        self.current_token.lineno
                    )
                    self.advance() # Consume the problematic token to avoid infinite loop
                # else: parse_statement reported an error and presumably synchronized.
                # The token stream should be at a new position. Loop will continue.
        
        if not self.consume('RBRACE', "Expected '}' to end block"):
            # If RBRACE is missing, error is reported by consume.
            # We might be at EOF or another token.
            pass # Statements gathered so far are returned.
        return statements

    def parse_control_statement(self, keyword: str, node_class: type) -> Optional[Node]:
        start_token = self.match(keyword)
        if not start_token: return None 
        if not self.consume('SEMI', f"Missing ';' after {keyword.lower()} statement"):
            # No complex sync here, simple statement. Error already reported.
            return None
        return node_class(lineno=start_token.lineno)

    def parse_expression(self) -> Optional[Node]: return self.parse_logical_or()
    def parse_logical_or(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        node = self.parse_logical_and()
        while node and self.current_token and self.current_token.type == 'LOR':
            op_token = self.current_token 
            self.advance()
            right = self.parse_logical_and()
            if not right: self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno); return None 
            node = LogicalOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
    def parse_logical_and(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        node = self.parse_equality()
        while node and self.current_token and self.current_token.type == 'LAND':
            op_token = self.current_token
            self.advance()
            right = self.parse_equality()
            if not right: self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno); return None
            node = LogicalOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
    def parse_equality(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        node = self.parse_relational()
        while node and self.current_token and self.current_token.type in ['EQ', 'NE']:
            op_token = self.current_token
            self.advance()
            right = self.parse_relational()
            if not right: self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno); return None
            node = CompareOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
    def parse_relational(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        node = self.parse_additive()
        while node and self.current_token and self.current_token.type in ['LT', 'GT', 'LE', 'GE']:
            op_token = self.current_token
            self.advance()
            right = self.parse_additive()
            if not right: self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno); return None
            node = CompareOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
    def parse_additive(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        node = self.parse_multiplicative()
        while node and self.current_token and self.current_token.type in ['PLUS', 'MINUS']:
            op_token = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            if not right: self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno); return None
            node = BinOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
    def parse_multiplicative(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        node = self.parse_unary()
        while node and self.current_token and self.current_token.type in ['TIMES', 'DIVIDE', 'MOD']:
            op_token = self.current_token
            self.advance()
            right = self.parse_unary()
            if not right: self.error_handler.add_syntax_error(f"Expected expression after '{op_token.value}'", op_token.lineno); return None
            node = BinOp(op_token.value, node, right, lineno=op_token.lineno)
        return node
    def parse_unary(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        if op_token := self.match('PLUS', 'MINUS', 'NOT'):
            operand = self.parse_unary() 
            if not operand:
                # Do not add error here if sub-call already did, or if it's just end of tokens
                return None
            return UnaryOp(op_token.value, operand, lineno=op_token.lineno)
        elif op_token := self.match('MEM_ALLOC'): 
            size_expr = self.parse_unary() 
            if not size_expr:
                self.error_handler.add_syntax_error(f"Expected size expression after memory allocation operator '{op_token.value}'", op_token.lineno)
                return None
            return MemoryAllocation(size_expr, lineno=op_token.lineno)
        return self.parse_primary()
    def parse_primary(self) -> Optional[Node]: # ... (same as before, ensure error propagation)
        token = self.current_token; node: Optional[Node] = None
        if not token: return None
        if lit_token := self.match('INTEGER', 'FLOAT', 'STRING', 'CHAR', 'TRUE', 'FALSE'):
            node = self.create_literal_node(lit_token)
        elif lparen_token := self.match('LPAREN'):
            expr = self.parse_expression()
            if not expr: return None
            if not self.consume('RPAREN', "Expected ')' after parenthesized expression"): return None
            node = expr
        elif backtick_token := self.match('BACKTICK'): 
            address_expression = self.parse_expression() 
            if not address_expression: return None
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
            else: node = Location(id_token.value, lineno=id_token.lineno)
        elif self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE']:
            type_keyword_token = self.consume_type("Expected type keyword for cast") 
            if not type_keyword_token: return None
            if not self.consume('LPAREN', f"Expected '(' after type '{type_keyword_token.value}' for cast"): return None
            expr_to_cast = self.parse_expression()
            if not expr_to_cast: return None
            if not self.consume('RPAREN', f"Expected ')' after cast to '{type_keyword_token.value}'"): return None
            node = TypeCast(type_keyword_token.value, expr_to_cast, lineno=type_keyword_token.lineno)
        # else: If no primary matches, return None. Caller will decide if it's an error.
        return node

    def create_literal_node(self, token: Token) -> Node: # ... (remains same)
        if token.type == 'INTEGER': return Integer(int(token.value), lineno=token.lineno)
        if token.type == 'FLOAT': return Float(float(token.value), lineno=token.lineno)
        if token.type == 'STRING': return String(token.value, lineno=token.lineno) 
        if token.type == 'CHAR': return Char(token.value, lineno=token.lineno)    
        if token.type == 'TRUE': return Boolean(True, lineno=token.lineno)
        if token.type == 'FALSE': return Boolean(False, lineno=token.lineno)
        raise ValueError(f"Internal error: create_literal_node called with non-literal token {token.type}")

    def parse_declaration(self) -> Optional[Node]: # ... (remains same)
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
            if not value: return None
        if is_const and value is None:
            self.error_handler.add_syntax_error(f"Constant '{name_token.value}' must be initialized.", name_token.lineno); return None
        if not is_const and type_spec is None and value is None:
            self.error_handler.add_syntax_error(f"Variable '{name_token.value}' must have an explicit type or an initial value.", name_token.lineno); return None
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
        if not name_token:
            self._synchronize_past_block() # Try to skip what might have been the function
            return None

        if not self.consume('LPAREN', "Expected '(' after function name"):
            self._synchronize_past_block()
            return None

        params: List[Parameter] = []
        if not self.match('RPAREN'): 
            while True: 
                param_name_token = self.consume('ID', "Expected parameter name")
                if not param_name_token:
                    self._synchronize_past_block() # Error in param name
                    return None 
                
                param_type_token = self.consume_type("Expected parameter type")
                if not param_type_token:
                    self._synchronize_past_block() # Error in param type
                    return None
                
                params.append(Parameter(param_name_token.value, param_type_token.value, lineno=param_name_token.lineno))
                
                if not self.match('COMMA'): break 
            if not self.consume('RPAREN', "Expected ')' or ',' after parameter"):
                 self._synchronize_past_block()
                 return None
        
        return_gox_type_token = self.consume_type("Expected return type for function (e.g., int, float, bool, string, char, void)")
        if not return_gox_type_token:
            self._synchronize_past_block()
            return None
        
        if not (self.current_token and self.current_token.type == 'LBRACE'):
            self.error_handler.add_syntax_error("Expected '{' to start function body.", 
                                                self.current_token.lineno if self.current_token else (return_gox_type_token.lineno if return_gox_type_token else name_token.lineno))
            self._synchronize_past_block() 
            # Return a FunctionDecl node with empty body to represent the parsed signature
            return FunctionDecl(name_token.value, params, return_gox_type_token.value, [], lineno=func_token.lineno)

        body_statements = self.parse_statements_block() 
        return FunctionDecl(name_token.value, params, return_gox_type_token.value, body_statements, lineno=func_token.lineno)

    def parse_import(self) -> Optional[Node]: # ... (remains same)
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
    def parse_print(self) -> Optional[Node]: # ... (remains same)
        print_token = self.match('PRINT')
        if not print_token: return None
        expr = self.parse_expression()
        if not expr:
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                 self.error_handler.add_syntax_error("Expected expression for print statement", print_token.lineno)
            return None
        if not self.consume('SEMI', "Expected ';' after print statement"): return None
        return Print(expr, lineno=print_token.lineno)
    def parse_if(self) -> Optional[Node]: # ... (remains same)
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
    def parse_while(self) -> Optional[Node]: # ... (remains same)
        while_token = self.match('WHILE')
        if not while_token: return None
        condition = self.parse_expression()
        if not condition:
            if not self.error_handler.has_errors_since(self.error_handler.get_error_count() -1 if self.error_handler.get_error_count() >0 else 0):
                self.error_handler.add_syntax_error("Expected condition expression in while statement", while_token.lineno)
            return None
        body_statements = self.parse_statements_block()
        return While(condition, body_statements, lineno=while_token.lineno)
    def parse_assignment(self) -> Optional[Node]: # ... (remains same, target parsed by parse_primary in parse_statement)
        # This is now mainly for the case where BACKTICK starts a statement that becomes an assignment
        start_lineno = self.current_token.lineno if self.current_token else -1
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
                 self.error_handler.add_syntax_error("Expected expression on RHS of assignment.", 
                                              self.current_token.lineno if self.current_token else start_lineno)
            return None
        if not self.consume('SEMI', "Expected ';' after assignment statement"): return None
        assign_lineno = target.lineno if hasattr(target, 'lineno') and target.lineno is not None else start_lineno
        return Assignment(target, expr, lineno=assign_lineno)

    def parse_return(self) -> Optional[Node]: # ... (remains same)
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

def main(): # ... (remains same)
    from Lexer import tokenize 
    from Error import ErrorHandler 
    from AST_to_JSON import ast_to_json, save_ast_to_json, pretty_print_json

    if len(sys.argv) < 2:
        print("Usage: python Parser.py <source_file> [output_json_file]")
        sys.exit(1)
    source_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else source_file.replace('.gox', '.ast.json')
    try:
        with open(source_file, 'r', encoding='utf-8') as f: code = f.read()
    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found."); sys.exit(1)
    error_handler = ErrorHandler()
    print(f"Tokenizing {source_file}...")
    tokens = tokenize(code, error_handler) 
    if error_handler.has_errors(): print("\nLexical errors found:"); error_handler.report_errors(); sys.exit(1)
    print(f"Tokenization successful ({len(tokens)} tokens).")
    print(f"\nParsing {source_file}...")
    parser = Parser(tokens, error_handler)
    ast_program_node = parser.parse() 
    if error_handler.has_errors():
        print("\nParsing errors found:"); error_handler.report_errors()
        if ast_program_node: 
            print("\nAttempting to output partial AST (due to parsing errors)...")
            try:
                ast_data_for_json = ast_to_json(ast_program_node)
                error_output_file = output_file.replace('.ast.json', '.ast_errors.json')
                save_ast_to_json(ast_data_for_json, error_output_file)
                pretty_print_json(ast_data_for_json)
            except Exception as e: print(f"Could not serialize/print partial AST due to: {e}")
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
