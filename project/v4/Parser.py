# Parser.py - GoxLang Parser Implementation

try:
    from Lexer import *
    from Nodes_AST import *
    from Error import ErrorHandler
    from AST_to_JSON import save_ast_to_json  # Importa la función de serialización
except ImportError as e:
    print(f"ImportError: {e}")
    exit(1)


class Parser:
    def __init__(self, tokens, error_handler):
        self.tokens = tokens
        self.error_handler = error_handler
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def peek(self, offset=1):
        """Look ahead at the next token without consuming it"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def expect(self, token_type, err_msg=None):
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            err = err_msg or f"Expected {token_type}"
            if self.current_token:
                self.error_handler.add_error(err, self.current_token.lineno)
            else:
                self.error_handler.add_error(err, "End of file")
            return False

    def parse(self):
        """Entry point for parsing the entire program"""
        statements = []
        while self.current_token:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            else:
                # Skip invalid token to avoid infinite loop.
                self.advance()

        # Wrap statements in a Program node to represent the full AST
        if self.current_token is None:
            return Program(statements)
        else:
            self.error_handler.add_error("Unexpected end of file", self.current_token.lineno)
            return None

    def parse_statement(self):
        """Parse a single statement"""
        if not self.current_token:
            return None

        # Añadir el caso para importar módulos
        if self.current_token.type == 'IMPORT':
            return self.parse_import()
        elif self.current_token.type in ['VAR', 'CONST']:
            stmt = self.parse_declaration()
            # Si la gramática requiere ';' al final de la declaración
            self.expect('SEMI', "Missing ';' after declaration")
            return stmt
        elif self.current_token.type == 'PRINT':
            return self.parse_print()
        elif self.current_token.type == 'IF':
            return self.parse_if()
        elif self.current_token.type == 'WHILE':
            return self.parse_while()
        elif self.current_token.type == 'FUNC':
            return self.parse_function()
        elif self.current_token.type == 'RETURN':
            return self.parse_return()
        elif self.current_token.type == 'BREAK':
            self.advance()
            self.expect('SEMI', "Missing ';' after break statement")
            return Break()
        elif self.current_token.type == 'CONTINUE':
            self.advance()
            self.expect('SEMI', "Missing ';' after continue statement")
            return Continue()
        elif self.current_token.type == 'ID':
            # Chequear si es llamada a función o asignación
            if self.peek() and self.peek().type == 'LPAREN':
                return self.parse_function_call()
            else:
                return self.parse_assignment()
        else:
            self.error_handler.add_error(
                f"Unexpected token type: {self.current_token.type}", 
                self.current_token.lineno
            )
            self.advance()
            return None
        
    def parse_expression(self):
                return self.parse_logic_and()

    def parse_logic_and(self):
        node = self.parse_comparison()
        while self.current_token and self.current_token.type == 'LAND':
            op = self.current_token.value
            self.advance()
            right = self.parse_comparison()
            node = BinOp(op, node, right)
        return node

    def parse_comparison(self):
        node = self.parse_term()
        while self.current_token and self.current_token.type in ['LT', 'GT', 'LE', 'GE', 'EQ', 'NE']:
            op = self.current_token.value
            self.advance()
            right = self.parse_term()
            node = BinOp(op, node, right)
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.current_token and self.current_token.type in ['PLUS', 'MINUS']:
            op = self.current_token.value
            self.advance()
            right = self.parse_factor()
            node = BinOp(op, node, right)
        return node

    def parse_factor(self):
        node = self.parse_unary()
        while self.current_token and self.current_token.type in ['TIMES', 'DIVIDE', 'MOD', 'INT_DIV']:
            op = self.current_token.value
            self.advance()
            right = self.parse_unary()
            node = BinOp(op, node, right)
        return node

    def parse_unary(self):
        if self.current_token and self.current_token.type in ['PLUS', 'MINUS', 'NOT', 'DEREF']:
            op = self.current_token.value
            self.advance()
            if self.current_token.type == 'DEREF':
                return Dereference(self.parse_primary())
            return UnaryOp(op, self.parse_primary())
        return self.parse_primary()

    def parse_primary(self):
        token = self.current_token
        if token.type == 'INTEGER':
            self.advance()
            return Integer(int(token.value))
        elif token.type == 'FLOAT':
            self.advance()
            return Float(float(token.value))
        elif token.type == 'LPAREN':
            self.advance()
            expr = self.parse_expression()
            self.expect('RPAREN', "Missing closing parenthesis")
            return expr
        elif token.type == 'ID':
            self.advance()
            node = Location(token.value)
            if self.current_token and self.current_token.type == 'LPAREN':
                # Llamada a función
                self.advance()  # Consume LPAREN
                args = []
                if self.current_token.type != 'RPAREN':
                    args.append(self.parse_expression())
                    while self.current_token and self.current_token.type == 'COMMA':
                        self.advance()
                        args.append(self.parse_expression())
                self.expect('RPAREN', "Expected ')' after function arguments")
                node = FunctionCall(token.value, args)
            return node
        elif token.type in ['TRUE', 'FALSE']:
            self.advance()
            return Boolean(token.value.lower() == 'true')
        elif token.type == 'STRING':
            self.advance()
            return String(token.value)
        elif token.type == 'CHAR':
            self.advance()
            return Char(token.value)
        else:
            self.error_handler.add_error("Invalid expression", token.lineno)
            self.advance()
            return None

    def parse_location(self):
        ident = self.current_token.value
        self.expect('ID')
        return Location(ident)

    def parse_print(self):
        self.expect('PRINT')
        expr = self.parse_expression()
        self.expect('SEMI', "Missing ';' after print statement")
        return Print(expr)

    def parse_assignment(self):
        location = self.parse_location()
        self.expect('ASSIGN', "Missing '=' in assignment")
        expr = self.parse_expression()
        self.expect('SEMI', "Missing ';' after assignment")
        return Assignment(location, expr)

    def parse_if(self):
        self.expect('IF')
        test = self.parse_expression()
        self.expect('LBRACE', "Missing '{' after if condition")
        consequence = []
        while self.current_token and self.current_token.type != 'RBRACE':
            consequence.append(self.parse_statement())
        self.expect('RBRACE', "Missing '}' at the end of if block")
        
        alternative = []
        if self.current_token and self.current_token.type == 'ELSE':
            self.advance()
            self.expect('LBRACE', "Missing '{' after else")
            while self.current_token and self.current_token.type != 'RBRACE':
                alternative.append(self.parse_statement())
            self.expect('RBRACE', "Missing '}' at the end of else block")
        return If(test, consequence, alternative)

    def parse_while(self):
        self.expect('WHILE')
        test = self.parse_expression()
        self.expect('LBRACE', "Missing '{' after while condition")
        body = []
        while self.current_token and self.current_token.type != 'RBRACE':
            body.append(self.parse_statement())
        self.expect('RBRACE', "Missing '}' at the end of while block")
        return While(test, body)
    
    def parse_declaration(self):
        is_const = self.current_token.type == 'CONST'
        self.advance()
        ident = self.expect('ID', "Expected identifier in declaration").value
        
        # Add proper type handling
        var_type = None
        if self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'ID']:
            var_type = self.current_token.value
            self.advance()
        
        self.expect('ASSIGN', "Expected '=' in declaration")
        value = self.parse_expression()
        
        # Expect semicolon after declaration
        self.expect('SEMI', "Missing ';' after declaration")
        
        if is_const:
            return ConstantDecl(ident, value)
        else:
            return VariableDecl(ident, var_type, value)

    def parse_function_call(self):
        name = self.current_token.value  # Se asume que es una llamada a función
        self.expect('ID', "Expected function name")
        self.expect('LPAREN', "Expected '(' after function name")
        args = []
        if self.current_token.type != 'RPAREN':
            args.append(self.parse_expression())
            while self.current_token and self.current_token.type == 'COMMA':
                self.advance()
                args.append(self.parse_expression())
        self.expect('RPAREN', "Expected ')' after function arguments")
        self.expect('SEMI', "Missing ';' after function call")
        return FunctionCall(name, args)

    def parse_function(self):
        self.expect('FUNC')
        name = self.expect('ID', "Expected function name after FUNC keyword").value
        self.expect('LPAREN', "Expected '(' after function name in declaration")
        params = []
        if self.current_token.type != 'RPAREN':
            param_name = self.expect('ID', "Expected parameter name").value
            
            # Handle parameter type
            param_type = None
            if self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'ID']:
                param_type = self.current_token.value
                self.advance()
            
            params.append(Parameter(param_name, param_type))
            while self.current_token and self.current_token.type == 'COMMA':
                self.advance()
                param_name = self.expect('ID', "Expected parameter name").value
                
                # Handle parameter type
                param_type = None
                if self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'ID']:
                    param_type = self.current_token.value
                    self.advance()
                
                params.append(Parameter(param_name, param_type))
        self.expect('RPAREN', "Expected ')' after parameters in function declaration")
        
        # Handle return type
        return_type = None
        if self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'ID']:
            return_type = self.current_token.value
            self.advance()
        
        self.expect('LBRACE', "Expected '{' to start function body")
        body = []
        while self.current_token and self.current_token.type != 'RBRACE':
            body.append(self.parse_statement())
        self.expect('RBRACE', "Expected '}' to end function body")
        return FunctionDecl(name, params, return_type, body)

    def parse_return(self):
        self.expect('RETURN')
        expr = self.parse_expression()
        self.expect('SEMI', "Missing ';' after return statement")
        return Return(expr)

    def parse_import(self):
        """Parse an import declaration"""
        self.expect('IMPORT')
        
        # Check if it's a function import
        is_func_import = False
        if self.current_token and self.current_token.type == 'FUNC':
            is_func_import = True
            self.advance()
        
        module_name = self.expect('ID', "Expected module name after IMPORT").value
        
        # If it's a function import, parse the signature
        if is_func_import:
            self.expect('LPAREN', "Expected '(' after function name in import")
            params = []
            
            # Parse parameters
            while self.current_token and self.current_token.type != 'RPAREN':
                param_name = self.expect('ID', "Expected parameter name").value
                
                # Handle parameter type
                param_type = None
                if self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'ID']:
                    param_type = self.current_token.value
                    self.advance()
                
                params.append(Parameter(param_name, param_type))
                
                if self.current_token.type == 'COMMA':
                    self.advance()
            
            self.expect('RPAREN', "Expected ')' after parameters in import")
            
            # Handle return type
            return_type = None
            if self.current_token and self.current_token.type in ['INT', 'FLOAT_TYPE', 'BOOL', 'STRING_TYPE', 'CHAR_TYPE', 'ID']:
                return_type = self.current_token.value
                self.advance()
            
            self.expect('SEMI', "Missing ';' after import declaration")
            return FunctionImportDecl(module_name, params, return_type)
        else:
            self.expect('SEMI', "Missing ';' after import declaration")
            return ImportDecl(module_name)