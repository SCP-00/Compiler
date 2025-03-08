# UTP_Parser.py

from UTP_lexer import Token, tokenize
from UTP_AST_nodes import Integer, Float, Boolean, BinOp, UnaryOp, Location, FunctionCall, Print, Assignment, If, ConstantDecl, VariableDecl, FunctionDecl, Return, While
from UTP_error_handler import ErrorHandler


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

    def expect(self, token_type, err_msg=None):
        if self.current_token and self.current_token.type == token_type:
            self.advance()
            return True
        else:
            err = err_msg or f"Expected {token_type}"
            self.error_handler.add_error(err, self.current_token.lineno if self.current_token else 0)
            return False

    def parse(self):
        statements = []
        while self.current_token:
            statements.append(self.parse_statement()) # Use parse_statement instead of directly checking token types
        return statements

    def parse_statement(self):
        if self.current_token.type in ['VAR', 'CONST']:
            return self.parse_declaration()
        elif self.current_token.type == 'PRINT':
            return self.parse_print()
        elif self.current_token.type == 'IF':
            return self.parse_if()
        elif self.current_token.type == 'WHILE':
            return self.parse_while()
        elif self.current_token.type == 'FUNC':
            return self.parse_function()
        elif self.current_token.type == 'RETURN':
            return self.parse_return() # Added return statement parsing
        elif self.current_token.type == 'ID':
            # Check if it's a function call or assignment by peeking at the next token
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'LPAREN':
                return self.parse_function_call()
            else:
                return self.parse_assignment()
        else:
            self.error_handler.add_error(f"Unexpected token type: {self.current_token.type}", self.current_token.lineno if self.current_token else 0)
            self.advance() # To avoid infinite loop on error
            return None # Or raise exception

    def parse_expression(self):
        return self.parse_logic_or()

    def parse_logic_or(self):
        node = self.parse_logic_and()
        while self.current_token and self.current_token.type == 'LOR':
            op = self.current_token.value
            self.advance()
            node = BinOp(op, node, self.parse_logic_and())
        return node

    def parse_logic_and(self):
        node = self.parse_comparison()
        while self.current_token and self.current_token.type == 'LAND':
            op = self.current_token.value
            self.advance()
            node = BinOp(op, node, self.parse_comparison())
        return node

    def parse_comparison(self):
        node = self.parse_term()
        while self.current_token and self.current_token.type in ['LT', 'GT', 'LE', 'GE', 'EQ', 'NE']:
            op = self.current_token.value
            self.advance()
            node = BinOp(op, node, self.parse_term())
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.current_token and self.current_token.type in ['PLUS', 'MINUS']:
            op = self.current_token.value
            self.advance()
            node = BinOp(op, node, self.parse_factor())
        return node

    def parse_factor(self):
        node = self.parse_unary()
        while self.current_token and self.current_token.type in ['TIMES', 'DIVIDE', 'MOD', 'INT_DIV']:
            op = self.current_token.value
            self.advance()
            node = BinOp(op, node, self.parse_unary())
        return node

    def parse_unary(self):
        if self.current_token and self.current_token.type in ['PLUS', 'MINUS', 'NOT', 'DEREF']:
            op = self.current_token.value
            self.advance()
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
            # Check if it's a function call or location is now handled in parse_statement
            return self.parse_location()
        elif token.type in ['TRUE', 'FALSE']:
            self.advance()
            return Boolean(token.value == 'true')
        elif token.type == 'FUNC': # Corrected from previous version which had FUNC here in primary
            self.error_handler.add_error("Unexpected function keyword in expression", token.lineno) # Function keyword should start a function declaration, not be in expression
            self.advance()
            return None
        else:
            self.error_handler.add_error("Invalid expression", token.lineno)
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
            consequence.append(self.parse_statement()) # Changed to parse_statement()
        self.expect('RBRACE', "Missing '}' at the end of if block")
        alternative = []
        if self.current_token and self.current_token.type == 'ELSE':
            self.advance()
            self.expect('LBRACE', "Missing '{' after else")
            while self.current_token and self.current_token.type != 'RBRACE':
                alternative.append(self.parse_statement()) # Changed to parse_statement()
            self.expect('RBRACE', "Missing '}' at the end of else block")
        return If(test, consequence, alternative)

    def parse_while(self):
        self.expect('WHILE')
        test = self.parse_expression()
        self.expect('LBRACE', "Missing '{' after while condition")
        body = []
        while self.current_token and self.current_token.type != 'RBRACE':
            body.append(self.parse_statement()) # Changed to parse_statement()
        self.expect('RBRACE', "Missing '}' at the end of while block")
        return While(test, body)

    def parse_declaration(self):
        is_const = self.current_token.type == 'CONST'
        self.advance()
        ident = self.current_token.value
        self.expect('ID')
        var_type = None # Type handling is still basic, can be enhanced
        value = None
        if self.current_token and self.current_token.type == 'ASSIGN':
            self.advance()
            value = self.parse_expression() # Parse expression for initial value
        return ConstantDecl(ident, value) if is_const else VariableDecl(ident, var_type, value)

    def parse_function_call(self):
        name = self.current_token.value # Function name is expected to be an ID
        self.expect('ID', "Expected function name") # Expect ID for function name
        self.expect('LPAREN', "Expected '(' after function name")
        args = []
        if self.current_token.type != 'RPAREN': # Check if argument list is not empty
            args.append(self.parse_expression())
            while self.current_token.type == 'COMMA':
                self.advance()
                args.append(self.parse_expression())
        self.expect('RPAREN', "Expected ')' after function arguments")
        self.expect('SEMI', "Missing ';' after function call") # Function call is a statement
        return FunctionCall(name, args)

    def parse_function(self):
        self.expect('FUNC')
        name = self.current_token.value
        self.expect('ID', "Expected function name after FUNC keyword")
        self.expect('LPAREN', "Expected '(' after function name in declaration")
        params = [] # Parameter parsing is basic for now, can be enhanced
        if self.current_token.type != 'RPAREN':
            param_name = self.current_token.value
            self.expect('ID', "Expected parameter name")
            params.append(Parameter(param_name, None)) # No type for params for now
            while self.current_token.type == 'COMMA':
                self.advance()
                param_name = self.current_token.value
                self.expect('ID', "Expected parameter name")
                params.append(Parameter(param_name, None))
        self.expect('RPAREN', "Expected ')' after parameters in function declaration")
        return_type = None # No return type for now
        self.expect('LBRACE', "Expected '{' to start function body")
        body = []
        while self.current_token and self.current_token.type != 'RBRACE':
            body.append(self.parse_statement()) # Use parse_statement for function body
        self.expect('RBRACE', "Expected '}' to end function body")
        return FunctionDecl(name, params, return_type, body)

    def parse_return(self):
        self.expect('RETURN')
        expr = self.parse_expression() # Expression to return
        self.expect('SEMI', "Missing ';' after return statement")
        return Return(expr)


from UTP_AST_nodes import Parameter # Import Parameter class here

if __name__ == "__main__":
    # Usage example
    error_handler = ErrorHandler()
    code = """
    var x = 2 + 3 * 4;
    print x;
    func myFunction(a, b) {
        var y = a + b;
        print y;
        return y;
    }
    myFunction(x, 5);
    """
    tokens = tokenize(code)
    parser = Parser(tokens, error_handler)
    ast = parser.parse()

    if error_handler.has_errors():
        error_handler.report_errors()
    else:
        for node in ast:
            print(node)