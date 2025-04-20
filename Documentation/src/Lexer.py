# Lexer.py
import re

# Definición de tokens con orden adecuado:
TOKEN_SPEC = [
    # Palabras reservadas
    ('CONST', r'\bconst\b'),
    ('VAR', r'\bvar\b'),
    ('PRINT', r'\bprint\b'),
    ('RETURN', r'\breturn\b'),
    ('BREAK', r'\bbreak\b'),
    ('CONTINUE', r'\bcontinue\b'),
    ('IF', r'\bif\b'),
    ('ELSE', r'\belse\b'),
    ('WHILE', r'\bwhile\b'),
    ('FUNC', r'\bfunc\b'),
    ('IMPORT', r'\bimport\b'),
    ('TRUE', r'\btrue\b'),
    ('FALSE', r'\bfalse\b'),
    ('INPUT', r'\binput\b'),

    # Data types
    ('INT', r'\bint\b'),
    ('FLOAT_TYPE', r'\bfloat\b'),
    ('BOOL', r'\bbool\b'),
    ('STRING_TYPE', r'\bstring\b'),
    ('CHAR_TYPE', r'\bchar\b'),

    # Numeric Literals
    ('HEX', r'0[xX][0-9a-fA-F]+'),
    ('BINARY', r'0[bB][01]+'),
    ('FLOAT', r'\d+\.\d*|\.\d+'),
    ('INTEGER', r'\d+'),

    ('CHAR', r"'([^\\]|\\.)'"),
    ('STRING', r'"([^\\"]|\\.)*"'),

    # Identificadores
    ('ID', r'[a-zA-Z_][a-zA-Z_0-9]*'),

    # Operadores compuestos
    ('INT_DIV', r'(?<!/)//(?=\s*\d)'),
    ('POWER', r'\*\*'),

    # Comentarios (placeholder)
    ('COMMENT', None),
    ('LE', r'<='), ('GE', r'>='), ('EQ', r'=='), ('NE', r'!='),
    ('LAND', r'&&'), ('LOR', r'\|\|'),
    ('INC', r'\+\+'), ('DEC', r'--'),
    ('PLUS_ASSIGN', r'\+='), ('MINUS_ASSIGN', r'-='),
    ('TIMES_ASSIGN', r'\*='), ('DIV_ASSIGN', r'/='),
    ('MOD_ASSIGN', r'%='), ('POW_ASSIGN', r'\^='),

    # Operadores de un solo carácter
    ('LT', r'<'), ('GT', r'>'), ('PLUS', r'\+'), ('MINUS', r'-'),
    ('TIMES', r'\*'), ('DIVIDE', r'/'), ('MOD', r'%'), ('GROW', r'\^'), ('ASSIGN', r'='),

    # Otros símbolos
    ('SEMI', r';'), ('LPAREN', r'\('), ('RPAREN', r'\)'),
    ('LBRACE', r'\{'), ('RBRACE', r'\}'), ('LBRACKET', r'\['), ('RBRACKET', r'\]'),
    ('COMMA', r','), ('DOT', r'\.'), ('COLON', r':'), ('DEREF', r'`'),

    # Espacios en blanco
    ('WHITESPACE', r'\s+'),

    # Cualquier otro carácter
    ('MISMATCH', r'.')
]

# Crear la expresión regular combinada a partir de los tokens
token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC if pattern)

class Token:
    def __init__(self, type: str, value: str, lineno: int):
        self.type = type
        self.value = value
        self.lineno = lineno

    def __repr__(self):
        return f"Token(type='{self.type}', value='{self.value}', lineno={self.lineno})"

def tokenize(text, error_handler):
    """
    Función que recibe un código fuente y retorna una lista de tokens.
    Los tokens se generan utilizando la expresión regular compuesta.
    """
    tokens = []
    lineno = 1
    pos = 0
    while pos < len(text):
        # Comentario de una línea
        if text[pos:pos+2] == '//':
            end = text.find('\n', pos)
            if end == -1:
                end = len(text)
            value = text[pos:end]
            lineno += value.count('\n')
            pos = end
            continue

        # Comentario de bloque (anidado)
        if text[pos:pos+2] == '/*':
            depth = 1
            i = pos + 2
            while i < len(text) and depth > 0:
                if text[i:i+2] == '/*':
                    depth += 1
                    i += 2
                elif text[i:i+2] == '*/':
                    depth -= 1
                    i += 2
                else:
                    if text[i] == '\n':
                        lineno += 1
                    i += 1
            if depth > 0:
                error_handler.add_error("Unterminated block comment", lineno)
                pos = i
            else:
                pos = i
            continue

        # Otros tokens
        match = re.match(token_regex, text[pos:], re.DOTALL)
        if not match:
            error_handler.add_error(f"Caracter ilegal '{text[pos]}'", lineno)
            pos += 1
            continue
        kind = match.lastgroup
        value = match.group()
        if kind == 'WHITESPACE':
            lineno += value.count('\n')
            pos += len(value)
            continue
        elif kind == 'MISMATCH':
            error_handler.add_error(f"Caracter ilegal '{value}'", lineno)
            pos += len(value)
            continue
        tokens.append(Token(kind, value, lineno))
        pos += len(value)
    return tokens

# Ejemplo de uso del lexer
if __name__ == "__main__":
    from Error import ErrorHandler
    test_code = """
/* ******************************************************************* *
 *                                                                     *
 * factorize.gox  (compilador gox)                                     *
 *                                                                     *
 * Dado un numero N, lo descompone en sus factores primos.             *
 * Ejemplo: 21 = 3x7                                                   *
 *                                                                     *
 ********************************************************************* *
 */

func mod(x int, y int) int {
	return x - (x/y) * y;
}

func isprime(n int) bool {
    if n < 2 {
        return false;
    }
    var i int = 2;
    while i * i <= n {
        if mod(n, i) == 0 {
            return false;
        }
        i = i + 1;
    }
    return true;
}

func factorize(n int) int {
    var factor int = 2;
    // print "factores primos de " + n + ": ";

    while n > 1 {
        while mod(n, factor) == 0 {
            print factor;
            n = n / factor;
        }
        factor = factor + 1;
    }
}

var num int = 21;
print factorize(num);

"""
    error_handler = ErrorHandler()
    tokens = tokenize(test_code, error_handler)
    for tok in tokens:
        print(tok)
    if error_handler.has_errors():
        print("--- Errors found ---")
        error_handler.report_errors()