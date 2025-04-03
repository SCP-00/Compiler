import re

# Definición de patrones de tokens con un orden adecuado:
TOKEN_PATTERNS = [
    # Palabras reservadas y funciones de E/S (colocadas primero)
    ("CONST", r"\bconst\b"),
    ("VAR", r"\bvar\b"),
    ("INPUT", r"\binput\b"),
    ("PRINT", r"\bprint\b"),
    ("RETURN", r"\breturn\b"),
    ("BREAK", r"\bbreak\b"),
    ("CONTINUE", r"\bcontinue\b"),
    ("IF", r"\bif\b"),
    ("ELSE", r"\belse\b"),
    ("WHILE", r"\bwhile\b"),
    ("FUNC", r"\bfunc\b"),
    ("IMPORT", r"\bimport\b"),
    ("TRUE", r"\btrue\b"),
    ("FALSE", r"\bfalse\b"),
    
    # Identificadores (se coloca después de las palabras reservadas)
    ("ID", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    
    # Literales numéricos y de caracteres
    ("FLOAT", r"\b\d+\.\d*|\.\d+\b"),  # Se coloca antes de INTEGER
    ("INTEGER", r"\b\d+\b"),
    ("CHAR", r"'([^'\\]|\\[n\\x\\']|\\x[0-9A-Fa-f]{2})'"),
    
    # Operadores de dos caracteres (deben ir antes de los de un solo carácter)
    ("LE", r"<="),
    ("GE", r">="),
    ("EQ", r"=="),
    ("NE", r"!="),
    ("LAND", r"&&"),
    ("LOR", r"\|\|"),
    
    # Operadores de un solo carácter
    ("LT", r"<"),
    ("GT", r">"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("TIMES", r"\*"),
    ("DIVIDE", r"/"),
    ("GROW", r"\^"),
    ("ASSIGN", r"="),
    
    # Otros símbolos
    ("SEMI", r";"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("COMMA", r","),
    ("DEREF", r"`"),
    
    # Comentarios y espacios en blanco (se ignoran)
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*.*?\*/"),
    ("WHITESPACE", r"\s+"),
]


# Compilamos cada patrón solo una vez para mayor eficiencia.
COMPILED_PATTERNS = [(token_type, re.compile(pattern)) for token_type, pattern in TOKEN_PATTERNS]

def lexer(input_code):
    """
    Analiza el código fuente y retorna una lista de tokens.
    
    Parámetros:
        input_code (str): Cadena con el código a analizar.
        
    Retorna:
        List[Tuple[str, str]]: Lista de tuplas donde cada tupla es (tipo_token, lexema).
    """
    tokens = []  # Lista para almacenar los tokens encontrados.
    position = 0  # Posición actual en el código.
    
    while position < len(input_code):
        match = None
        
        # Se recorre cada patrón para ver cuál coincide en la posición actual.
        for token_type, regex in COMPILED_PATTERNS:
            match = regex.match(input_code, position)
            
            if match:
                lexeme = match.group(0)
                # Ignoramos espacios y comentarios.
                if token_type not in ("WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"):
                    tokens.append((token_type, lexeme))
                position = match.end()  # Actualizamos la posición
                break  # Salimos del ciclo for al encontrar el primer match.
        
        # Si no se encontró coincidencia, se lanza un error.
        if not match:
            raise ValueError(f"Token no reconocido en la posición {position}: {input_code[position]}")
    
    return tokens

# Ejemplo de uso
if __name__ == "__main__":
    input_code = """
    var x = 10;
    if (x >= 5) {
        print(x);
    }
    """
    
    try:
        tokens = lexer(input_code)
        print("Tokens generados:")
        for token in tokens:
            print(token)
    except ValueError as e:
        print(e)
