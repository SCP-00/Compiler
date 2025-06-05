# Lexer.py

import re
from typing import List, Tuple, Optional # Added Optional
from Error import ErrorHandler # Assuming Error.py is in the same directory or accessible

# Token types - ORDER IS CRUCIAL for correct matching!
TOKENS = {
    # 1. Whitespace and Comments (ignore these first)
    'SKIP': r'[ \t\n]+',
    'COMMENT': r'(?:/\*[\s\S]*?\*/|//[^\n]*)',
    
    # 2. Literals (FLOAT must come before INTEGER for longest match)
    'FLOAT': r'-?\d*\.\d+|\d+\.|\.\d+', # Allows .5, 1., 1.0, -2.0, etc.
    'INTEGER': r'\d+',
    'STRING': r'"[^"\\]*(?:\\.[^"\\]*)*"',
    'CHAR': r'\'(?:[^\'\\]|\\[\'\"\\nrt]|\\x[0-9a-fA-F]{2})\'', # Hex escapes like '\xff'
    'TRUE': r'true',
    'FALSE': r'false',
    
    # 3. Keywords (specific identifiers)
    'IMPORT': r'import',
    'FUNC': r'func',
    'VAR': r'var',
    'CONST': r'const',
    'IF': r'if',
    'ELSE': r'else',
    'WHILE': r'while',
    'RETURN': r'return',
    'BREAK': r'break',
    'CONTINUE': r'continue',
    'PRINT': r'print',
    
    # 4. Data Types (keywords for types)
    'INT': r'int',
    'FLOAT_TYPE': r'float', # Distinguish from FLOAT literal
    'BOOL': r'bool',
    'STRING_TYPE': r'string', # Distinguish from STRING literal
    'CHAR_TYPE': r'char',   # Distinguish from CHAR literal
    
    # 5. Operators (longer ones first to avoid partial matches, e.g., == before =)
    'EQ': r'==',
    'NE': r'!=',
    'LE': r'<=',
    'GE': r'>=',
    'LAND': r'&&',
    'LOR': r'\|\|',
    # Single character operators
    'PLUS': r'\+',
    'MINUS': r'-',
    'TIMES': r'\*',
    'DIVIDE': r'/',
    'MOD': r'%',
    # 'INT_DIV': r'//', # Assuming GoX uses / for float division if operands are float, and integer division if int
    'ASSIGN': r'=',
    'LT': r'<',
    'GT': r'>',
    'NOT': r'!',
    'MEM_ALLOC': r'\^',  # Changed from BITWISE_COMPLEMENT, used for memory allocation like ^1000
    
    # 6. Memory operations
    'BACKTICK': r'`',  # For memory access `addr
    
    # 7. Delimiters
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'LBRACE': r'\{',
    'RBRACE': r'\}',
    'SEMI': r';',
    'COMMA': r',',
    
    # 8. Identifiers (general names, should be last among keywords/types)
    'ID': r'[a-zA-Z_][a-zA-Z0-9_]*'
}

# Create combined regex pattern
# The order in TOKENS dict matters for how re.match with | will try to match.
# Python 3.7+ dicts preserve insertion order.
token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKENS.items())

class Token:
    """Represents a token in the source code."""
    def __init__(self, type: str, value: str, lineno: int, column: int = 0):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __repr__(self) -> str:
        # Escape newlines in value for cleaner representation
        value_repr = self.value.replace('\n', '\\n')
        return f"Token(type='{self.type}', value='{value_repr}', lineno={self.lineno}, column={self.column})"

def handle_escape_sequences(value: str) -> str:
    """Handle escape sequences in string and character literals."""
    # This function is applied to the *content* of the char/string, not the whole token value
    
    # First, handle hexadecimal escapes like \xHH
    def replace_hex(match):
        hex_val = match.group(1)
        try:
            return chr(int(hex_val, 16))
        except ValueError:
            # If \x is followed by invalid hex, treat it literally (or raise error)
            # For now, let's keep it as is, error handling can be more strict later
            return f'\\x{hex_val}' 
            
    value = re.sub(r'\\x([0-9a-fA-F]{2})', replace_hex, value)

    # Then, handle common named escapes
    # Important: \\ must be replaced first to avoid issues with other sequences
    escape_map = {
        '\\\\': '\\',
        '\\n': '\n',
        '\\r': '\r',
        '\\t': '\t',
        '\\"': '"',
        '\\\'': '\''
    }    
    for escape_seq, char_val in escape_map.items():
        value = value.replace(escape_seq, char_val)
        
    return value

def tokenize(text: str, error_handler: ErrorHandler) -> List[Token]:
    """Tokenize the input text into a list of tokens."""
    tokens: List[Token] = []
    lineno = 1
    line_start_pos = 0 # To calculate column number
    pos = 0
    
    while pos < len(text):
        match = re.match(token_regex, text[pos:])
        if match:
            kind = match.lastgroup
            value = match.group(kind)
            column = pos - line_start_pos + 1
            
            if kind in ['SKIP', 'COMMENT']:
                # Update lineno and line_start_pos for SKIP/COMMENT tokens
                newlines_in_value = value.count('\n')
                if newlines_in_value > 0:
                    lineno += newlines_in_value
                    line_start_pos = pos + value.rfind('\n') + 1
            else:
                # Handle special processing for string and char literal contents
                processed_value = value
                if kind == 'STRING':
                    # Remove quotes and process escapes from the content
                    processed_value = handle_escape_sequences(value[1:-1])
                elif kind == 'CHAR':
                    # Remove quotes and process escapes from the content
                    processed_value = handle_escape_sequences(value[1:-1])
                    # Ensure char literal is a single character after escape processing
                    if len(processed_value) != 1:
                        error_handler.add_lexical_error(
                            f"Character literal must be a single character, got '{processed_value}' after escapes", 
                            lineno, column
                        )
                
                tokens.append(Token(kind, processed_value, lineno, column))
            
            pos += len(value) # Move past the matched token
        else:
            # If no token matches, it's an illegal character
            column = pos - line_start_pos + 1
            if not text[pos].isspace(): # Report error only if not whitespace
                error_handler.add_lexical_error(f"Illegal character '{text[pos]}'", lineno, column)
            
            # If the illegal char is a newline, update lineno
            if text[pos] == '\n':
                lineno += 1
                line_start_pos = pos + 1
            pos += 1 # Move to the next character

    # Add an EOF token for easier parsing (optional, but can be helpful)
    # tokens.append(Token('EOF', '', lineno, pos - line_start_pos + 1))
    return tokens


def main():
    """Main function for direct execution of the lexer."""
    import sys
    import os
    # from Error import ErrorHandler # Already imported if running standalone

    if len(sys.argv) != 2:
        print("Usage: python Lexer.py <file.gox>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if not file_path.endswith('.gox'):
        print(f"Error: File must have .gox extension: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)

    error_handler = ErrorHandler()
    tokens = tokenize(content, error_handler)

    if error_handler.has_errors():
        print("\nLexical Errors:")
        # Assuming ErrorHandler has a method to print errors nicely
        # or get_errors() returns a string.
        error_handler.report_errors()
        sys.exit(1)

    print("\nTokens:")
    for token in tokens:
        print(token) # Uses Token.__repr__

if __name__ == "__main__":
    main()