## Parser.py - Analizador Sintáctico (Parser)

Este módulo implementa el analizador sintáctico (parser) que convierte la secuencia de tokens generada por el lexer en un Árbol de Sintaxis Abstracta (AST).

### Clases

*   `Parser`:
    *   `__init__(self, tokens)`: Inicializa el parser con la lista de tokens.
    *   `parse()`:  Punto de entrada para el análisis sintáctico. Retorna el nodo raíz del AST (`Program`).
    *   `parse_program()`:  Analiza un programa.
    *   `parse_statement()`: Analiza una sentencia.
    *   `parse_declaration()`: Analiza una declaración (variable, constante, función).
    *   `parse_expression()`: Analiza una expresión.
    *   `parse_primary()`: Analiza una expresión primaria (literal, identificador, llamada a función).
    *   `expect(self, token_type, message)`:  Verifica que el token actual sea del tipo esperado.
    *   `advance()`: Avanza al siguiente token.
    *   `parse_import()`: Analiza una declaración de importación.
