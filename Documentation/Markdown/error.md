## Error.py - Manejo de Errores

Este módulo define un sistema para la gestión y reporte de errores durante las fases de compilación (léxica, sintáctica, semántica, interpretación).

### Clases

*   `ErrorType`: Define constantes para los tipos de errores (LEXICAL, SYNTAX, SEMANTIC, INTERPRETATION, GENERAL).
    *   `normalize(cls, t)`: Normaliza un tipo de error a una de las constantes válidas.
*   `ErrorEntry`: Representa una entrada de error individual, con mensaje, línea, columna y tipo.
*   `CompilerError`: Clase base para excepciones específicas del compilador.
*   `LexicalError`, `SyntaxError`, `SemanticError`, `InterpretationError`: Subclases de `CompilerError` para tipos de error específicos.
*   `ErrorHandler`:  Manejador centralizado de errores.
    *   `add_error(self, message, lineno=None, colno=None, error_type=ErrorType.GENERAL)`: Registra un error.
    *   `add_lexical_error`, `add_syntax_error`, `add_semantic_error`, `add_interpretation_error`: Métodos de conveniencia para registrar errores de tipo específico.
    *   `has_errors(self)`:  Indica si hay errores registrados.
    *   `report_errors(self)`: Imprime los errores registrados.
    *   `exit_if_errors(self, exit_code=1)`: Imprime los errores y sale del programa si hay errores.
    *   `clear_errors(self)`: Elimina todos los errores registrados.