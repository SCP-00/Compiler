## Interpreter.py - Intérprete del Lenguaje

Este módulo contiene la implementación del intérprete para el lenguaje.  Recorre el AST y ejecuta el código.

### Clases

*   `Interpreter`: Clase principal del intérprete.
    *   `__init__(self)`: Inicializa el entorno global, la memoria y la tabla de símbolos.
    *   `visit_Program(self, node)`:  Punto de entrada para la ejecución del programa.
    *   `visit_*`: Métodos `visit_*` para cada tipo de nodo AST (e.g., `visit_VariableDecl`, `visit_Print`, `visit_BinOp`).  Estos métodos contienen la lógica de ejecución para cada construcción del lenguaje.
    *   `_read_mem(self, addr)`: Lee un valor de la memoria.
    *   `_write_mem(self, addr, val)`: Escribe un valor en la memoria.
*   `ReturnException`:  Excepción utilizada para implementar la sentencia `return`.

### Funciones auxiliares

*   `_eval_binop(op, a, b)`:  Evalúa operaciones binarias.
*   `_eval_unary(op, a)`: Evalúa operaciones unarias.