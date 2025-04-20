## Semantic_cheking.py - Analizador Semántico

Este módulo implementa el analizador semántico que realiza verificaciones de tipo y otras validaciones sobre el AST.

### Clases

*   `Checker`:
    *   `__init__(self)`: Inicializa la tabla de símbolos, el manejador de errores, la pila de tipos de retorno y el nivel de bucle.
    *   `check(self, prog)`:  Punto de entrada para el análisis semántico.
    *   `visit_*`: Métodos `visit_*` para cada tipo de nodo AST (e.g., `visit_VariableDecl`, `visit_Print`, `visit_BinOp`).  Estos métodos contienen la lógica de verificación semántica para cada construcción del lenguaje.
    *   `generic_visit(self, node, env)`: Método de respaldo para nodos no visitados explícitamente.

### Lógica de Verificación Semántica

*   **Declaraciones:** Verifica redeclaraciones, compatibilidad de tipos en inicializaciones.
*   **Asignaciones:** Verifica que la variable esté declarada y que el tipo de la expresión asignada coincida con el tipo de la variable.
*   **Expresiones:**  Verifica la validez de operaciones binarias y unarias, conversiones de tipo.
*   **Sentencias de Control de Flujo:**  Verifica que las condiciones de `if` y `while` sean de tipo `bool`, que `break` y `continue` estén dentro de bucles.
*   **Llamadas a Funciones:**  Verifica que la función esté declarada, que el número de argumentos coincida con el número de parámetros, y que los tipos de los argumentos coincidan con los tipos de los parámetros.
*   **Retornos:** Verifica que el tipo de la expresión retornada coincida con el tipo de retorno declarado de la función.