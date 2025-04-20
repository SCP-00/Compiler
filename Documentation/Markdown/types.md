## Types.py - Sistema de Tipos

Este módulo define el sistema de tipos del lenguaje y funciones para la verificación de tipos.

### Constantes

*   `typenames`:  Conjunto de nombres de tipos válidos (int, float, bool, string, char).

### Funciones

*   `check_binop(op, type1, type2)`:  Verifica la validez de una operación binaria entre dos tipos.  Retorna el tipo resultante o `None` si la operación no es válida.
*   `check_unaryop(op, type)`: Verifica la validez de una operación unaria sobre un tipo. Retorna el tipo resultante o `None` si la operación no es válida.