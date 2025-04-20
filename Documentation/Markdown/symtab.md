## Sym_tab.py - Tabla de Símbolos

Este módulo define la clase `SymbolTab` que implementa una tabla de símbolos jerárquica para gestionar ámbitos.

### Clases

*   `SymbolTab`:
    *   `__init__(self, name, parent=None)`: Inicializa la tabla de símbolos con un nombre y un padre (opcional).
    *   `add(self, name, value)`: Añade un símbolo a la tabla. Lanza excepciones si el símbolo ya está definido o si hay un conflicto de tipos.
    *   `get(self, name)`: Busca un símbolo en la tabla (primero en el ámbito actual, luego recursivamente en los ámbitos padres).
    *   `print(self)`: Imprime el contenido de la tabla de símbolos.
    *   `__contains__(self, key)`: Permite usar el operador `in` para verificar si un símbolo está definido.

### Clases de Excepción

*   `symbolDefinedError`:  Indica un error de redeclaración de un símbolo.
*   `SymbolConflictError`: Indica un error de conflicto de tipos en la redeclaración de un símbolo.