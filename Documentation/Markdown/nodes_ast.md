## Nodes_AST.py - Definición de Nodos del AST

Este módulo define las clases para los nodos del Árbol de Sintaxis Abstracta (AST).

### Clases Base

*   `ASTNode`: Clase base abstracta para todos los nodos del AST.
    *   `accept(self, visitor, env)`: Método para implementar el patrón Visitor.

### Nodos de Expresión

*   `Integer`, `Float`, `String`, `Char`, `Boolean`: Representan literales de diferentes tipos.
*   `BinOp`: Representa una operación binaria (e.g., `+`, `-`, `*`, `/`).
*   `UnaryOp`: Representa una operación unaria (e.g., `-`, `!`).
*   `TypeCast`: Representa una conversión de tipo.
*   `Location`: Representa una variable o un identificador.
*   `FunctionCall`: Representa una llamada a función.

### Nodos de Declaración

*   `VariableDecl`: Representa la declaración de una variable.
*   `ConstantDecl`: Representa la declaración de una constante.
*   `FunctionDecl`: Representa la declaración de una función.
*   `Parameter`: Representa un parámetro de función.
*   `ImportDecl`: Representa una declaración de importación de módulo.
*   `FunctionImportDecl`: Representa una declaración de importación de función con firma.

### Nodos de Sentencia

*   `Assignment`: Representa una asignación.
*   `Print`: Representa una sentencia `print`.
*   `If`: Representa una sentencia `if`.
*   `While`: Representa un bucle `while`.
*   `Return`: Representa una sentencia `return`.
*   `Break`: Representa una sentencia `break`.
*   `Continue`: Representa una sentencia `continue`.
*   `Block`: Representa un bloque de sentencias.

### Otros

*   `Program`: Nodo raíz que representa un programa completo.