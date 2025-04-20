## AST_to_JSON.py - Serialización del AST a JSON

Este módulo proporciona una función para convertir el Árbol de Sintaxis Abstracta (AST) a formato JSON.

### Funciones

*   `save_ast_to_json(ast_node, filename)`: Toma un nodo del AST y un nombre de archivo, y guarda la representación JSON del AST en el archivo especificado.  Utiliza la función `convert_to_serializable` para convertir los nodos del AST a un formato serializable.
*   `convert_to_serializable(obj)`:  Convierte objetos Python, incluyendo instancias de clases AST, a diccionarios que pueden ser serializados a JSON.  Maneja diferentes tipos de datos y estructuras recursivamente.