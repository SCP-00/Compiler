# Documentación del Compilador GoX

Este documento detalla la arquitectura y el funcionamiento de los componentes del compilador para el lenguaje GoX. El compilador sigue una estructura de pases secuenciales.

## 1. Visión General del Compilador

El compilador GoX procesa el código fuente en las siguientes etapas:

1.  **Análisis Léxico:** `Lexer.py` convierte el código fuente GoX en una secuencia de tokens.
2.  **Análisis Sintáctico:** `Parser.py` construye un Árbol de Sintaxis Abstracta (AST) a partir de los tokens. Los nodos del AST se definen en `Nodes_AST.py`.
3.  **Análisis Semántico:** `SemanticAnalyzer.py` verifica la coherencia semántica del AST, la correspondencia de tipos y la validez del uso de símbolos, utilizando `Types.py` y `SymbolTable.py`. El AST es enriquecido con información de tipos y semántica.
4.  **Generación de Código Intermedio (IR):** `IRCodeGenerator.py` traduce el AST semánticamente validado a una representación de código intermedio basada en instrucciones de pila.
5.  **Ejecución/Interpretación del IR:** `StackMachine.py` interpreta y ejecuta el código IR.

`Error.py` proporciona una infraestructura centralizada para el reporte de errores. `AST_to_JSON.py` es una utilidad para serializar el AST.

## 2. Manejo de Errores (`Error.py`)

*   **Propósito:** Proporcionar un sistema unificado para la recolección y reporte de errores en todas las fases de la compilación.
*   **Errores que Maneja:** Errores `LEXICAL`, `SYNTAX`, `SEMANTIC`, `INTERPRETATION`, y `GENERAL`.
*   **Entradas:** Mensajes de error, números de línea (`lineno`), números de columna (`colno`) y tipos de error.
*   **Salidas/Exportaciones:** La clase `ErrorHandler` almacena los errores internamente y permite consultarlos mediante métodos como `has_errors()`, `get_formatted_errors()`, `report_errors()`.
*   **Clases y Métodos Clave:**
    *   `ErrorEntry`: Representa un error individual.
    *   `ErrorType`: Define los tipos de error.
    *   `CompilerError`: Clase base para excepciones de compilador.
    *   `ErrorHandler`:
        *   `add_error()`: Registra un nuevo error.
        *   `add_lexical_error()`, `add_syntax_error()`, `add_semantic_error()`, `add_interpretation_error()`: Métodos de conveniencia para añadir errores específicos.
        *   `has_errors()`: Verifica si hay errores registrados.
        *   `report_errors()`: Imprime los errores formateados.
        *   `exit_if_errors()`: Reporta errores y termina la ejecución si existen.
        *   `get_error_count()`, `has_errors_since()`: Utlidades para seguimiento de errores.

## 3. Análisis Léxico (`Lexer.py`)

*   **Propósito:** Convertir el código fuente GoX en una secuencia de `Token`s.
*   **Errores que Maneja:** Errores léxicos, como caracteres ilegales o literales de carácter (`CHAR`) malformados (ej. `''` o `'\xZz'`).
*   **Entradas:** Una cadena de texto con el código fuente GoX y una instancia de `ErrorHandler`.
*   **Salidas/Exportaciones:** Una lista de objetos `Token` (`List[Token]`) al `Parser.py`.
*   **Clases y Funciones Clave:**
    *   `Token`: Representa un token con `type`, `value`, `lineno`, `column`.
    *   `TOKENS`: Un diccionario que define las expresiones regulares para cada tipo de token. El orden de las entradas es crítico para el emparejamiento (ej. `FLOAT` antes de `INTEGER`).
    *   `tokenize(text, error_handler)`: Función principal que realiza el análisis léxico. Utiliza `re.match` y procesa las secuencias de escape mediante `handle_escape_sequences()`.
    *   `handle_escape_sequences(value)`: Procesa escapes (`\n`, `\xHH`) dentro de literales de cadena y carácter.

## 4. Definición de Nodos del AST (`Nodes_AST.py`)

*   **Propósito:** Definir la estructura de datos del Árbol de Sintaxis Abstracta (AST) para representar la jerarquía sintáctica del código GoX.
*   **Errores que Maneja:** Este archivo solo define la estructura; no maneja errores directamente.
*   **Entradas:** Ninguna (módulo de definiciones).
*   **Salidas/Exportaciones:** Clases de nodos AST que son instanciadas por el `Parser.py` y utilizadas por subsiguientes fases.
*   **Clases Clave:**
    *   `Node`: Clase base para todos los nodos AST. Incluye `lineno`, `gox_type` (tipo GoX resuelto por el Analizador Semántico), y `semantic_info` (información semántica adicional). Implementa `to_dict()` para serialización JSON y `accept()` para el patrón Visitor.
    *   **Nodos de Expresión:** `Integer`, `Float`, `String`, `Char`, `Boolean`, `Location`, `BinOp`, `UnaryOp`, `MemoryAllocation`, `MemoryAddress`, `FunctionCall`, `TypeCast`, `CompareOp`, `LogicalOp`.
    *   **Nodos de Declaración:** `VariableDecl`, `ConstantDecl`, `Parameter`, `FunctionDecl`, `ImportDecl`, `FunctionImportDecl`. `ConstantDecl` y `Parameter` incluyen `type_spec`/`param_gox_type` para almacenar el tipo.
    *   **Nodos de Sentencia:** `Assignment`, `Print`, `If`, `While`, `Return`, `Break`, `Continue`. `Return` permite un campo de expresión opcional.
    *   `Program`: Nodo raíz del AST, que contiene la lista de elementos de alto nivel (`body`).

## 5. Análisis Sintáctico (`Parser.py`)

*   **Propósito:** Validar la gramática del código GoX y construir el AST. Permite "scripting global" (sentencias ejecutables a nivel superior, no solo dentro de funciones).
*   **Errores que Maneja:** Errores de sintaxis (tokens inesperados, falta de delimitadores, estructuras gramaticales inválidas).
*   **Entradas:** Una lista de objetos `Token` de `Lexer.py` y una instancia de `ErrorHandler`.
*   **Salidas/Exportaciones:** Un objeto `Program` (el AST) al `SemanticAnalyzer.py`.
*   **Clases y Métodos Clave:**
    *   `Parser`:
        *   `__init__(tokens, error_handler)`: Inicializa el estado del parser.
        *   `advance()`: Mueve el cursor al siguiente token.
        *   `peek(offset)`: Permite inspeccionar tokens futuros.
        *   `match(*expected_types)`: Intenta consumir el token actual si coincide con los tipos esperados.
        *   `consume(expected_type, error_message)`: Consume el token actual solo si coincide con el tipo esperado, registrando un error si no lo hace.
        *   `consume_type(error_message_prefix)`: Helper para consumir tokens de tipo GoX y normalizar su valor.
        *   `_synchronize(recovery_tokens)`: Método de recuperación de errores genérico que avanza el cursor hasta encontrar un token de recuperación.
        *   `_synchronize_function_error()`: Estrategia específica para recuperar errores durante el parseo de funciones, buscando el `RBRACE` de cierre.
        *   `parse()`: Método principal que construye el AST `Program`. Permite que las declaraciones de alto nivel (`IMPORT`, `VAR`, `CONST`, `FUNC`) o cualquier sentencia (`parse_statement()`) estén a nivel global.
        *   `parse_statement()`: Parsea una única sentencia. Distingue entre declaraciones locales, sentencias de control de flujo (`IF`, `WHILE`, `RETURN`, `BREAK`, `CONTINUE`), llamadas a funciones (`FunctionCall`) y asignaciones (`Assignment`). Detecta y reporta errores de funciones anidadas.
        *   `parse_statements_block()`: Parsea un bloque de sentencias (`{ ... }`), llamando a `parse_statement()` para cada sentencia interna.
        *   `parse_expression()` y sus métodos de precedencia (`parse_logical_or`, `parse_logical_and`, `parse_equality`, `parse_relational`, `parse_additive`, `parse_multiplicative`, `parse_unary`, `parse_primary`): Implementan la gramática de expresiones.
        *   `parse_primary()`: Parsea los elementos más básicos de una expresión, incluyendo literales, identificadores, llamadas a función como expresión, conversiones de tipo, y accesos/reservas de memoria (`MemoryAddress`, `MemoryAllocation`).
        *   `parse_declaration()`, `parse_function()`, `parse_import()`, `parse_print()`, `parse_if()`, `parse_while()`, `parse_assignment()`, `parse_return()`, `parse_control_statement()`: Métodos específicos para parsear estas construcciones.

## 6. Definiciones de Tipos (`Types.py`)

*   **Propósito:** Definir los tipos de datos fundamentales en GoX y las reglas de tipado para operaciones binarias y unarias.
*   **Errores que Maneja:** Este módulo no reporta errores; simplemente define las reglas que el `SemanticAnalyzer.py` consulta.
*   **Entradas:** Ninguna.
*   **Salidas/Exportaciones:**
    *   `gox_typenames`: Conjunto de cadenas con los nombres de tipos GoX válidos (ej., `'int'`, `'float'`, `'bool'`, `'char'`, `'string'`, `'void'`).
    *   `bin_ops_type_rules`: Diccionario que mapea tuplas `(tipo_operando_izq, operador, tipo_operando_der)` a la cadena del tipo resultante. Es estricto con los tipos, no permitiendo `int + float`.
    *   `unary_ops_type_rules`: Diccionario que mapea tuplas `(operador, tipo_operando)` a la cadena del tipo resultante.
    *   `check_binop_type()`: Función que busca el tipo resultante para una operación binaria.
    *   `check_unaryop_type()`: Función que busca el tipo resultante para una operación unaria.

## 7. Tabla de Símbolos (`SymbolTable.py` o `Sym_tab.py`)

*   **Propósito:** Gestionar la información sobre los identificadores (símbolos) declarados en el programa, como variables, constantes y funciones, y sus propiedades (tipo, alcance). Permite la gestión de alcances anidados.
*   **Errores que Maneja:** Lanza `SymbolAlreadyDefinedError` si un símbolo intenta ser redeclarado en el mismo alcance.
*   **Entradas:** Nombres de símbolos y sus `SymbolEntry`s.
*   **Salidas/Exportaciones:** Permite la búsqueda de símbolos a través de su jerarquía de alcances. Es un componente interno clave del `SemanticAnalyzer.py`.
*   **Clases Clave:**
    *   `SymbolEntry`: Almacena el `name`, `gox_type` (tipo GoX del símbolo), `declaration_node` (el nodo AST donde se declaró), `is_constant`, `scope_level` (nivel de anidamiento del alcance), y opcionalmente `value` (para constantes).
    *   `SymbolTable`: Representa un alcance.
        *   `scope_name`: Un identificador para el alcance.
        *   `entries`: Un diccionario de `name -> SymbolEntry` para símbolos definidos en este alcance.
        *   `parent_scope`: Referencia al alcance padre.
        *   `child_scopes`: Lista de alcances hijos.
        *   `scope_level`: Profundidad del alcance.
    *   `SymbolAlreadyDefinedError`: Excepción levantada en caso de redefinición.
*   **Métodos Clave:**
    *   `add_symbol(entry)`: Añade un símbolo al alcance actual.
    *   `lookup_symbol(name, current_scope_only)`: Busca un símbolo por nombre, recursivamente en alcances padres si `current_scope_only` es falso.
    *   `print_table(indent)`: Imprime la tabla de símbolos y sus alcances hijos de forma jerárquica.

## 8. Análisis Semántico (`SemanticAnalyzer.py`)

*   **Propósito:** Verificar la corrección semántica del programa GoX, aplicando reglas de tipo, resolución de nombres y validación de flujo de control. Enriquece el AST con información de tipo y símbolos.
*   **Errores que Maneja:** Errores semánticos como:
    *   Tipos incompatibles en operaciones, asignaciones, casts.
    *   Uso de variables o funciones no definidas.
    *   Asignación a constantes.
    *   Condiciones de `if`/`while` no booleanas.
    *   Sentencias `break`/`continue` fuera de un bucle.
    *   Sentencias `return` fuera de una función o con tipo de retorno incorrecto.
    *   Declaración de variables sin tipo ni valor.
    *   Redefinición de símbolos en el mismo alcance.
    *   Número o tipo de argumentos incorrectos en llamadas a funciones.
*   **Entradas:** El AST (nodo `Program`) del `Parser.py` y una instancia de `ErrorHandler`. Utiliza `SymbolTable.py` y `Types.py`.
*   **Salidas/Exportaciones:** El mismo AST de entrada, pero **modificado/enriquecido** con `node.gox_type` y `node.semantic_info`.
*   **Clases y Métodos Clave:**
    *   `SemanticAnalyzer`:
        *   `__init__(error_handler)`: Inicializa la tabla de símbolos global (`global_scope`) y el `loop_depth` (para control de bucles).
        *   `enter_scope(scope_name_prefix)`: Abre un nuevo alcance.
        *   `exit_scope()`: Cierra el alcance actual.
        *   `analyze(node)`: Método principal que inicia el recorrido del AST.
        *   **`visit_NodeType(node)`:** Métodos especializados para cada tipo de nodo AST.
            *   Manejan la adición de símbolos a la `SymbolTable` (`visit_VariableDecl`, `visit_ConstantDecl`, `visit_FunctionDecl`, `visit_FunctionImportDecl`).
            *   Realizan comprobaciones de tipo y validez para expresiones (`visit_BinOp`, `visit_UnaryOp`, `visit_CompareOp`, `visit_TypeCast`).
            *   Verifican el uso de variables y funciones (`visit_Location`, `visit_FunctionCall`).
            *   Aplican reglas de control de flujo (`visit_If`, `visit_While`, `visit_Break`, `visit_Continue`, `visit_Return`).
            *   Establecen el `node.gox_type` para las expresiones y declaraciones.
            *   Detectan errores específicos y los reportan al `ErrorHandler`.

## 9. Serialización del AST a JSON (`AST_to_JSON.py`)

*   **Propósito:** Convertir el AST a un formato JSON, útil para depuración e inspección del árbol.
*   **Errores que Maneja:** Puede reportar `TypeError` si los datos del AST no son serializables por JSON.
*   **Entradas:** Un nodo raíz del AST (objeto `Program`).
*   **Salidas/Exportaciones:** Una estructura de datos Python (diccionarios y listas) que representa el AST en formato serializable por JSON.
*   **Funciones Clave:**
    *   `ast_to_json(node_or_value)`: Función recursiva que delega la serialización al método `to_dict()` de los nodos AST.
    *   `save_ast_to_json(ast_serializable_data, filename)`: Guarda la estructura serializable en un archivo JSON.
    *   `pretty_print_json(json_data, indent)`: Imprime la representación JSON a la consola de forma legible.

## 10. Generación de Código Intermedio (`IRCodeGenerator.py`)

*   **Propósito:** Traducir el AST semánticamente validado y enriquecido a una secuencia de instrucciones de Código Intermedio (IR) basado en pila.
*   **Errores que Maneja:** Errores durante la traducción del AST a IR (ej. tipos GoX no mapeables a IR, construcciones no soportadas por el IR, o fallos en la lógica de generación de IR).
*   **Entradas:** El AST enriquecido (nodo `Program`) del `SemanticAnalyzer.py` y una instancia de `ErrorHandler`.
*   **Salidas/Exportaciones:** Un objeto `IRProgramRepresentation` que contiene las instrucciones IR generadas. También imprime las instrucciones IR a la consola.
*   **Clases Clave (definidas internamente o importadas para la representación del IR):**
    *   `IRFunctionRepresentation`: Representa una función en IR con `name`, `params` (nombres y tipos IR), `locals` (variables locales y sus tipos IR), `code` (lista de tuplas de instrucciones IR), `return_ir_type`, y `is_imported`.
    *   `IRProgramRepresentation`: Contenedor para todo el programa IR. Incluye `globals` (variables globales y sus tipos IR) y `functions` (un diccionario de objetos `IRFunctionRepresentation`).
*   **Funciones y Métodos Clave:**
    *   `IRCodeGenerator`:
        *   `__init__(error_handler)`: Inicializa el generador y el contador de variables temporales.
        *   `generate_ir(program_node)`: Método principal para generar el IR. Realiza una primera pasada para identificar firmas de funciones y globales. Crea la función `_init` (implícita para el código global). Luego, en una segunda pasada, visita los nodos del AST para emitir instrucciones IR.
        *   `_emit(*args)`: Añade una instrucción IR a la lista de código de la función actual (`current_ir_function.code`) y la imprime.
        *   `_map_gox_type_to_ir(gox_type)`: Mapea un tipo GoX a su representación IR (`'I'`, `'F'`, etc.).
        *   **`visit_NodeType(node)`:** Métodos especializados para cada tipo de nodo AST. Traducen la semántica del nodo a una secuencia de instrucciones IR. Ejemplos: `visit_Integer` emite `('CONSTI', value)`, `visit_Assignment` emite las instrucciones para evaluar el valor, la dirección y luego un `('POKEI',)`. Los métodos para control de flujo (`visit_If`, `visit_While`) generan las instrucciones IR estructuradas (`IF`, `ELSE`, `ENDIF`, `LOOP`, `CBREAK`, `CONTINUE`, `ENDLOOP`).

## 11. Máquina de Pila (`StackMachine.py`)

*   **Propósito:** Interpretar y ejecutar el Código Intermedio (IR) generado por `IRCodeGenerator.py`. Simula el comportamiento de una máquina virtual de pila.
*   **Errores que Maneja:** Errores de tiempo de ejecución (ej. división por cero, acceso a memoria fuera de límites, pila vacía, tipos incompatibles en la pila).
*   **Entradas:** Un objeto `IRProgramRepresentation` (o su representación serializada) y una instancia de `ErrorHandler`.
*   **Salidas/Exportaciones:** La salida estándar (consola) producida por las instrucciones `PRINTI/F/B` del programa GoX.
*   **Clases y Atributos Clave:**
    *   `StackMachine`:
        *   `__init__(error_handler, memory_size_bytes)`: Inicializa la pila de operandos (`stack`), la memoria (`memory` como `bytearray`), el almacenamiento de variables globales (`globals`), la pila de contextos locales (`locals_env_stack`), la pila de llamadas (`call_stack_return_pcs`), los metadatos de las funciones (`functions_meta`), y el programa IR plano (`program_ir`).
        *   `_pop_stack()`, `_pop_typed_val()`: Métodos para extraer valores de la pila con comprobación de tipo.
        *   `_push_locals_scope()`, `_pop_locals_scope()`, `_current_locals_scope()`: Para gestionar los alcances de variables locales durante la ejecución de funciones.
        *   `_assign_local()`, `_lookup_local()`: Para asignar y buscar valores de variables locales.
        *   `load_ir(ir_program_repr)`: Carga el IR desde el objeto `IRProgramRepresentation`, poblando `functions_meta` y `program_ir`.
        *   `run(entry_function_name)`: El bucle principal de ejecución (fetch-decode-execute). Inicia desde la función de entrada (normalmente `_init`).
        *   **`op_OPCODE(*args)`:** Métodos específicos para cada código de operación IR.
            *   Ejemplos: `op_CONSTI()`, `op_ADDI()`, `op_SUBF()`, `op_PRINTI()`, `op_ITOF()`, `op_LOCAL_GET()`, `op_GLOBAL_SET()`.
            *   `op_CALL(func_name)`: Implementa el mecanismo de llamada a función (salvar PC, crear nuevo alcance local, pasar argumentos, inicializar locales, saltar a la función).
            *   `op_RET()`: Restaura el contexto de la función llamadora.
            *   `op_PEEKI/F/B()`, `op_POKEI/F/B()`, `op_GROW()`: Operaciones de memoria, incluyendo conversión de tipos y gestión de límites.
            *   `op_IF()`, `op_ELSE()`, `op_ENDIF()`, `op_LOOP()`, `op_CBREAK()`, `op_CONTINUE()`, `op_ENDLOOP()`: Operaciones de control de flujo. Estas manipulan directamente el `pc` y utilizan una pila de marcadores (`_control_flow_markers_stack`) para gestionar las estructuras de bucle y condicionales.