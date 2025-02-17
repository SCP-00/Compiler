# Pruebas Unitarias para el Lexer

Este archivo contiene pruebas unitarias para el **analizador léxico** (lexer) definido en `analizar_lexico.py`. Utiliza `unittest` para verificar que el lexer tokeniza correctamente distintos tipos de entradas.

##  Descripción de las pruebas

Cada prueba analiza un fragmento de código y compara los tokens generados con los esperados.

### 1. Prueba de Palabras Reservadas

Verifica que el lexer reconozca correctamente las palabras clave del lenguaje, como `var`, `return`, `if`, `else`, `while`, `print`.

- **Código de prueba:**
  ```c
  var return if else while print
  ```
- **Tokens esperados:**
  ```
  Token("VAR", "var", 1)
  Token("RETURN", "return", 1)
  Token("IF", "if", 1)
  Token("ELSE", "else", 1)
  Token("WHILE", "while", 1)
  Token("PRINT", "print", 1)
  ```

### 2. Prueba de Identificadores y Números

Comprueba que el lexer reconozca correctamente identificadores, enteros, flotantes y caracteres.

- **Código de prueba:**
  ```c
  x = 42; y = 3.14; z = 'a';
  ```
- **Tokens esperados:**
  ```
  Token("ID", "x", 1)
  Token("ASSIGN", "=", 1)
  Token("INTEGER", "42", 1)
  Token("SEMI", ";", 1)
  Token("ID", "y", 1)
  Token("ASSIGN", "=", 1)
  Token("FLOAT", "3.14", 1)
  Token("SEMI", ";", 1)
  Token("ID", "z", 1)
  Token("ASSIGN", "=", 1)
  Token("CHAR", "'a'", 1)
  Token("SEMI", ";", 1)
  ```

### 3. Prueba de Operadores

Verifica el reconocimiento de operadores lógicos y de comparación.

- **Código de prueba:**
  ```c
  x <= 10 && y >= 5 || z == 3
  ```
- **Tokens esperados:**
  ```
  Token("ID", "x", 1)
  Token("LE", "<=", 1)
  Token("INTEGER", "10", 1)
  Token("LAND", "&&", 1)
  Token("ID", "y", 1)
  Token("GE", ">=", 1)
  Token("INTEGER", "5", 1)
  Token("LOR", "||", 1)
  Token("ID", "z", 1)
  Token("EQ", "==", 1)
  Token("INTEGER", "3", 1)
  ```

### 4. Prueba de Caracteres Ilegales

Comprueba que el lexer detecte caracteres ilegales y no los incluya en la lista de tokens.

- **Código de prueba:**
  ```c
  var a = @10;
  ```
- **Salida esperada:**
  ```
  Línea 1: Error - Caracter ilegal '@'
  ```
- **Tokens esperados:**
  ```
  Token("VAR", "var", 1)
  Token("ID", "a", 1)
  Token("ASSIGN", "=", 1)
  Token("INTEGER", "10", 1)
  Token("SEMI", ";", 1)
  ```

## Cómo Ejecutar las Pruebas

Ejecuta las pruebas con el siguiente comando en la terminal:
```bash
python -m unittest nombre_del_archivo.py
```
Si el lexer funciona correctamente, no se mostrarán errores en la salida.

**Nota:** Asegúrate de que `analizar_lexico.py` esté en la misma carpeta o en una ruta accesible para ser importado en este archivo de pruebas.

---

Estas pruebas ayudan a garantizar que el lexer identifica correctamente los tokens en distintos escenarios. ¡Úsalas para validar y mejorar el lexer conforme agregues más características! 🎯

