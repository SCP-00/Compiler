# Compiler
Este archivo muestra el avance de un grupo de estudiante en crear su propio compilador básico (Elementary Compiler Program)
# Lexer para un Lenguaje de Programación
### Integrantes
- Juan Felipe Arbeláez Osorio
- Victor Alejandro Buendia Henao
- Juan Sebastián valencia Uribe

Este proyecto es un **analizador léxico** (lexer) escrito en Python. Su función es tomar un fragmento de código fuente y descomponerlo en **tokens**, que son las unidades fundamentales del lenguaje.

## 📌 Características

- Soporta palabras reservadas como `var`, `if`, `while`, `return`, etc.
- Identifica operadores (`+`, `-`, `*`, `<=`, `==`, etc.)
- Reconoce identificadores, números enteros, flotantes y caracteres
- Soporta comentarios de línea (`//`) y de bloque (`/* ... */`)
- Ignora espacios en blanco
- Detecta caracteres ilegales

---

## 📜 Ejemplo de Código de Entrada

El lexer analiza código como este:

```c
var x = 10;
if (x >= 5) {
    print(x);
}
```

### 🔍 Tokens Generados

El código anterior se convierte en la siguiente lista de tokens:

```
Token(type='VAR', value='var', lineno=1)
Token(type='ID', value='x', lineno=1)
Token(type='ASSIGN', value='=', lineno=1)
Token(type='INTEGER', value='10', lineno=1)
Token(type='SEMI', value=';', lineno=1)
Token(type='IF', value='if', lineno=2)
Token(type='LPAREN', value='(', lineno=2)
Token(type='ID', value='x', lineno=2)
Token(type='GE', value='>=', lineno=2)
Token(type='INTEGER', value='5', lineno=2)
Token(type='RPAREN', value=')', lineno=2)
Token(type='LBRACE', value='{', lineno=2)
Token(type='PRINT', value='print', lineno=3)
Token(type='LPAREN', value='(', lineno=3)
Token(type='ID', value='x', lineno=3)
Token(type='RPAREN', value=')', lineno=3)
Token(type='SEMI', value=';', lineno=3)
Token(type='RBRACE', value='}', lineno=4)
```

---

## Cómo Usarlo?

Para ejecutar el lexer, simplemente corre el script en Python:

```bash
python lexer.py
```

Si dentro del código fuente hay caracteres ilegales, el lexer los detecta y muestra un error. Por ejemplo, si el código contiene `var b = @20;`, la salida incluirá:

```
Línea 2: Error - Caracter ilegal '@'
```

---

## 🛠 Estructura del Código

El lexer usa expresiones regulares para identificar distintos tipos de tokens. Los pasos clave son:

1. Definir los **patrones de tokens** en `TOKEN_SPEC` para categorizar su clase.
2. Crear una **expresión regular combinada**.
3. Usar `re.finditer()` para recorrer el código fuente y generar una lista de tokens.
4. Ignorar espacios en blanco y comentarios.
5. Detectar errores de caracteres ilegales.

---
# Problemas iniciales
Por falta de atencion durante la clase cometimos el error de ubicar primero algunos tokenz de un solo caracteres antes que sus versiones compuestas de dos caracteres
### Ejemplo
```
# Operadores compuestos (más de un carácter)
    ('LE', r'<='), ('GE', r'>='), ('EQ', r'=='), ('NE', r'!='), 
    ('LAND', r'&&'), ('LOR', r'\|\|'),
    
    # Operadores de un solo carácter
    ('LT', r'<'), ('GT', r'>'), ('PLUS', r'\+'), ('MINUS', r'-'),
    ('TIMES', r'\*'), ('DIVIDE', r'/'), ('GROW', r'\^'), ('ASSIGN', r'='),
```
donde los operadores de un solo caracter como el `mayor que` y el `menor que` estaban antes que su version compuesta `mayor o igual que` y `menor o igual que` respectivamente
## 📝 Nota

- Es un lexer básico, no un parser y mucho menos un compilador completo. No verifica la sintaxis completa, solo descompone el código en tokens.
- Se puede ampliar fácilmente agregando más palabras clave u operadores como por ejemplo el uso de un token **CLASSES**.

# Universidad Tecnologica de Pereira
## 2025-1
### Ingenieria de Sistemas // Clase Compiladores IS75
### Docente: Angel Augusto Agudelo Zapata
