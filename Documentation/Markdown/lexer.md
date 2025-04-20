## Lexer.py - Análisis Detallado del Código y Resumen

Este documento proporciona un análisis detallado del archivo `Lexer.py`, cubriendo su funcionalidad, estructura y componentes clave. También incluye un resumen de su propósito y comportamiento.

**1. Propósito:**

El archivo `Lexer.py` implementa un analizador léxico (lexer o escáner) para un lenguaje de programación personalizado (implícito por las palabras clave y los operadores definidos). La función principal del lexer es tomar una cadena de código fuente como entrada y dividirla en un flujo de tokens. Cada token representa una unidad significativa del lenguaje, como palabras clave, identificadores, literales, operadores y signos de puntuación. Estos tokens se utilizan luego como entrada para la siguiente etapa del compilador, el analizador sintáctico (parser).

**2. Estructura y Componentes:**

El código está organizado en las siguientes secciones clave:

*   **Definiciones de Tokens (`TOKEN_SPEC`):** Esta es una lista crucial de tuplas. Cada tupla define un tipo de token (por ejemplo, `INTEGER`, `ID`, `PLUS`) y su patrón de expresión regular correspondiente. El *orden* de estas definiciones es crítico, porque el lexer intenta hacer coincidir los tokens de arriba a abajo. Las palabras clave deben aparecer antes de los identificadores para evitar que las palabras clave se identifiquen erróneamente como identificadores. El uso de `\b` asegura que se coincida con la palabra completa, evitando que se reconozcan subcadenas de palabras clave.

*   **Clase Token (`Token`):** Una clase simple para representar un token. Almacena el tipo de token (`type`), su valor (`value`) y el número de línea (`lineno`) donde se encontró en el código fuente. El método `__repr__` proporciona una representación de cadena fácil de usar de los objetos Token para la depuración.

*   **Función de Tokenización (`tokenize`):** Este es el núcleo del lexer. Toma el texto del código fuente y un objeto de manejador de errores como entrada y devuelve una lista de tokens. Itera a través del texto de entrada, intentando hacer coincidir los tokens basándose en las expresiones regulares definidas en `TOKEN_SPEC`. Maneja:

    *   **Comentarios:** Comentarios de una sola línea (`//`) y comentarios de bloque anidados (`/* ... */`). Es importante destacar que el manejo de comentarios de bloque anidados cuenta correctamente la profundidad de anidamiento, maneja los comentarios no terminados y actualiza el número de línea.
    *   **Espacios en Blanco:** Los espacios en blanco se omiten, pero los números de línea se actualizan en consecuencia.
    *   **Desajustes:** Si ningún token coincide con una porción de la entrada, se informa un error utilizando el manejador de errores.
    *   **Coincidencia de Expresiones Regulares:** La expresión regular combinada `token_regex` construida a partir de `TOKEN_SPEC` se utiliza para encontrar eficientemente el siguiente token. Se especifica `re.DOTALL` en la coincidencia de expresiones regulares para permitir que '.' coincida con caracteres de nueva línea dentro de comentarios de bloque y literales de cadena.

*   **Expresión Regular Combinada:** La variable `token_regex` se crea utilizando `re.compile`. Esto combina todos los patrones de token individuales en una sola expresión regular, que es más eficiente que intentar hacer coincidir cada patrón por separado. La sintaxis `(?P<name>pattern)` se utiliza para nombrar cada grupo en la expresión regular, lo que permite que `match.lastgroup` determine qué tipo de token coincidió.

*   **Ejemplo de Uso:** El bloque `if __name__ == "__main__":` demuestra cómo usar el lexer. Incluye código de muestra (`test_code`), crea una instancia de `ErrorHandler`, llama a la función `tokenize`, imprime los tokens resultantes e informa cualquier error encontrado.

**3. Características Clave y Consideraciones:**

*   **Expresiones Regulares:** El lexer se basa en gran medida en expresiones regulares para definir los patrones para cada token. Comprender las expresiones regulares es esencial para comprender y modificar el lexer.
*   **Orden de las Definiciones de Tokens:** El orden de los tokens en `TOKEN_SPEC` es crucial para la tokenización correcta. Por ejemplo, las palabras clave deben definirse antes de los identificadores. Los operadores más largos (por ejemplo, `++`, `+=`) deben definirse antes que sus contrapartes más cortas (`+`, `=`).
*   **Manejo de Errores:** El lexer incluye un manejo de errores básico utilizando una clase `ErrorHandler` (que se supone que está definida en `Error.py`). Informa caracteres ilegales y comentarios de bloque no terminados.
*   **Seguimiento del Número de Línea:** El lexer realiza un seguimiento de los números de línea para proporcionar informes de errores precisos.
*   **Eficiencia:** La combinación de las expresiones regulares en una sola `token_regex` mejora el rendimiento.
*   **Comentarios de Bloque Anidados:** El código maneja correctamente los comentarios de bloque anidados utilizando un contador de profundidad.
*   **Literales de Cadena y Caracteres:** Las expresiones regulares para literales de cadena y caracteres permiten caracteres de escape (por ejemplo, `\n`, `\"`).
*   **Lookarounds:** La expresión regular para `INT_DIV` utiliza un lookbehind negativo `(?<!/)` y un lookahead positivo `(?=\s*\d)` para asegurar que coincida solo con el operador de división entera `//` cuando no es parte de una secuencia más larga de barras y cuando es seguido por un dígito (con espacios en blanco opcionales). Esto ayuda a evitar la interpretación errónea de los comentarios.

**4. Mejoras Potenciales:**

*   **Recuperación de Errores:** La recuperación de errores del lexer es básica. Podría mejorarse para intentar resincronizar después de un error, permitiéndole continuar tokenizando el resto de la entrada.
*   **Manejo Más Robusto de Literales de Cadena y Caracteres:** Las expresiones regulares para literales de cadena y caracteres podrían hacerse más robustas para manejar secuencias de escape más complejas y posibles errores en el formato de cadenas.
*   **Soporte Unicode:** El lexer podría extenderse para admitir caracteres Unicode en identificadores y literales de cadena.
*   **Informes de Errores Más Granulares:** El lexer podría proporcionar mensajes de error más específicos, como indicar el token esperado o el tipo de error de sintaxis encontrado.

**5. Resumen:**

El archivo `Lexer.py` proporciona un lexer funcional y bien estructurado para un lenguaje de programación personalizado. Utiliza expresiones regulares para definir tokens, maneja comentarios (incluidos comentarios de bloque anidados), realiza un seguimiento de los números de línea y proporciona un manejo de errores básico. Si bien podría mejorarse con una recuperación de errores más sofisticada y soporte Unicode, sirve como una base sólida para un compilador o intérprete. La inclusión de un código de prueba completo y un informe de errores claro lo convierte en una herramienta valiosa para el procesamiento del lenguaje.