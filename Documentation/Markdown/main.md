## main.py - Punto de Entrada del Compilador

Este módulo es el punto de entrada principal para el compilador.  Coordina las diferentes fases del compilador (lectura, análisis léxico, análisis sintáctico, análisis semántico, interpretación).

### Funciones

*   `run_reader_script(filename)`: Ejecuta el script `Reader_script.py` en un subproceso para leer el contenido de un archivo.
*   `open_file_dialog()`: Abre un diálogo para seleccionar un archivo `.gox`.
*   `main()`:
    *   Lee el código fuente del archivo especificado como argumento de línea de comandos.
    *   Realiza el análisis léxico (tokenización) utilizando el `Lexer`.
    *   Realiza el análisis sintáctico (parsing) utilizando el `Parser`.
    *   Realiza el análisis semántico utilizando el `SemanticChecker`.
    *   Ejecuta el código utilizando el `Interpreter`.