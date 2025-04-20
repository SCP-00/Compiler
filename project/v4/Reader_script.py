# Reader_script.py
import sys
from Error import ErrorHandler
from Lexer import tokenize
from Parser import Parser

def main():
    if len(sys.argv) < 2:
        print("Usage: python Reader_script.py [filename].gox")
        sys.exit(1)

    filename = sys.argv[1]
    if not filename.endswith('.gox'):
        print("Error: File must have .gox extension")
        sys.exit(1)

    try:
        with open(filename, 'r') as file:
            code = file.read()

        error_handler = ErrorHandler()
        tokens = tokenize(code, error_handler)
        parser = Parser(tokens, error_handler)
        ast = parser.parse()

        output_filename = filename.rsplit('.', 1)[0] + '.json'

        if error_handler.has_errors():
            print(f"Parsing failed for {filename}:")
            error_handler.report_errors()
            sys.exit(1)
        else:
            print(f"Successfully parsed {filename}")
            parser.save_ast_to_json(output_filename)
            print(f"AST saved to {output_filename}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()