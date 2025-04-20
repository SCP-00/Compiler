import sys

class ErrorType:
    LEXICAL       = 'LEXICAL'
    SYNTAX        = 'SYNTAX'
    SEMANTIC      = 'SEMANTIC'
    INTERPRETATION= 'INTERPRETATION'
    GENERAL       = 'GENERAL'

    @classmethod
    def normalize(cls, t):
        t = t.upper() if isinstance(t, str) else str(t).upper()
        return t if t in {
            cls.LEXICAL, cls.SYNTAX, cls.SEMANTIC,
            cls.INTERPRETATION, cls.GENERAL
        } else cls.GENERAL

class ErrorEntry:
    def __init__(self, message, lineno, colno, type):
        self.message = message
        self.lineno = lineno
        self.colno = colno
        self.type = type

class CompilerError(Exception):
    """Base exception for compiler phases."""
    def __init__(self, message, lineno=None, colno=None):
        super().__init__(message)
        self.message = message
        self.lineno = lineno
        self.colno = colno

class LexicalError(CompilerError):       pass
class SyntaxError(CompilerError):        pass
class SemanticError(CompilerError):      pass
class InterpretationError(CompilerError): pass

class ErrorHandler:
    def __init__(self):
        self.errors = []

    def add_error(self, message, lineno=None, colno=None, error_type=ErrorType.GENERAL):
        """Registers an error in any compiler phase."""
        et = ErrorType.normalize(error_type)
        entry = ErrorEntry(message, lineno or 0, colno, et)
        self.errors.append(entry)

    def add_lexical_error(self, message, lineno, colno=None):
        self.add_error(message, lineno, colno, ErrorType.LEXICAL)

    def add_syntax_error(self, message, lineno, colno=None):
        self.add_error(message, lineno, colno, ErrorType.SYNTAX)

    def add_semantic_error(self, message, lineno, colno=None):
        self.add_error(message, lineno, colno, ErrorType.SEMANTIC)

    def add_interpretation_error(self, message, lineno, colno=None):
        self.add_error(message, lineno, colno, ErrorType.INTERPRETATION)

    def has_errors(self):
        return bool(self.errors)

    def report_errors(self):
        """Prints all errors sorted by location."""
        for err in sorted(self.errors, key=lambda e: (e.lineno, e.colno or 0)):
            loc = f"Line {err.lineno}"
            if err.colno is not None:
                loc += f":{err.colno}"
            print(f"{loc} [{err.type}] {err.message}")

    def exit_if_errors(self, exit_code=1):
        """Report and exit if any errors are registered."""
        if self.has_errors():
            self.report_errors()
            sys.exit(exit_code)

    def clear_errors(self):
        """Clears all registered errors."""
        self.errors.clear()

# Usage example
if __name__ == "__main__":
    eh = ErrorHandler()
    eh.add_lexical_error("Unexpected token", 3, 5)
    eh.add_syntax_error("Missing ';'", 4)
    eh.add_semantic_error("Type mismatch", 7, 12)
    eh.add_interpretation_error("Division by zero", 15, 2)
    if eh.has_errors():
        eh.report_errors()
