# Error.py

import sys
from typing import List, Optional, Any # Added Any

class ErrorType:
    LEXICAL       = 'LEXICAL'
    SYNTAX        = 'SYNTAX'
    SEMANTIC      = 'SEMANTIC'
    INTERPRETATION= 'INTERPRETATION'
    GENERAL       = 'GENERAL' # Default or unspecified error type

    @classmethod
    def normalize(cls, t: Any) -> str: # Accept Any for t
        t_str = str(t).upper() if not isinstance(t, str) else t.upper()
        if t_str in {cls.LEXICAL, cls.SYNTAX, cls.SEMANTIC, cls.INTERPRETATION, cls.GENERAL}:
            return t_str
        return cls.GENERAL

class ErrorEntry:
    def __init__(self, message: str, lineno: Optional[int], colno: Optional[int], error_type: str):
        self.message: str = message
        self.lineno: Optional[int] = lineno
        self.colno: Optional[int] = colno # Column number is optional
        self.type: str = ErrorType.normalize(error_type)

    def __str__(self) -> str:
        loc_parts = []
        if self.lineno is not None:
            loc_parts.append(f"Line {self.lineno}")
        if self.colno is not None:
            loc_parts.append(f"Col {self.colno}")
        
        loc_str = ", ".join(loc_parts)
        return f"{self.type}: {self.message} [{loc_str}]" if loc_str else f"{self.type}: {self.message}"

    def to_dict(self) -> dict:
        """Converts the error entry to a dictionary for serialization."""
        return {
            "type": self.type,
            "message": self.message,
            "lineno": self.lineno,
            "colno": self.colno,
        }

class CompilerError(Exception):
    """Base exception for compiler phases. Can be caught by specific phases."""
    def __init__(self, message: str, lineno: Optional[int] = None, colno: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.lineno = lineno
        self.colno = colno

# Specific error types can inherit from CompilerError if needed for try-except blocks
class LexicalError(CompilerError): pass
class SyntaxError(CompilerError): pass
class SemanticError(CompilerError): pass
class InterpretationError(CompilerError): pass

class ErrorHandler:
    def __init__(self):
        self._errors: List[ErrorEntry] = []
        self.had_error_since_last_check: bool = False # For Parser's use

    def add_error(self, message: str, lineno: Optional[int] = None, colno: Optional[int] = None, error_type: str = ErrorType.GENERAL):
        """Registers an error in any compiler phase."""
        # self.had_error = True # This was the old flag, deprecated for has_errors()
        self.had_error_since_last_check = True
        entry = ErrorEntry(message, lineno, colno, error_type)
        self._errors.append(entry)

    def add_lexical_error(self, message: str, lineno: int, colno: Optional[int] = None):
        self.add_error(message, lineno, colno, ErrorType.LEXICAL)

    def add_syntax_error(self, message: str, lineno: Optional[int], colno: Optional[int] = None):
        self.add_error(message, lineno, colno, ErrorType.SYNTAX)

    def add_semantic_error(self, message: str, lineno: Optional[int], colno: Optional[int] = None):
        self.add_error(message, lineno, colno, ErrorType.SEMANTIC)

    def add_interpretation_error(self, message: str, lineno: Optional[int] = None, colno: Optional[int] = None):
        self.add_error(message, lineno, colno, ErrorType.INTERPRETATION)

    def get_formatted_errors(self) -> str: # Renamed for clarity
        """Returns a formatted string of all errors, sorted."""
        if not self._errors:
            return "No errors found."
        
        # Sort errors by line number, then column number (if available)
        sorted_errors = sorted(
            self._errors, 
            key=lambda e: (e.lineno if e.lineno is not None else float('inf'), 
                           e.colno if e.colno is not None else float('inf'))
        )
        
        error_messages = [str(error) for error in sorted_errors]
        return "\n".join(error_messages)

    def get_entries(self) -> List[ErrorEntry]:
        """Returns the list of collected error entries."""
        return self._errors

    def has_errors(self) -> bool:
        """Checks if any errors have been registered."""
        return bool(self._errors)

    def report_errors(self, out=sys.stderr):
        """Prints all registered errors, sorted by location, to the specified output stream."""
        if not self.has_errors():
            return
        
        print("\n--- Compilation Errors ---", file=out)
        print(self.get_formatted_errors(), file=out) # Use the sorted, formatted string
        print(f"Total errors: {len(self._errors)}", file=out)

    def exit_if_errors(self, exit_code: int = 1):
        """Report errors and exit if any are registered."""
        if self.has_errors():
            self.report_errors()
            sys.exit(exit_code)

    def clear_errors(self):
        """Clears all registered errors."""
        self._errors = []
        self.had_error_since_last_check = False
    
    def has_errors_since(self, previous_error_count: int) -> bool:
        """Checks if new errors were added since a certain point."""
        return len(self._errors) > previous_error_count

    def get_error_count(self) -> int:
        """Returns the current number of errors."""
        return len(self._errors)