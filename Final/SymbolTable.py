# Symbol_Tab.py (o SymbolTable.py)
from typing import Any, Dict, Optional, List, Union
# from Nodes_AST import Node # Import Node if symbol values are AST nodes

# Forward declaration for type hinting if Node is used for symbol values
# This is a common pattern if Nodes_AST.py imports SymbolTable.py, creating a circular dependency.
# If SymbolTable stores simple dicts or dedicated SymbolEntry objects, this isn't strictly needed here.
Node = Any # Placeholder if Node objects are stored directly

class SymbolEntry:
    """
    Represents an entry in the symbol table.
    This provides a more structured way to store symbol information.
    """
    def __init__(self, name: str, gox_type: str, declaration_node: Optional[Node] = None, 
                 is_constant: bool = False, scope_level: int = 0, 
                 value: Any = None): # value can be actual runtime value for constants
        self.name: str = name
        self.gox_type: str = gox_type              # e.g., "int", "float", "function"
        self.declaration_node: Optional[Node] = declaration_node # AST node where it was declared
        self.is_constant: bool = is_constant
        self.scope_level: int = scope_level        # Scope depth where defined
        self.value: Any = value                    # For constants, can store the actual value
        # For functions, value could be the FunctionDecl node or a more specific callable representation
        # For variables, value is typically managed at runtime, not compile-time in symbol table
        # (unless it's a global with a known compile-time constant value)

    def __str__(self) -> str:
        const_str = " (const)" if self.is_constant else ""
        return f"SymbolEntry(name='{self.name}', type='{self.gox_type}'{const_str}, scope={self.scope_level})"

    def __repr__(self) -> str:
        return self.__str__()

class SymbolTable:
    """
    Manages symbol tables for different scopes.
    A new SymbolTable instance is typically created for each scope.
    """
    class SymbolAlreadyDefinedError(Exception):
        """Raised when a symbol is redefined in the same scope."""
        pass

    class SymbolNotFoundError(Exception): # Not typically raised by add/get, but useful for other operations
        """Raised when a symbol is not found."""
        pass

    def __init__(self, scope_name: str, parent_scope: Optional['SymbolTable'] = None, scope_level: int = 0):
        self.scope_name: str = scope_name  # e.g., "global", "function_foo", "block_if_line_10"
        self.entries: Dict[str, SymbolEntry] = {}
        self.parent_scope: Optional[SymbolTable] = parent_scope
        self.child_scopes: List[SymbolTable] = [] # For visualizing hierarchy
        self.scope_level: int = scope_level # Depth of this scope

        if self.parent_scope:
            self.parent_scope.child_scopes.append(self)

    def add_symbol(self, entry: SymbolEntry) -> None:
        """
        Adds a SymbolEntry to the current scope.
        Raises SymbolAlreadyDefinedError if the name already exists in this scope.
        """
        if entry.name in self.entries:
            # Could add more complex logic for conflicting types if re-declaration with different type is an error
            raise SymbolTable.SymbolAlreadyDefinedError(
                f"Symbol '{entry.name}' already defined in scope '{self.scope_name}'."
            )
        self.entries[entry.name] = entry
        entry.scope_level = self.scope_level # Ensure entry knows its scope level

    def lookup_symbol(self, name: str, current_scope_only: bool = False) -> Optional[SymbolEntry]:
        """
        Looks up a symbol by name.
        If current_scope_only is True, only searches the current scope.
        Otherwise, searches current scope then recursively searches parent scopes.
        Returns the SymbolEntry if found, else None.
        """
        if name in self.entries:
            return self.entries[name]
        elif not current_scope_only and self.parent_scope:
            return self.parent_scope.lookup_symbol(name)
        return None # Symbol not found

    def __contains__(self, name: str) -> bool:
        """Allows using 'if name in symbol_table:' (checks current scope and parents)."""
        return self.lookup_symbol(name) is not None

    def print_table(self, indent: int = 0) -> None:
        """Prints the symbol table and its children recursively in a formatted way."""
        prefix = "  " * indent
        print(f"{prefix}Symbol Table: '{self.scope_name}' (Level: {self.scope_level})")
        header = f"{prefix}| {'Name':<18} | {'GoX Type':<10} | {'Is Const?':<10} | {'Decl. Line':<10} |"
        print(header)
        print(f"{prefix}|{'-'*20}|{'-'*12}|{'-'*12}|{'-'*12}|")

        if not self.entries:
            print(f"{prefix}| {'(empty)':<59} |")
        else:
            for name, entry in self.entries.items():
                decl_line = str(entry.declaration_node.lineno) if entry.declaration_node and entry.declaration_node.lineno is not None else "N/A"
                print(f"{prefix}| {entry.name:<18} | {entry.gox_type:<10} | {'Yes' if entry.is_constant else 'No':<10} | {decl_line:<10} |")
        print(f"{prefix}{'-' * (len(header) - len(prefix))}\n")

        for child in self.child_scopes:
            child.print_table(indent + 1)

# Example Usage (for testing SymbolTable directly):
if __name__ == "__main__":
    global_scope = SymbolTable("global", scope_level=0)
    
    # Mock declaration nodes for example
    class MockNode:
        def __init__(self, lineno): self.lineno = lineno

    try:
        entry_x = SymbolEntry("x", "int", declaration_node=MockNode(5))
        global_scope.add_symbol(entry_x)

        entry_pi = SymbolEntry("PI", "float", declaration_node=MockNode(6), is_constant=True, value=3.14)
        global_scope.add_symbol(entry_pi)

        # entry_x_redefined = SymbolEntry("x", "float", declaration_node=MockNode(7))
        # global_scope.add_symbol(entry_x_redefined) # This would raise SymbolAlreadyDefinedError

    except SymbolTable.SymbolAlreadyDefinedError as e:
        print(f"Error: {e}")


    func_scope = SymbolTable("function_foo", parent_scope=global_scope, scope_level=1)
    entry_param_a = SymbolEntry("a", "int", declaration_node=MockNode(10))
    func_scope.add_symbol(entry_param_a)
    
    entry_local_y = SymbolEntry("y", "bool", declaration_node=MockNode(11))
    func_scope.add_symbol(entry_local_y)

    block_scope = SymbolTable("if_block", parent_scope=func_scope, scope_level=2)
    entry_block_z = SymbolEntry("z", "char", declaration_node=MockNode(15))
    block_scope.add_symbol(entry_block_z)


    print("--- Symbol Table Hierarchy ---")
    global_scope.print_table()

    print("\n--- Lookups ---")
    print(f"Lookup 'x' from global: {global_scope.lookup_symbol('x')}")
    print(f"Lookup 'a' from global: {global_scope.lookup_symbol('a')}") # Should be None
    print(f"Lookup 'a' from func_scope: {func_scope.lookup_symbol('a')}")
    print(f"Lookup 'x' from func_scope: {func_scope.lookup_symbol('x')}") # Found in global
    print(f"Lookup 'y' from block_scope: {block_scope.lookup_symbol('y')}") # Found in func_scope
    print(f"Lookup 'z' from block_scope: {block_scope.lookup_symbol('z')}") 
    print(f"Lookup 'PI' from block_scope (const): {block_scope.lookup_symbol('PI')}") 

    print(f"\n'x' in global_scope: {'x' in global_scope}")
    print(f"'z' in func_scope: {'z' in func_scope}") # False
    print(f"'z' in block_scope: {'z' in block_scope}") # True