# Symbol_Tab.py
class SymbolTab:
    
    class symbolDefinedError(Exception):
        pass

    class SymbolConflictError(Exception):
        pass

    def __init__(self, name, parent=None):
        self.name = name
        self.entries = {}
        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
        self.children = []
        
    def add(self, name, value):
        if name in self.entries:
            if self.entries[name].dtype != value.dtype:
                raise SymbolTab.SymbolConflictError()
            else:
                raise SymbolTab.symbolDefinedError()
        self.entries[name] = value

    def get(self, name):
        if name in self.entries:
            return self.entries[name]
        elif self.parent:
            return self.parent.get(name)
        return None

    def print(self):
        # Simple table print without rich
        print(f"Symbol Table: '{self.name}'")
        print(f"{'key':<20} {'value':<30}")
        print('-' * 50)
        for k, v in self.entries.items():
            value = f"{v.__class__.__name__}({v.name})"
            print(f"{k:<20} {value:<30}")
        print()
        for child in self.children:
            child.print()
    def __contains__(self, key):
        """Permite usar ‘if key in symtab:’."""
        return self.get(key) is not None