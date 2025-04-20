# Semantic_checking.py

from Types import typenames, check_binop, check_unaryop
from Nodes_AST import *
from Sym_tab import SymbolTab
from Error import ErrorHandler

class Checker:
    def __init__(self):
        # tabla de símbolos global
        self.env = SymbolTab("global")
        # recolector de errores semánticos
        self.errors = ErrorHandler()
        # pila para control de returns
        self.return_types: list[str | None] = []
        # nivel de anidamiento de bucles
        self.loop_level = 0

    def check(self, prog: Program):
        """Inicia el pase semántico."""
        prog.accept(self, self.env)
        return self.errors

    # Program
    def visit_Program(self, node: Program, env: SymbolTab):
        for stmt in node.statements:
            stmt.accept(self, env)

    # Declarations
    def visit_VariableDecl(self, node: VariableDecl, env: SymbolTab):
        if node.name in env.entries:
            self.errors.add_semantic_error(f"Variable '{node.name}' redeclarada",
                                          getattr(node, 'lineno', None))
            return
        # revisa inicialización
        if node.value:
            vtype = node.value.accept(self, env)
            if vtype != node.var_type:
                self.errors.add_semantic_error(
                    f"Inicializador de tipo '{vtype}' no coincide con declarado '{node.var_type}'",
                    getattr(node, 'lineno', None))
        env.add(node.name, node)

    def visit_ConstantDecl(self, node: ConstantDecl, env: SymbolTab):
        if node.name in env.entries:
            self.errors.add_semantic_error(f"Constante '{node.name}' redeclarada",
                                          getattr(node, 'lineno', None))
            return
        ctype = node.value.accept(self, env)
        env.add(node.name, node)

    def visit_FunctionDecl(self, node: FunctionDecl, env: SymbolTab):
        if node.name in env.entries:
            self.errors.add_semantic_error(f"Función '{node.name}' redeclarada",
                                          getattr(node, 'lineno', None))
            return
        env.add(node.name, node)
        # nuevo ámbito para la función
        fn_env = SymbolTab(node.name, parent=env)
        # registra parámetros
        for p in node.params:
            if p.name in fn_env.entries:
                self.errors.add_semantic_error(f"Parámetro '{p.name}' redeclarado",
                                              getattr(node, 'lineno', None))
            else:
                fn_env.add(p.name, p)
        # tipo de retorno
        self.return_types.append(node.return_type)
        # analiza cuerpo
        for stmt in node.body.statements:
            stmt.accept(self, fn_env)
        self.return_types.pop()

    def visit_Parameter(self, node: Parameter, env: SymbolTab):
        # el tipo del parámetro ya fue registrado en visit_FunctionDecl
        return node.param_type

    # Statements
    def visit_Assignment(self, node: Assignment, env: SymbolTab):
        # variable declarada?
        if not env.get(node.location.name):
            self.errors.add_semantic_error(
                f"Asignación a no declarada '{node.location.name}'",
                getattr(node, 'lineno', None))
            return None
        expected = None
        sym = env.get(node.location.name)
        if isinstance(sym, VariableDecl):
            expected = sym.var_type
        elif isinstance(sym, ConstantDecl):
            expected = sym.value.accept(self, env)
        actual = node.expr.accept(self, env)
        if expected and actual != expected:
            self.errors.add_semantic_error(
                f"No se puede asignar '{actual}' a '{expected}'",
                getattr(node, 'lineno', None))
        return expected

    def visit_Print(self, node: Print, env: SymbolTab):
        node.expr.accept(self, env)

    def visit_If(self, node: If, env: SymbolTab):
        cond = node.test.accept(self, env)
        if cond != 'bool':
            self.errors.add_semantic_error(
                f"Condición de if debe ser 'bool', no '{cond}'",
                getattr(node, 'lineno', None))
        # blocks
        for stmt in node.consequence.statements:
            stmt.accept(self, env)
        if node.alternative:
            for stmt in node.alternative.statements:
                stmt.accept(self, env)

    def visit_While(self, node: While, env: SymbolTab):
        cond = node.test.accept(self, env)
        if cond != 'bool':
            self.errors.add_semantic_error(
                f"Condición de while debe ser 'bool', no '{cond}'",
                getattr(node, 'lineno', None))
        self.loop_level += 1
        for stmt in node.body.statements:
            stmt.accept(self, env)
        self.loop_level -= 1

    def visit_Break(self, node: Break, env: SymbolTab):
        if self.loop_level == 0:
            self.errors.add_semantic_error("Break fuera de bucle",
                                          getattr(node, 'lineno', None))

    def visit_Continue(self, node: Continue, env: SymbolTab):
        if self.loop_level == 0:
            self.errors.add_semantic_error("Continue fuera de bucle",
                                          getattr(node, 'lineno', None))

    def visit_Return(self, node: Return, env: SymbolTab):
        actual = node.expr.accept(self, env)
        expected = self.return_types[-1] if self.return_types else None
        if expected and actual != expected:
            self.errors.add_semantic_error(
                f"Return de tipo '{actual}' no coincide con '{expected}'",
                getattr(node, 'lineno', None))

    # Expressions
    def visit_Integer(self, node: Integer, env: SymbolTab):
        return 'int'

    def visit_Float(self, node: Float, env: SymbolTab):
        return 'float'

    def visit_String(self, node: String, env: SymbolTab):
        return 'string'

    def visit_Char(self, node: Char, env: SymbolTab):
        return 'char'

    def visit_Boolean(self, node: Boolean, env: SymbolTab):
        return 'bool'

    def visit_BinOp(self, node: BinOp, env: SymbolTab):
        l = node.left.accept(self, env)
        r = node.right.accept(self, env)
        res = check_binop(node.op, l, r)
        if res is None:
            self.errors.add_semantic_error(
                f"Operación binaria inválida: {l} {node.op} {r}",
                getattr(node, 'lineno', None))
        return res

    def visit_UnaryOp(self, node: UnaryOp, env: SymbolTab):
        v = node.operand.accept(self, env)
        res = check_unaryop(node.op, v)
        if res is None:
            self.errors.add_semantic_error(
                f"Operación unaria inválida: {node.op}{v}",
                getattr(node, 'lineno', None))
        return res

    def visit_TypeCast(self, node: TypeCast, env: SymbolTab):
        v = node.expr.accept(self, env)
        if node.target_type not in typenames:
            self.errors.add_semantic_error(
                f"Casteo a tipo desconocido '{node.target_type}'",
                getattr(node, 'lineno', None))
        return node.target_type

    def visit_FunctionCall(self, node: FunctionCall, env: SymbolTab):
        fn = env.get(node.name)
        if not fn or not isinstance(fn, FunctionDecl):
            self.errors.add_semantic_error(
                f"Llamada a función no definida '{node.name}'",
                getattr(node, 'lineno', None))
            return None
        # chequea args vs params
        if len(node.args) != len(fn.params):
            self.errors.add_semantic_error(
                f"Aridad incorrecta en '{node.name}' ({len(node.args)} vs {len(fn.params)})",
                getattr(node, 'lineno', None))
        for arg, param in zip(node.args, fn.params):
            at = arg.accept(self, env)
            if at != param.param_type:
                self.errors.add_semantic_error(
                    f"Arg '{param.name}' en '{node.name}' espera '{param.param_type}', recibe '{at}'",
                    getattr(node, 'lineno', None))
        return fn.return_type

    def visit_Location(self, node: Location, env: SymbolTab):
        sym = env.get(node.name)
        if not sym:
            self.errors.add_semantic_error(
                f"Uso de variable no declarada '{node.name}'",
                getattr(node, 'lineno', None))
            return None
        if isinstance(sym, VariableDecl):
            return sym.var_type
        if isinstance(sym, ConstantDecl):
            return sym.value.accept(self, env)
        return None

    # caída por defecto
    def generic_visit(self, node, env):
        # nodos no tipados o sin acción semántica
        return None

